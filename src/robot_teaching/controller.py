"""Teach controller: one MIT control callback with a small state machine.

Modes
-----
``DISABLED``   motors off (after an e-stop or before start).
``HOLD``       stiff position hold with gravity feed-forward. Jog commands move
               the hold target toward a goal at a limited speed.
``FREE_DRIVE`` gravity-compensated compliance for drag teaching, with an
               end-effector velocity lock: LOCKED while still (low PD + integral
               on the locked target), FOLLOW while being pushed (target follows
               the measured position).
``PLAYBACK``   streams a planned :class:`Trajectory` as ``(q, q̇)`` setpoints
               with gravity feed-forward; a speed scale slews smoothly, and a
               stop ramps the speed to zero before returning to HOLD.

Gain changes between modes are faded with a smoothstep so stiffness never
jumps. Every error path falls back to HOLD at the measured position; motors are
disabled only by an explicit e-stop.

Threading: API commands run synchronously under the controller lock, which
the tick also holds, so a command sees consistent state and returns a result
immediately. Timers run on the controller's own clock (the sum of tick ``dt``),
so the controller is deterministic when a simulator steps it by hand.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .backend.base import ArmBackend
from .config import TeachingConfig, resolve_vector
from .model import Pose, RobotModel
from .planning import Trajectory


class Mode(str, Enum):
    DISABLED = "disabled"
    HOLD = "hold"
    FREE_DRIVE = "free_drive"
    PLAYBACK = "playback"


@dataclass(frozen=True)
class Snapshot:
    time: float
    mode: str
    enabled: bool
    q: np.ndarray
    qd: np.ndarray
    q_target: np.ndarray
    pose: Pose | None
    gripper: float | None
    gripper_target: float | None
    tau_g: np.ndarray
    ee_speed: tuple[float, float]
    free_drive_status: str
    gains_settled: bool
    jogging: bool
    playback: dict | None
    error: str | None
    tick_dt: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "mode": self.mode,
            "enabled": self.enabled,
            "q": self.q.tolist(),
            "qd": self.qd.tolist(),
            "q_target": self.q_target.tolist(),
            "pose": self.pose.to_dict() if self.pose is not None else None,
            "gripper": self.gripper,
            "gripper_target": self.gripper_target,
            "tau_g": self.tau_g.tolist(),
            "ee_speed": list(self.ee_speed),
            "free_drive_status": self.free_drive_status,
            "gains_settled": self.gains_settled,
            "jogging": self.jogging,
            "playback": self.playback,
            "error": self.error,
            "tick_dt": self.tick_dt,
        }


@dataclass
class _Playback:
    traj: Trajectory
    s: float = 0.0
    speed: float = 1.0
    speed_target: float = 1.0
    loop: bool = False
    stopping: bool = False
    event_idx: int = 0
    point_index: int = -1
    laps: int = 0
    q_cmd: np.ndarray = field(default_factory=lambda: np.zeros(0))


class TeachController:
    STOP_RAMP = 0.3        # s to ramp the playback speed to zero on stop
    SPEED_SLEW = 2.0       # 1/s, max rate of change of the playback speed scale
    MAX_START_GAP = 0.05   # rad, allowed gap between the arm and a trajectory's first sample

    def __init__(self, backend: ArmBackend, cfg: TeachingConfig, model: RobotModel | None = None) -> None:
        self.backend = backend
        self.cfg = cfg
        self.model = model or RobotModel(backend.n_arm)
        self.n = backend.n_arm
        n = self.n
        fd = cfg.free_drive
        self.rate = float(cfg.control.rate or backend.rate)

        self._tau_scale = resolve_vector(fd.tau_scale, n, "free_drive.tau_scale")
        self._free_kp = np.full(n, float(fd.kp))
        self._free_kd = np.full(n, float(fd.kd))
        hkp, hkd = backend.hold_gains()
        self._hold_kp = resolve_vector(cfg.hold.kp, n, "hold.kp") if cfg.hold.kp is not None else np.asarray(hkp, float)
        self._hold_kd = resolve_vector(cfg.hold.kd, n, "hold.kd") if cfg.hold.kd is not None else np.asarray(hkd, float)
        gkp, gkd = backend.gripper_gains()
        self._gripper_kp = float(cfg.gripper.kp) if cfg.gripper.kp is not None else gkp
        self._gripper_kd = float(cfg.gripper.kd) if cfg.gripper.kd is not None else gkd
        self._margin = float(cfg.limits.joint_position_margin)
        self._jog_vel = float(cfg.jog.joint_velocity)

        # Control-thread state.
        self._mode = Mode.DISABLED
        self._q = np.zeros(n)
        self._qd = np.zeros(n)
        self._q_prev: np.ndarray | None = None
        self._clock = 0.0
        self._q_target = np.zeros(n)
        self._q_goal = np.zeros(n)
        self._gripper: float | None = None
        self._gripper_target: float | None = None
        self._tau_g = np.zeros(n)
        self._integral = np.zeros(n)
        self._kp = self._hold_kp.copy()
        self._kd = self._hold_kd.copy()
        self._gain_from = (self._hold_kp.copy(), self._hold_kd.copy())
        self._gain_to = (self._hold_kp.copy(), self._hold_kd.copy())
        self._gain_t0: float | None = None
        self._gain_T = float(fd.transition_duration)
        self._fd_following = False
        self._fd_quiet_since: float | None = None
        self._fd_status = ""
        self._fd_q_hold: np.ndarray | None = None
        self._ee_speed = (0.0, 0.0)
        self._pb: _Playback | None = None
        self._error: str | None = None
        self._enabled = False
        self._started = False

        self._lock = threading.RLock()
        self._snap_lock = threading.Lock()
        self._snapshot: Snapshot | None = None
        self._ticks = 0
        self._last_dt = 1.0 / self.rate

    # ══════════════════════════════════════════════════════════════════════
    # lifecycle (API thread)
    # ══════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        if self._started:
            return
        self.backend.connect()
        self.backend.enable()
        q, g = self.backend.read_positions()
        self._q = np.asarray(q, float)[: self.n].copy()
        self._q_target = self._q.copy()
        self._q_goal = self._q.copy()
        self._gripper = g
        self._gripper_target = g
        self._tau_g = self._gravity(self._q)
        self._set_gains(self._hold_kp, self._hold_kd, 0.0)
        self._enabled = True
        self._mode = Mode.HOLD
        self._publish(0.0)
        self.backend.start_loop(self._tick, self.rate)
        self._started = True

    def shutdown(self, disable: bool = False) -> None:
        """Stop the loop, leaving the motors holding (or disabled when asked)."""
        if not self._started:
            return
        self.backend.stop_loop()
        if self._enabled:
            try:
                if disable:
                    self.backend.disable()
                else:
                    tau = self._gravity(self._q)
                    self.backend.send_arm_mit(self._q_target, np.zeros(self.n), self._hold_kp, self._hold_kd, tau)
            except Exception:
                pass
        self.backend.disconnect()
        self._started = False

    # ══════════════════════════════════════════════════════════════════════
    # public commands (API thread)
    # ══════════════════════════════════════════════════════════════════════

    def snapshot(self) -> Snapshot | None:
        with self._snap_lock:
            return self._snapshot

    def _send(self, kind: str, payload: Any = None) -> tuple[bool, str]:
        with self._lock:
            try:
                return self._handle(kind, payload)
            except Exception as e:  # a bad command must never kill the loop
                return False, f"{type(e).__name__}: {e}"
            finally:
                # A state read right after a command must already reflect it.
                self._publish(self._last_dt)

    def hold(self) -> tuple[bool, str]:
        return self._send("hold")

    def free_drive(self) -> tuple[bool, str]:
        return self._send("free_drive")

    def stop_motion(self) -> tuple[bool, str]:
        return self._send("stop")

    def estop(self) -> tuple[bool, str]:
        return self._send("estop")

    def enable(self) -> tuple[bool, str]:
        return self._send("enable")

    def set_gripper(self, position: float) -> tuple[bool, str]:
        return self._send("gripper", float(position))

    def jog_to(self, q_goal: np.ndarray) -> tuple[bool, str]:
        q_goal = np.asarray(q_goal, float)[: self.n]
        if not self.model.within_limits(q_goal, self._margin):
            return False, "goal outside joint limits"
        return self._send("jog_goal", q_goal.copy())

    def play(self, traj: Trajectory, speed: float, loop: bool = False) -> tuple[bool, str]:
        if not (0.0 < speed <= 1.0):
            return False, "speed must be in (0, 1]"
        return self._send("play", (traj, float(speed), bool(loop)))

    def set_speed(self, speed: float) -> tuple[bool, str]:
        if not (0.0 < speed <= 1.0):
            return False, "speed must be in (0, 1]"
        return self._send("speed", float(speed))

    # ══════════════════════════════════════════════════════════════════════
    # control thread
    # ══════════════════════════════════════════════════════════════════════

    def _gravity(self, q: np.ndarray) -> np.ndarray:
        return self._tau_scale * self.model.gravity(q)

    def _set_gains(self, kp: np.ndarray, kd: np.ndarray, duration: float) -> None:
        self._gain_from = (self._kp.copy(), self._kd.copy())
        self._gain_to = (np.asarray(kp, float).copy(), np.asarray(kd, float).copy())
        self._gain_T = max(0.0, float(duration))
        self._gain_t0 = self._clock if self._gain_T > 0.0 else None
        if self._gain_T == 0.0:
            self._kp, self._kd = self._gain_to[0].copy(), self._gain_to[1].copy()

    def _gain_blend(self) -> float:
        if self._gain_t0 is None:
            return 1.0
        r = min(1.0, max(0.0, (self._clock - self._gain_t0) / max(self._gain_T, 1e-6)))
        b = r * r * (3.0 - 2.0 * r)
        self._kp = self._gain_from[0] + b * (self._gain_to[0] - self._gain_from[0])
        self._kd = self._gain_from[1] + b * (self._gain_to[1] - self._gain_from[1])
        if r >= 1.0:
            self._gain_t0 = None
        return b

    def _estimate_velocity(self, q: np.ndarray, dt: float) -> np.ndarray:
        fd = self.cfg.free_drive
        if self._q_prev is None:
            self._q_prev = q.copy()
            self._qd = np.zeros(self.n)
            return self._qd
        q_prev, self._q_prev = self._q_prev, q.copy()
        if dt <= 0.0 or dt > fd.max_valid_sample_dt:
            self._qd = np.zeros(self.n)
            return self._qd
        raw = (q - q_prev) / dt
        alpha = 1.0 - math.exp(-dt / max(fd.velocity_filter_time_constant, 1e-6))
        self._qd = self._qd + alpha * (raw - self._qd)
        return self._qd

    # ── mode transitions ──────────────────────────────────────────────────

    def _enter_hold(self, q_target: np.ndarray, fade: float) -> None:
        self._mode = Mode.HOLD
        self._q_target = np.asarray(q_target, float).copy()
        self._q_goal = self._q_target.copy()
        self._pb = None
        self._fd_status = ""
        self._integral[:] = 0.0
        self._set_gains(self._hold_kp, self._hold_kd, fade)

    def _enter_free_drive(self) -> None:
        self._mode = Mode.FREE_DRIVE
        self._pb = None
        self._fd_q_hold = self._q.copy()
        self._q_target = self._q.copy()
        self._integral[:] = 0.0
        self._fd_following = False
        self._fd_quiet_since = None
        self._fd_status = "STARTUP"
        self._set_gains(self._free_kp, self._free_kd, self.cfg.free_drive.transition_duration)

    # ── command handling ──────────────────────────────────────────────────

    def _handle(self, k: str, payload: Any) -> tuple[bool, str]:
        if k == "estop":
            self.backend.disable()
            self._enabled = False
            self._mode = Mode.DISABLED
            self._pb = None
            self._fd_status = ""
            return True, "motors disabled"
        if k == "enable":
            if self._enabled:
                return True, "already enabled"
            self.backend.enable()
            self._enabled = True
            self._error = None
            self._enter_hold(self._q, 0.0)
            return True, "enabled, holding"
        if not self._enabled:
            return False, "motors are disabled; enable first"
        if k == "hold":
            if self._mode == Mode.PLAYBACK:
                return self._handle("stop", None)
            if self._mode == Mode.FREE_DRIVE:
                self._enter_hold(self._q, self.cfg.free_drive.transition_duration)
            return True, "hold"
        if k == "free_drive":
            if self._mode == Mode.PLAYBACK:
                return False, "stop playback first"
            if self._mode != Mode.FREE_DRIVE:
                self._enter_free_drive()
            return True, "free drive"
        if k == "stop":
            if self._mode == Mode.PLAYBACK and self._pb is not None:
                self._pb.stopping = True
                self._pb.speed_target = 0.0
                return True, "stopping"
            if self._mode == Mode.HOLD:
                self._q_goal = self._q_target.copy()
                return True, "stopped"
            return True, "free drive: nothing commanded to stop (press Hold to stiffen)"
        if k == "gripper":
            self._gripper_target = float(payload)
            return True, "gripper target set"
        if k == "jog_goal":
            if self._mode != Mode.HOLD:
                return False, "jogging is only available in hold mode"
            if self._gain_t0 is not None:
                return False, "wait for the hold gains to settle"
            self._q_goal = np.asarray(payload, float).copy()
            return True, "jogging"
        if k == "play":
            traj, speed, loop = payload
            if self._mode != Mode.HOLD:
                return False, "playback can only start from hold mode"
            if self._gain_t0 is not None:
                return False, "wait for the hold gains to settle"
            if np.max(np.abs(self._q_goal - self._q_target)) > 1e-6:
                return False, "a jog is still in progress"
            if traj.n_joints != self.n:
                return False, "trajectory joint count mismatch"
            gap = float(np.max(np.abs(traj.q[0] - self._q)))
            if gap > self.MAX_START_GAP:
                return False, f"trajectory does not start at the current position (gap {gap:.3f} rad)"
            self._pb = _Playback(traj=traj, speed=0.0, speed_target=speed, loop=loop, q_cmd=traj.q[0].copy())
            self._mode = Mode.PLAYBACK
            self._fd_status = ""
            return True, "playing"
        if k == "speed":
            if self._pb is None:
                return False, "not playing"
            if not self._pb.stopping:
                self._pb.speed_target = float(payload)
            return True, "speed set"
        return False, f"unknown command {k!r}"

    # ── mode updates ──────────────────────────────────────────────────────

    def _update_hold(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        step = self._jog_vel * dt
        delta = np.clip(self._q_goal - self._q_target, -step, step)
        self._q_target = self._q_target + delta
        return self._q_target, np.zeros(self.n)

    def _update_free_drive(self, dt: float, blend: float) -> tuple[np.ndarray, np.ndarray]:
        fd = self.cfg.free_drive
        q = self._q
        now = self._clock
        if blend < 1.0:
            # Fade from the stiff hold into compliance while the target drifts to the measured pose.
            q_hold = self._fd_q_hold if self._fd_q_hold is not None else q
            self._q_target = q_hold + blend * (q - q_hold)
            self._integral[:] = 0.0
            self._fd_following = False
            self._fd_quiet_since = None
            self._fd_status = "STARTUP"
            return self._q_target, np.zeros(self.n)

        self._fd_q_hold = None
        lin, ang = self._ee_speed
        release = lin > fd.linear_release_threshold or ang > fd.angular_release_threshold
        quiet = lin < fd.linear_lock_threshold and ang < fd.angular_lock_threshold
        if not self._fd_following and release:
            self._fd_following = True
            self._fd_quiet_since = None
            self._integral[:] = 0.0
        if self._fd_following:
            self._q_target = q.copy()
            self._integral[:] = 0.0
            if quiet:
                if self._fd_quiet_since is None:
                    self._fd_quiet_since = now
                elif now - self._fd_quiet_since >= fd.lock_settle_duration:
                    self._fd_following = False
                    self._fd_quiet_since = None
                    self._q_target = q.copy()
            else:
                self._fd_quiet_since = None
        else:
            self._integral += fd.ki * (self._q_target - q) * min(max(dt, 0.0), 0.02)
            np.clip(self._integral, -fd.integral_limit, fd.integral_limit, out=self._integral)
        self._fd_status = "FOLLOW" if self._fd_following else "LOCKED"
        return self._q_target, np.zeros(self.n)

    def _update_playback(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        pb = self._pb
        assert pb is not None
        # Slew the speed scale so velocity never jumps.
        if pb.stopping:
            pb.speed = max(0.0, pb.speed - dt / self.STOP_RAMP)
        else:
            dv = np.clip(pb.speed_target - pb.speed, -self.SPEED_SLEW * dt, self.SPEED_SLEW * dt)
            pb.speed += float(dv)
        pb.s += dt * pb.speed
        traj = pb.traj
        # Fire events crossed since the last tick.
        while pb.event_idx < len(traj.events) and traj.events[pb.event_idx].time <= pb.s:
            ev = traj.events[pb.event_idx]
            if ev.kind == "gripper":
                self._gripper_target = float(ev.value)
            elif ev.kind == "point":
                pb.point_index = int(ev.value)
            pb.event_idx += 1
        if pb.s >= traj.duration:
            if pb.loop and not pb.stopping:
                pb.laps += 1
                pb.s = traj.loop_start_time + (pb.s - traj.duration)
                pb.event_idx = int(np.searchsorted([e.time for e in traj.events], pb.s, side="left"))
            else:
                q_end = traj.q[-1].copy()
                self._enter_hold(q_end, 0.0)
                return q_end, np.zeros(self.n)
        q_cmd, qd_cmd = traj.sample(pb.s)
        pb.q_cmd = q_cmd
        self._q_target = q_cmd
        if pb.stopping and pb.speed <= 0.0:
            self._enter_hold(q_cmd, 0.0)
            return q_cmd, np.zeros(self.n)
        return q_cmd, qd_cmd * pb.speed

    # ── the tick ──────────────────────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        with self._lock:
            self._clock += dt
            self._last_dt = dt
            try:
                q, g = self.backend.read_positions()
                self._q = np.asarray(q, float)[: self.n]
                self._gripper = g
                self._estimate_velocity(self._q, dt)
                self._tau_g = self._gravity(self._q)
                if self._mode == Mode.FREE_DRIVE:
                    self._ee_speed = self.model.ee_velocity(self._q, self._qd)
                else:
                    self._ee_speed = (0.0, 0.0)
                blend = self._gain_blend()

                if self._mode == Mode.DISABLED:
                    self._publish(dt)
                    return
                if self._mode == Mode.HOLD:
                    q_cmd, qd_cmd = self._update_hold(dt)
                    tau = self._tau_g
                elif self._mode == Mode.FREE_DRIVE:
                    q_cmd, qd_cmd = self._update_free_drive(dt, blend)
                    tau = self._tau_g + self._integral
                else:
                    q_cmd, qd_cmd = self._update_playback(dt)
                    tau = self._tau_g
                self.backend.send_arm_mit(q_cmd, qd_cmd, self._kp, self._kd, tau)
                if self.backend.has_gripper and self._gripper_target is not None:
                    self.backend.send_gripper_mit(self._gripper_target, self._gripper_kp, self._gripper_kd)
            except Exception as e:
                # Fail safe: stiff hold where the arm is, report, keep running.
                self._error = f"{type(e).__name__}: {e}"
                try:
                    if self._enabled:
                        self._enter_hold(self._q, 0.0)
                        self.backend.send_arm_mit(self._q_target, np.zeros(self.n), self._hold_kp, self._hold_kd, self._tau_g)
                except Exception:
                    pass
            self._ticks += 1
            self._publish(dt)

    def _publish(self, dt: float) -> None:
        pb = self._pb
        playback = None
        if pb is not None:
            playback = {
                "elapsed": float(pb.s),
                "duration": float(pb.traj.duration),
                "progress": float(min(1.0, pb.s / pb.traj.duration)) if pb.traj.duration > 0 else 1.0,
                "speed": float(pb.speed),
                "speed_target": float(pb.speed_target),
                "loop": pb.loop,
                "laps": pb.laps,
                "point_index": pb.point_index,
                "stopping": pb.stopping,
            }
        try:
            pose = self.model.pose(self._q)
        except Exception:
            pose = None
        snap = Snapshot(
            time=time.time(),
            mode=self._mode.value,
            enabled=self._enabled,
            q=self._q.copy(),
            qd=self._qd.copy(),
            q_target=self._q_target.copy(),
            pose=pose,
            gripper=self._gripper,
            gripper_target=self._gripper_target,
            tau_g=self._tau_g.copy(),
            ee_speed=self._ee_speed,
            free_drive_status=self._fd_status,
            gains_settled=self._gain_t0 is None,
            jogging=bool(self._mode == Mode.HOLD and np.max(np.abs(self._q_goal - self._q_target)) > 1e-9),
            playback=playback,
            error=self._error,
            tick_dt=float(dt),
        )
        with self._snap_lock:
            self._snapshot = snap
