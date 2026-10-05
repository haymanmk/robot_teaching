"""Simulated arm with the same interface as the hardware adapter.

Each joint is a rigid body with the URDF inertia (diagonal of the mass matrix),
the URDF gravity torque, viscous + Coulomb friction, and an MIT impedance motor
``tau = kp (q_cmd - q) + kd (qd_cmd - qd) + tau_ff`` clipped to the URDF effort
limit. ``gravity_scale`` ≠ 1 emulates a model error; ``push()`` applies an
external joint torque to emulate the operator's hand in free-drive.

``realtime=True`` runs a thread at the control rate, like the hardware.
``realtime=False`` lets tests advance the simulation with :meth:`step`.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from ..model import RobotModel
from .base import ArmBackend, BackendError

_DEFAULT_HOLD_KP = np.array([50.0, 150.0, 150.0, 50.0, 50.0, 50.0])
_DEFAULT_HOLD_KD = np.array([3.0, 10.0, 10.0, 5.0, 4.0, 4.0])
_DEFAULT_EFFORT = np.array([36.0, 36.0, 36.0, 14.0, 14.0, 14.0])


class SimBackend(ArmBackend):
    name = "sim"

    def __init__(
        self,
        n_arm: int = 6,
        has_gripper: bool = True,
        rate: float = 500.0,
        realtime: bool = True,
        gravity_scale: float = 1.0,
        viscous_friction: float = 0.2,
        coulomb_friction: float = 0.3,     # N·m, in the range measured on the RS build
        q0: np.ndarray | None = None,
        model: RobotModel | None = None,
    ) -> None:
        self._model = model or RobotModel(n_arm)
        self._n = self._model.n
        self._has_gripper = bool(has_gripper)
        self._rate = float(rate)
        self._realtime = bool(realtime)
        self.gravity_scale = float(gravity_scale)
        self.viscous = float(viscous_friction)
        self.coulomb = float(coulomb_friction)

        self._q = np.zeros(self._n) if q0 is None else np.asarray(q0, dtype=float)[: self._n].copy()
        self._qd = np.zeros(self._n)
        self._gripper = 0.0
        self._cmd = None                       # (pos, vel, kp, kd, tau)
        self._gripper_cmd = None               # (pos, kp, kd)
        self._tau_ext = np.zeros(self._n)
        self._push_until = 0.0
        self._sim_time = 0.0
        self._enabled = False
        self._connected = False
        self._lock = threading.Lock()

        self._cb: Callable[[float], None] | None = None
        self._loop_rate = self._rate
        self._thread: threading.Thread | None = None
        self._running = False
        self.effort = _DEFAULT_EFFORT[: self._n].copy()
        self.ticks = 0

    # ── description ──────────────────────────────────────────────────────
    @property
    def n_arm(self) -> int:
        return self._n

    @property
    def has_gripper(self) -> bool:
        return self._has_gripper

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def joint_names(self) -> list[str]:
        return list(self._model.joint_names)

    @property
    def sim_time(self) -> float:
        return self._sim_time

    # ── lifecycle ─────────────────────────────────────────────────────────
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self.stop_loop()
        self._enabled = False
        self._connected = False

    def enable(self) -> None:
        if not self._connected:
            raise BackendError("connect() first")
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        self._cmd = None
        self._gripper_cmd = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── I/O ───────────────────────────────────────────────────────────────
    def read_positions(self) -> tuple[np.ndarray, float | None]:
        with self._lock:
            return self._q.copy(), (self._gripper if self._has_gripper else None)

    def read_velocities(self) -> np.ndarray:
        with self._lock:
            return self._qd.copy()

    def send_arm_mit(self, q, qd, kp, kd, tau) -> None:
        self._cmd = tuple(np.asarray(x, dtype=float).reshape(-1)[: self._n].copy() for x in (q, qd, kp, kd, tau))

    def send_gripper_mit(self, pos: float, kp: float | None = None, kd: float | None = None) -> None:
        self._gripper_cmd = (float(pos), kp, kd)

    # ── test hooks ────────────────────────────────────────────────────────
    def push(self, tau_ext: np.ndarray, duration: float) -> None:
        """Apply an external joint torque for ``duration`` seconds of simulated time."""
        with self._lock:
            self._tau_ext = np.asarray(tau_ext, dtype=float)[: self._n].copy()
            self._push_until = self._sim_time + float(duration)

    def set_state(self, q: np.ndarray, qd: np.ndarray | None = None, gripper: float | None = None) -> None:
        with self._lock:
            self._q = np.asarray(q, dtype=float)[: self._n].copy()
            self._qd = np.zeros(self._n) if qd is None else np.asarray(qd, dtype=float)[: self._n].copy()
            if gripper is not None:
                self._gripper = float(gripper)

    # ── physics ───────────────────────────────────────────────────────────
    def _physics(self, dt: float) -> None:
        with self._lock:
            q, qd = self._q, self._qd
            # Motor torque split into an explicit part and a damping coefficient so the
            # damping can be integrated implicitly (stable for any kd at this step size).
            tau_explicit = np.zeros(self._n)
            damping = np.full(self._n, self.viscous)
            if self._enabled and self._cmd is not None:
                pos, vel, kp, kd, tau = self._cmd
                tau_explicit = np.clip(kp * (pos - q) + kd * vel + tau, -self.effort, self.effort)
                damping = damping + kd
            g = self.gravity_scale * self._model.gravity(q)
            coulomb = self.coulomb * np.tanh(qd / 0.05)
            tau_ext = self._tau_ext if self._sim_time < self._push_until else 0.0
            M = np.maximum(self._model.mass_matrix_diag(q), 0.01)
            a_explicit = (tau_explicit - g - coulomb + tau_ext) / M
            # Semi-implicit Euler, two substeps, implicit damping.
            h = dt / 2.0
            for _ in range(2):
                qd = (qd + a_explicit * h) / (1.0 + damping * h / M)
                q = q + qd * h
            below, above = q < self._model.lower, q > self._model.upper
            q = np.clip(q, self._model.lower, self._model.upper)
            qd = np.where(below | above, 0.0, qd)
            self._q, self._qd = q, qd

            if self._has_gripper and self._enabled and self._gripper_cmd is not None:
                target, kp, _ = self._gripper_cmd
                if kp is None or kp > 0.0:
                    max_step = 10.0 * dt          # rad/s gripper speed
                    delta = np.clip(target - self._gripper, -max_step, max_step)
                    self._gripper += delta
            self._sim_time += dt

    def step(self, n: int = 1, dt: float | None = None) -> None:
        """Advance ``n`` ticks: controller callback, then physics (same order as hardware)."""
        dt = 1.0 / self._loop_rate if dt is None else dt
        for _ in range(n):
            if self._cb is not None:
                self._cb(dt)
            self._physics(dt)
            self.ticks += 1

    # ── loop ──────────────────────────────────────────────────────────────
    def start_loop(self, callback: Callable[[float], None], rate: float) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise BackendError("loop already running")
        self._cb = callback
        self._loop_rate = float(rate)
        if not self._realtime:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="sim-control-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / self._loop_rate
        last = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            dt = min(max(now - last, 1e-4), 0.05)
            last = now
            self.step(1, dt)
            sleep = period - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)

    def stop_loop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        self._cb = None

    # ── gains ─────────────────────────────────────────────────────────────
    def hold_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return _DEFAULT_HOLD_KP[: self._n].copy(), _DEFAULT_HOLD_KD[: self._n].copy()

    def gripper_gains(self) -> tuple[float, float]:
        return 50.0, 4.0

    def diagnostics(self) -> dict:
        return {"sim_time": self._sim_time, "ticks": self.ticks, "enabled": self._enabled}
