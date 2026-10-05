"""Hardware adapter over ``reBotArm_control_py.actuator.RebotArm``.

Motor ids, CAN channel, gains and the control rate come from the upstream
hardware YAML (``config/rebotarm_rs.yaml`` by default). The arm and gripper
groups are both driven in MIT mode.

RS firmware notes (from the upstream calibration): positions must be requested
explicitly every cycle (``get_positions`` does so) and the reported velocity is
not in rad/s, so the controller estimates velocity by finite differences.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .base import ArmBackend, BackendError


class RebotArmBackend(ArmBackend):
    name = "rebotarm"

    def __init__(self, hw_yaml: str | None = None) -> None:
        try:
            from reBotArm_control_py.actuator import RebotArm
        except ImportError as e:  # motorbridge missing, most likely
            raise BackendError(
                "reBotArm_control_py.actuator could not be imported; install the hardware extra "
                "(uv sync --extra hardware) and check the submodule"
            ) from e
        self.robot = RebotArm(hw_yaml)
        if "arm" not in self.robot.groups:
            raise BackendError("the hardware YAML has no 'arm' group")
        self._callback: Callable[[float], None] | None = None

    # ── description ──────────────────────────────────────────────────────
    @property
    def n_arm(self) -> int:
        return self.robot.arm.num_joints

    @property
    def has_gripper(self) -> bool:
        return self.robot.has_gripper

    @property
    def rate(self) -> float:
        return float(self.robot.rate)

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot.arm.joint_names)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def connect(self) -> None:
        self.robot.connect()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def enable(self) -> None:
        self.robot.arm.mode_mit()
        if self.has_gripper:
            self.robot.gripper.mode_mit()
        self.robot.enable_all()

    def disable(self) -> None:
        self.robot.disable_all()

    # ── I/O ───────────────────────────────────────────────────────────────
    def read_positions(self) -> tuple[np.ndarray, float | None]:
        q = np.asarray(self.robot.arm.get_positions(), dtype=float)
        g = None
        if self.has_gripper:
            gp = self.robot.gripper.get_positions()
            g = float(gp[0]) if len(gp) else None
        return q, g

    def send_arm_mit(self, q, qd, kp, kd, tau) -> None:
        self.robot.arm.send_mit(pos=q, vel=qd, kp=kp, kd=kd, tau=tau)

    def send_gripper_mit(self, pos: float, kp: float | None = None, kd: float | None = None) -> None:
        if not self.has_gripper:
            return
        self.robot.gripper.send_mit(
            np.array([float(pos)]),
            kp=None if kp is None else np.array([float(kp)]),
            kd=None if kd is None else np.array([float(kd)]),
        )

    # ── loop ──────────────────────────────────────────────────────────────
    def start_loop(self, callback: Callable[[float], None], rate: float) -> None:
        # Upstream passes the nominal period; measure the real one (clamped) so the
        # controller's timers and velocity estimate stay honest when a tick overruns.
        period = 1.0 / float(rate)
        last: list[float | None] = [None]

        def wrapped(_robot, nominal_dt: float) -> None:
            now = time.perf_counter()
            dt = nominal_dt if last[0] is None else min(max(now - last[0], 0.2 * period), 5.0 * period)
            last[0] = now
            callback(dt)

        self._callback = callback
        self.robot.start_control_loop(wrapped, rate=rate)

    def stop_loop(self) -> None:
        self.robot.stop_control_loop()
        self._callback = None

    # ── gains ─────────────────────────────────────────────────────────────
    def hold_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.robot.arm._mit_kp, dtype=float).copy(), np.array(self.robot.arm._mit_kd, dtype=float).copy()

    def gripper_gains(self) -> tuple[float, float]:
        if not self.has_gripper:
            return 0.0, 0.0
        return float(self.robot.gripper._mit_kp[0]), float(self.robot.gripper._mit_kd[0])

    def diagnostics(self) -> dict:
        out = {}
        for name, motor in getattr(self.robot, "_motor_map", {}).items():
            try:
                st = motor.get_state()
            except Exception:
                st = None
            if st is not None:
                out[name] = {"status": st.status_code, "t_mos": st.t_mos, "t_rotor": st.t_rotor, "torque": st.torq}
        return out
