"""Backend interface the teach controller talks to.

Everything is MIT (impedance) mode: the host streams ``pos, vel, kp, kd, tau``
setpoints and the motors close the loop. The backend owns the periodic thread
that calls the controller callback and must call it with the real elapsed time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class BackendError(RuntimeError):
    pass


class ArmBackend(ABC):
    name: str = "base"

    # ── static description ────────────────────────────────────────────────
    @property
    @abstractmethod
    def n_arm(self) -> int: ...

    @property
    @abstractmethod
    def has_gripper(self) -> bool: ...

    @property
    @abstractmethod
    def rate(self) -> float:
        """Default control rate in Hz."""

    @property
    @abstractmethod
    def joint_names(self) -> list[str]: ...

    # ── lifecycle ─────────────────────────────────────────────────────────
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def enable(self) -> None:
        """Put every motor in MIT mode and enable it."""

    @abstractmethod
    def disable(self) -> None:
        """Disable every motor (torque off). This is the e-stop path."""

    # ── I/O ───────────────────────────────────────────────────────────────
    @abstractmethod
    def read_positions(self) -> tuple[np.ndarray, float | None]:
        """Arm joint positions (rad, length n_arm) and gripper position (rad) or None."""

    @abstractmethod
    def send_arm_mit(self, q: np.ndarray, qd: np.ndarray, kp: np.ndarray, kd: np.ndarray, tau: np.ndarray) -> None: ...

    @abstractmethod
    def send_gripper_mit(self, pos: float, kp: float | None = None, kd: float | None = None) -> None: ...

    # ── control loop ──────────────────────────────────────────────────────
    @abstractmethod
    def start_loop(self, callback: Callable[[float], None], rate: float) -> None:
        """Call ``callback(dt)`` periodically at ``rate`` Hz from a dedicated thread."""

    @abstractmethod
    def stop_loop(self) -> None: ...

    # ── gains ─────────────────────────────────────────────────────────────
    @abstractmethod
    def hold_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Stiff per-joint MIT (kp, kd) suitable for holding and tracking."""

    @abstractmethod
    def gripper_gains(self) -> tuple[float, float]: ...

    def diagnostics(self) -> dict:
        return {}
