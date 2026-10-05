"""Kinematic/dynamic model of the arm, wrapping the upstream Pinocchio helpers.

One :class:`RobotModel` owns one Pinocchio ``Data`` object, which is not thread
safe: the control thread and the API thread each create their own instance.
Only the first ``n_arm`` generalized coordinates are controlled; the gripper's
prismatic joints in the URDF are padded with zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics import (
    get_end_effector_frame_id,
    get_joint_names,
    load_robot_model,
)
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik


@dataclass
class Pose:
    """End-effector pose: position in metres, roll/pitch/yaw in radians (XYZ convention)."""

    xyz: np.ndarray
    rpy: np.ndarray

    def to_dict(self) -> dict:
        return {"xyz": [float(v) for v in self.xyz], "rpy": [float(v) for v in self.rpy]}

    @classmethod
    def from_se3(cls, T: pin.SE3) -> "Pose":
        return cls(xyz=T.translation.copy(), rpy=pin.rpy.matrixToRpy(T.rotation))

    def to_se3(self) -> pin.SE3:
        return pin.SE3(pin.rpy.rpyToMatrix(*[float(v) for v in self.rpy]), np.asarray(self.xyz, dtype=float))

    @classmethod
    def relative(cls, T_ref: pin.SE3, T: pin.SE3) -> "Pose":
        """``T`` seen from the reference pose: position offset along the base axes, and the
        orientation as the rotation about the base axes that takes ``T_ref`` to ``T``
        (``R · R_refᵀ``). Both read zero when ``T == T_ref``; yaw stays a rotation about
        the vertical even when the tool points down at the reference."""
        return cls(xyz=T.translation - T_ref.translation,
                   rpy=pin.rpy.matrixToRpy(T.rotation @ T_ref.rotation.T))


class RobotModel:
    """Arm model with FK, IK, Jacobian, gravity vector and joint limits."""

    def __init__(self, n_arm: int = 6, urdf_path: str | None = None, ik_params: IKParams | None = None):
        self.model = load_robot_model(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = get_end_effector_frame_id(self.model)
        self.n = int(n_arm)
        if self.n > self.model.nq:
            raise ValueError(f"n_arm={self.n} exceeds model.nq={self.model.nq}")
        self.joint_names = get_joint_names(self.model)[: self.n]
        self.lower = np.array(self.model.lowerPositionLimit[: self.n], dtype=float)
        self.upper = np.array(self.model.upperPositionLimit[: self.n], dtype=float)
        self.ik_params = ik_params or IKParams(max_iter=300, tolerance=1e-4, step_size=0.5, damping=1e-6)

    # ── helpers ───────────────────────────────────────────────────────────

    def pad(self, q: np.ndarray) -> np.ndarray:
        full = np.zeros(self.model.nq)
        k = min(self.n, len(q))
        full[:k] = q[:k]
        return full

    def clamp(self, q: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower + margin, self.upper - margin)

    def within_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.lower + margin - 1e-9) and np.all(q <= self.upper - margin + 1e-9))

    # ── kinematics ────────────────────────────────────────────────────────

    def fk(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, self.pad(q))
        pin.updateFramePlacements(self.model, self.data)
        return pin.SE3(self.data.oMf[self.ee_frame_id])

    def pose(self, q: np.ndarray) -> Pose:
        return Pose.from_se3(self.fk(q))

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6×n end-effector Jacobian in the LOCAL_WORLD_ALIGNED frame (world axes at the EE point)."""
        qf = self.pad(q)
        pin.computeJointJacobians(self.model, self.data, qf)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return np.array(J[:, : self.n])

    def ee_velocity(self, q: np.ndarray, qd: np.ndarray) -> tuple[float, float]:
        """Linear (m/s) and angular (rad/s) end-effector speed for joint velocity ``qd``."""
        v = self.jacobian(q) @ np.asarray(qd, dtype=float)[: self.n]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def ik(self, target: pin.SE3, q_seed: np.ndarray, position_only: bool = False):
        """Damped-least-squares IK from the upstream solver; ``result.q`` has ``n`` entries."""
        return solve_ik(
            self.model, self.data, self.ee_frame_id, target,
            np.asarray(q_seed, dtype=float)[: self.n], self.ik_params,
            controlled_joints=self.n, position_only=position_only,
        )

    # ── dynamics ──────────────────────────────────────────────────────────

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Generalized gravity torque g(q) for the arm joints, N·m."""
        pin.computeGeneralizedGravity(self.model, self.data, self.pad(q))
        return np.array(self.data.g[: self.n])

    def mass_matrix_diag(self, q: np.ndarray) -> np.ndarray:
        """Diagonal of the joint-space inertia matrix (used by the simulator)."""
        pin.crba(self.model, self.data, self.pad(q))
        M = np.array(self.data.M)
        M = np.triu(M) + np.triu(M, 1).T
        return np.diag(M)[: self.n].copy()
