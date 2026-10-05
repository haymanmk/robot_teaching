"""Straight-line end-effector moves (MoveL).

The position moves on a straight line and the orientation on the SO(3)
geodesic, both under one minimum-jerk time profile (the SE(3) exponential used
by the upstream sampler is a screw motion whose translation bows sideways when
the orientation changes, so it is not used here). The Cartesian samples are
tracked in joint space with the upstream closed-loop IK. Duration comes from
the Cartesian limits; the joint-space result is re-checked against joint
limits and joint velocity/acceleration limits and re-timed if they are
exceeded.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from reBotArm_control_py.trajectory import (
    CartesianTrajectory,
    IKParams as ClikParams,
    track_trajectory,
)

from ..model import RobotModel
from .profiles import MIN_JERK_PEAK_ACC, MIN_JERK_PEAK_VEL, quintic_coeffs, quintic_eval
from .trajectory import PlanningError, Trajectory

_CLIK = ClikParams(max_iter=100, tolerance=1e-4, damping=1e-6, step_size=0.8)
_MAX_RETIME = 4
_END_TOL = 1e-3        # rad: tracked end configuration must match the taught one
_END_FIX_MAX = 0.05    # rad: larger mismatch means a different IK branch → refuse


def _cartesian_duration(T0: pin.SE3, T1: pin.SE3, v_lin, a_lin, v_ang, a_ang, t_min: float) -> float:
    d = float(np.linalg.norm(T1.translation - T0.translation))
    theta = float(np.linalg.norm(pin.log3(T0.rotation.T @ T1.rotation)))
    cands = [
        t_min,
        MIN_JERK_PEAK_VEL * d / v_lin,
        np.sqrt(MIN_JERK_PEAK_ACC * d / a_lin),
        MIN_JERK_PEAK_VEL * theta / v_ang,
        np.sqrt(MIN_JERK_PEAK_ACC * theta / a_ang),
    ]
    return float(max(cands))


def sample_straight_line(T0: pin.SE3, T1: pin.SE3, duration: float, dt: float) -> CartesianTrajectory:
    """Minimum-jerk straight line in position, geodesic in orientation."""
    n = max(2, int(np.ceil(duration / dt)) + 1)
    p0, p1 = T0.translation.copy(), T1.translation.copy()
    R0 = T0.rotation.copy()
    w = pin.log3(R0.T @ T1.rotation)
    traj = CartesianTrajectory()
    for t in np.linspace(0.0, duration, n):
        tau = t / duration
        s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        traj.add_point(float(t), pin.SE3(R0 @ pin.exp3(w * s), p0 + s * (p1 - p0)))
    return traj


def plan_linear_segment(
    model: RobotModel,
    q_start: np.ndarray,
    q_end: np.ndarray,
    vmax: np.ndarray,
    amax: np.ndarray,
    v_lin: float,
    a_lin: float,
    v_ang: float,
    a_ang: float,
    dt: float,
    speed: float = 1.0,
    margin: float = 0.0,
    t_min: float = 0.05,
) -> Trajectory:
    """Plan a rest-to-rest straight-line move from ``q_start`` to the pose of ``q_end``.

    ``margin`` is kept for call compatibility; the path is checked against the hard joint
    limits (see below), the margin being a property of taught points.
    """
    q_start = np.asarray(q_start, dtype=float)[: model.n]
    q_end = np.asarray(q_end, dtype=float)[: model.n]
    T0, T1 = model.fk(q_start), model.fk(q_end)
    v_lim, a_lim = vmax * speed, amax * speed
    duration = _cartesian_duration(T0, T1, v_lin * speed, a_lin * speed, v_ang * speed, a_ang * speed, t_min)

    for _ in range(_MAX_RETIME):
        cart = sample_straight_line(T0, T1, duration, dt)
        pts = track_trajectory(model.model, model.ee_frame_id, cart, model.pad(q_start), _CLIK)
        failed = [i for i, p in enumerate(pts) if not p.ik_success]
        if failed:
            frac = failed[0] / max(1, len(pts) - 1)
            raise PlanningError(f"linear move is not reachable (IK failed at {frac:.0%} of the path)")
        t = np.array([p.time for p in pts])
        q = np.array([p.q[: model.n] for p in pts])
        # The path only has to respect the hard limits (the tracker clamps to them and reports a
        # tracking failure if the line needs more). The safety margin applies to taught points,
        # which the program validation enforces; the move may start on a limit, e.g. at the
        # fully extended home pose where joints 2 and 3 sit at their lower limit.
        bad = [i for i, row in enumerate(q) if not model.within_limits(row, 0.0)]
        if bad:
            j = int(np.argmax(np.maximum(model.lower - q[bad[0]], q[bad[0]] - model.upper)))
            raise PlanningError(
                f"linear move leaves the joint limits (joint {j + 1} at {bad[0] / max(1, len(q) - 1):.0%} "
                "of the path); use a joint move instead"
            )
        qd = np.gradient(q, t, axis=0)
        qd[0] = 0.0
        qd[-1] = 0.0
        qdd = np.gradient(qd, t, axis=0)
        r_v = float(np.max(np.abs(qd) / v_lim))
        r_a = float(np.max(np.abs(qdd) / a_lim))
        f = max(r_v, np.sqrt(r_a))
        if f <= 1.0 + 1e-2:
            break
        duration *= f * 1.05
    else:
        raise PlanningError("linear move cannot satisfy the joint velocity limits (near a singularity?)")

    mismatch = float(np.max(np.abs(q[-1] - q_end)))
    if mismatch > _END_FIX_MAX:
        raise PlanningError(
            f"linear move ends on a different IK branch (joint mismatch {mismatch:.3f} rad); use a joint move"
        )
    if mismatch <= _END_TOL:
        q[-1] = q_end      # within IK tolerance: land exactly on the taught configuration
    traj = Trajectory(t, q, qd)
    if mismatch > _END_TOL:
        # Tiny joint-space correction so playback lands exactly on the taught configuration.
        T_fix = max(t_min, float(np.max(MIN_JERK_PEAK_VEL * np.abs(q_end - q[-1]) / v_lim)))
        c = quintic_coeffs(q[-1], 0.0, 0.0, q_end, 0.0, 0.0, T_fix)
        tf = np.arange(0.0, T_fix, dt)
        if len(tf) == 0 or T_fix - tf[-1] > 1e-9:
            tf = np.append(tf, T_fix)
        qf, qdf, _ = quintic_eval(c, tf)
        traj = Trajectory.concatenate([traj, Trajectory(tf, qf, qdf)])
    return traj
