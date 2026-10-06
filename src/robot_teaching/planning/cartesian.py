"""Straight-line end-effector moves (MoveL), optionally blended through via-points.

A block of consecutive linear points is one Cartesian path: straight lines
between the taught poses, with every pass-through corner replaced by a
quadratic Bézier blend that starts ``blend_radius`` before the corner and ends
the same distance after it (less when a neighbouring segment is short), so
position and velocity are continuous and the corner is cut, as in an
industrial blend zone. Orientation follows the SO(3) geodesic between the
taught orientations along the path parameter.

Timing comes from a velocity-limit curve along the path: the Cartesian speed
limit, the angular speed limit on each segment, and the centripetal
acceleration the blends allow, with zero speed at both ends. A forward and a
backward pass at the Cartesian acceleration limit give the fastest profile
under that curve, so the arm slows down only where a corner needs it. The
samples are tracked in joint space with the upstream closed-loop IK, checked
against the hard joint limits (the safety margin is a property of taught
points) and against the joint velocity and acceleration limits, and
time-scaled if those are exceeded.

The SE(3) exponential used by the upstream sampler is a screw motion whose
translation bows sideways when the orientation changes, so it is not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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

_CLIK = ClikParams(max_iter=100, tolerance=1e-6, damping=1e-6, step_size=0.8)   # tight: 1e-4 leaves steps that look like accelerations
_END_TOL = 1e-3        # rad: tracked end configuration must match the taught one
_END_FIX_MAX = 0.05    # rad: larger mismatch means a different IK branch → refuse
ROTATION_LEVER = 0.1   # m per rad: path length credited to a pure rotation so it gets a duration
_TABLE_STEP = 0.002    # m, resolution of the geometric path table
_BLEND_SAMPLES = 48


@dataclass
class LinearBlockResult:
    trajectory: Trajectory
    arrival_times: list[float]     # one per waypoint after the start (pass-through: mid-blend)
    path_length: float
    time_scale: float              # > 1 when joint limits forced the move to slow down


class _GeomPath:
    """Polyline through the waypoint positions with Bézier corner blends, tabulated by path parameter.

    The path parameter is the arc length, plus ``ROTATION_LEVER × angle`` for
    segments without translation, so a reorientation in place still has length.
    """

    def __init__(self, P: list[np.ndarray], theta: list[float], radius: float):
        m = len(P) - 1
        seg = [P[i + 1] - P[i] for i in range(m)]
        L = [float(np.linalg.norm(v)) for v in seg]
        u = [seg[i] / L[i] if L[i] > 1e-9 else np.zeros(3) for i in range(m)]
        r = [0.0] * (m + 1)
        for i in range(1, m):
            if L[i - 1] > 1e-9 and L[i] > 1e-9:
                r[i] = min(radius, 0.5 * L[i - 1], 0.5 * L[i])
        S, X, K, corner = [0.0], [P[0].copy()], [0.0], [0.0]
        kappa_max, s = 0.0, 0.0
        for i in range(m):
            a = P[i] + r[i] * u[i]
            b = P[i + 1] - r[i + 1] * u[i]
            straight = max(L[i] - r[i] - r[i + 1], 0.0)
            eff = straight + (ROTATION_LEVER * theta[i] if L[i] <= 1e-9 else 0.0)
            if eff > 1e-12:
                n = max(2, int(np.ceil(eff / _TABLE_STEP)))
                for k in range(1, n + 1):
                    f = k / n
                    S.append(s + eff * f)
                    X.append(a + (b - a) * f)
                    K.append(0.0)
                s += eff
            if i + 1 < m and r[i + 1] > 0.0:
                A, C, B = b, P[i + 1], P[i + 1] + r[i + 1] * u[i + 1]
                prev, blen = A.copy(), 0.0
                d2 = 2.0 * (A - 2.0 * C + B)
                for k in range(1, _BLEND_SAMPLES + 1):
                    t = k / _BLEND_SAMPLES
                    x = (1 - t) ** 2 * A + 2 * t * (1 - t) * C + t**2 * B
                    blen += float(np.linalg.norm(x - prev))
                    prev = x
                    S.append(s + blen)
                    X.append(x)
                    d1 = 2 * (1 - t) * (C - A) + 2 * t * (B - C)
                    sp = float(np.linalg.norm(d1))
                    kappa = float(np.linalg.norm(np.cross(d1, d2)) / sp**3) if sp > 1e-9 else 0.0
                    K.append(kappa)
                    kappa_max = max(kappa_max, kappa)
                corner.append(s + 0.5 * blen)
                s += blen
            else:
                corner.append(s)
        self.S, self.X, self.K = np.array(S), np.array(X), np.array(K)
        self.corner = corner
        self.length = s
        self.kappa_max = kappa_max
        self.blend_radius = r

    def position(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.S, self.X[:, k]) for k in range(3)])

    def curvature(self, s) -> np.ndarray:
        return np.interp(s, self.S, self.K)

    def time_profile(self, v_lin: float, a_lin: float, v_ang: float, theta: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Fastest rest-to-rest timing under the velocity-limit curve with bounded acceleration.

        Returns ``(t_grid, s_grid)`` with ``s_grid`` from 0 to the path length.
        """
        n_grid = max(3, int(np.ceil(self.length / _TABLE_STEP)) + 1)
        grid = np.linspace(0.0, self.length, n_grid)
        v_lim = np.full(n_grid, v_lin)
        kappa = self.curvature(grid)
        curved = kappa > 1e-9
        v_lim[curved] = np.minimum(v_lim[curved], np.sqrt(a_lin / kappa[curved]))
        for i in range(len(self.corner) - 1):
            den = self.corner[i + 1] - self.corner[i]
            if theta[i] > 1e-9 and den > 1e-12:
                mask = (grid >= self.corner[i] - 1e-12) & (grid <= self.corner[i + 1] + 1e-12)
                v_lim[mask] = np.minimum(v_lim[mask], v_ang * den / theta[i])
        v_lim[0] = v_lim[-1] = 0.0
        ds = np.diff(grid)
        v = np.zeros(n_grid)
        for k in range(n_grid - 1):                       # forward: accelerate as allowed
            v[k + 1] = min(v_lim[k + 1], np.sqrt(v[k] ** 2 + 2.0 * a_lin * ds[k]))
        for k in range(n_grid - 2, -1, -1):               # backward: brake in time for what follows
            v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2.0 * a_lin * ds[k]))
        vsum = v[:-1] + v[1:]
        dt = np.where(vsum > 1e-12, 2.0 * ds / np.maximum(vsum, 1e-12), 0.0)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        self._profile = (t, grid, v)
        return t, grid

    def path_param_at(self, times: np.ndarray) -> np.ndarray:
        """s(t) for sample times, with constant acceleration inside each grid cell.

        Linear interpolation of ``s`` between grid points would give a constant
        speed per cell and therefore a speed jump out of rest at both ends; the
        quadratic law keeps the sampled velocity continuous.
        """
        t_grid, s_grid, v = self._profile
        k = np.clip(np.searchsorted(t_grid, times, side="right") - 1, 0, len(t_grid) - 2)
        ds = s_grid[k + 1] - s_grid[k]
        acc = np.where(ds > 1e-12, (v[k + 1] ** 2 - v[k] ** 2) / np.maximum(2.0 * ds, 1e-12), 0.0)
        tau = np.clip(times - t_grid[k], 0.0, None)
        out = s_grid[k] + v[k] * tau + 0.5 * acc * tau**2
        return np.clip(out, 0.0, self.length)


def plan_linear_block(
    model: RobotModel,
    q_start: np.ndarray,
    waypoints: Sequence[np.ndarray],
    vmax: np.ndarray,
    amax: np.ndarray,
    v_lin: float,
    a_lin: float,
    v_ang: float,
    a_ang: float,
    dt: float,
    speed: float = 1.0,
    blend_radius: float = 0.0,
    t_min: float = 0.05,
) -> LinearBlockResult:
    """Rest-to-rest Cartesian path from ``q_start`` through the poses of ``waypoints``.

    Every waypoint but the last is passed through with a corner blend of at
    most ``blend_radius``; ``blend_radius=0`` gives sharp corners, which are
    only sensible for a single waypoint.
    """
    q_start = np.asarray(q_start, dtype=float)[: model.n]
    Q = [np.asarray(q, dtype=float)[: model.n] for q in waypoints]
    if not Q:
        raise PlanningError("a linear block needs at least one waypoint")
    T = [model.fk(q_start)] + [model.fk(q) for q in Q]
    m = len(Q)
    P = [t.translation.copy() for t in T]
    R = [t.rotation.copy() for t in T]
    w = [pin.log3(R[i].T @ R[i + 1]) for i in range(m)]
    theta = [float(np.linalg.norm(v)) for v in w]
    path = _GeomPath(P, theta, blend_radius)
    v_lim, a_lim = vmax * speed, amax * speed

    if path.length < 1e-9:
        # Nothing to move in Cartesian space: land on the taught configuration with a short joint move.
        q_end = Q[-1]
        T_fix = max(t_min, float(np.max(MIN_JERK_PEAK_VEL * np.abs(q_end - q_start) / v_lim)))
        t = _times(T_fix, dt)
        qf, qdf, _ = quintic_eval(quintic_coeffs(q_start, 0.0, 0.0, q_end, 0.0, 0.0, T_fix), t)
        traj = Trajectory(t, qf, qdf)
        return LinearBlockResult(traj, [traj.duration] * m, 0.0, 1.0)

    # Timing: velocity-limit curve (linear, angular, centripetal) with bounded acceleration.
    t_grid, s_grid = path.time_profile(v_lin * speed, a_lin * speed, v_ang * speed, theta)
    duration = max(t_min, float(t_grid[-1]))
    if duration > t_grid[-1] + 1e-12:
        t_grid = t_grid * (duration / max(t_grid[-1], 1e-12))

    def orientation(s: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(path.corner, s, side="right") - 1, 0, m - 1))
        den = path.corner[i + 1] - path.corner[i]
        f = 1.0 if den <= 1e-12 else float(np.clip((s - path.corner[i]) / den, 0.0, 1.0))
        return R[i] @ pin.exp3(w[i] * f)

    t = _times(duration, dt)
    s_of_t = path.path_param_at(t * (t_grid[-1] / duration))
    cart = CartesianTrajectory()
    for tk, sk in zip(t, s_of_t):
        cart.add_point(float(tk), pin.SE3(orientation(sk), path.position(sk)))

    pts = track_trajectory(model.model, model.ee_frame_id, cart, model.pad(q_start), _CLIK)
    failed = [i for i, p in enumerate(pts) if not p.ik_success]
    if failed:
        frac = failed[0] / max(1, len(pts) - 1)
        raise PlanningError(f"linear move is not reachable (IK failed at {frac:.0%} of the path)")
    q = np.array([p.q[: model.n] for p in pts])
    # The path only has to respect the hard limits (the tracker clamps to them and reports a
    # tracking failure if the line needs more). The safety margin applies to taught points.
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
    scale = max(1.0, float(np.max(np.abs(qd) / v_lim)), float(np.sqrt(np.max(np.abs(qdd) / a_lim))))
    if scale > 1.0 + 1e-2:
        # Same path, slower: stretching time keeps the Cartesian samples and the IK solution.
        scale *= 1.02
        t = t * scale
        qd = qd / scale
    else:
        scale = 1.0
    arrival = [float(np.interp(path.corner[i + 1], s_of_t, t)) for i in range(m)]

    q_end = Q[-1]
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
        tf = _times(T_fix, dt)
        qf, qdf, _ = quintic_eval(quintic_coeffs(q[-1], 0.0, 0.0, q_end, 0.0, 0.0, T_fix), tf)
        traj = Trajectory.concatenate([traj, Trajectory(tf, qf, qdf)])
    arrival[-1] = traj.duration
    return LinearBlockResult(traj, arrival, path.length, scale)


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
    """Single rest-to-rest straight-line move (a one-waypoint block). ``margin`` is kept for call compatibility."""
    del margin
    return plan_linear_block(model, q_start, [q_end], vmax, amax, v_lin, a_lin, v_ang, a_ang, dt,
                             speed=speed, blend_radius=0.0, t_min=t_min).trajectory


def _times(duration: float, dt: float) -> np.ndarray:
    """Uniform sample times from 0 to ``duration`` with spacing as close to ``dt`` as possible.

    (Appending the end time to an ``arange`` can leave a last step of a few ms, which turns the
    forced zero end velocity into a large apparent acceleration.)
    """
    n = max(1, int(np.ceil(duration / dt)))
    return np.linspace(0.0, duration, n + 1)
