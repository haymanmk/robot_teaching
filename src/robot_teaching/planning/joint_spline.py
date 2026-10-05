"""Multi-waypoint joint-space planner.

Waypoints are joined by quintic Hermite segments with continuous velocity and
zero acceleration at every knot (C² path). Waypoints flagged ``stop`` get zero
velocity; the others are via-points whose velocity follows the classic
heuristic: the mean of the adjacent segment slopes when they agree in sign,
zero otherwise (so the path never overshoots a direction reversal).

Segment durations start from the rest-to-rest minimum and are stretched until
the sampled peak velocity and acceleration of every joint respect the limits.
Stretching one segment changes its neighbours' via velocities, so the check is
repeated until nothing changes. Durations only grow, so the loop converges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profiles import min_jerk_duration, quintic_coeffs, quintic_eval

_CHECK_SAMPLES = 64
_MAX_ITER = 100


@dataclass
class JointPathResult:
    t: np.ndarray            # (N,)
    q: np.ndarray            # (N, n)
    qd: np.ndarray           # (N, n)
    qdd: np.ndarray          # (N, n)
    knot_times: np.ndarray   # (m,) time of each waypoint, knot_times[0] == 0


def _via_velocities(Q: np.ndarray, T: np.ndarray, stop: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    m = Q.shape[0]
    V = np.zeros_like(Q)
    slopes = (Q[1:] - Q[:-1]) / T[:, None]           # (m-1, n)
    for k in range(1, m - 1):
        if stop[k]:
            continue
        s1, s2 = slopes[k - 1], slopes[k]
        same = (s1 * s2) > 0.0
        V[k] = np.where(same, 0.5 * (s1 + s2), 0.0)
        V[k] = np.clip(V[k], -vmax, vmax)
    return V


def plan_joint_path(
    waypoints: np.ndarray,
    stop: list[bool] | np.ndarray,
    vmax: np.ndarray,
    amax: np.ndarray,
    dt: float,
    segment_speed: list[float] | np.ndarray | None = None,
    t_min: float = 0.05,
) -> JointPathResult:
    """Plan through ``waypoints`` (m, n), starting and ending at rest.

    ``stop[k]`` forces zero velocity at waypoint ``k`` (the first and last are
    always stops). ``segment_speed[k]`` in (0, 1] scales the limits of segment
    ``k`` (from waypoint ``k`` to ``k+1``).
    """
    Q = np.asarray(waypoints, dtype=float)
    if Q.ndim != 2 or Q.shape[0] < 2:
        raise ValueError("need at least two waypoints")
    m, n = Q.shape
    vmax = np.asarray(vmax, dtype=float).reshape(-1)
    amax = np.asarray(amax, dtype=float).reshape(-1)
    if vmax.shape != (n,) or amax.shape != (n,):
        raise ValueError("vmax/amax must have one value per joint")
    if np.any(vmax <= 0) or np.any(amax <= 0):
        raise ValueError("velocity and acceleration limits must be positive")
    stop = np.asarray(stop, dtype=bool).reshape(-1)
    if stop.shape != (m,):
        raise ValueError("stop flags must have one entry per waypoint")
    stop = stop.copy()
    stop[0] = stop[-1] = True
    speed = np.ones(m - 1) if segment_speed is None else np.asarray(segment_speed, dtype=float).reshape(-1)
    if speed.shape != (m - 1,) or np.any(speed <= 0) or np.any(speed > 1.0):
        raise ValueError("segment_speed must have m-1 entries in (0, 1]")

    v_lim = vmax[None, :] * speed[:, None]     # (m-1, n)
    a_lim = amax[None, :] * speed[:, None]

    # Rest-to-rest minimum durations as the starting point.
    T = np.array([
        min_jerk_duration(Q[k + 1] - Q[k], v_lim[k], a_lim[k], t_min) for k in range(m - 1)
    ])

    tau = np.linspace(0.0, 1.0, _CHECK_SAMPLES)
    coeffs: list[np.ndarray] = []
    for _ in range(_MAX_ITER):
        V = _via_velocities(Q, T, stop, vmax)
        coeffs = [
            quintic_coeffs(Q[k], V[k], 0.0, Q[k + 1], V[k + 1], 0.0, T[k]) for k in range(m - 1)
        ]
        changed = False
        for k in range(m - 1):
            _, qd, qdd = quintic_eval(coeffs[k], tau * T[k])
            r_v = np.max(np.abs(qd) / v_lim[k])
            r_a = np.max(np.abs(qdd) / a_lim[k])
            f = max(r_v, np.sqrt(r_a))
            if f > 1.0 + 1e-3:
                T[k] *= f * 1.02
                changed = True
        if not changed:
            break
    else:  # pragma: no cover - defensive, the loop is monotone
        raise RuntimeError("joint path time scaling did not converge")

    knots = np.concatenate([[0.0], np.cumsum(T)])
    total = float(knots[-1])
    t = np.arange(0.0, total, dt)
    if len(t) == 0 or total - t[-1] > 1e-9:
        t = np.append(t, total)
    seg = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, m - 2)
    q = np.empty((len(t), n))
    qd = np.empty_like(q)
    qdd = np.empty_like(q)
    for k in range(m - 1):
        idx = np.flatnonzero(seg == k)
        if idx.size == 0:
            continue
        qk, qdk, qddk = quintic_eval(coeffs[k], t[idx] - knots[k])
        q[idx], qd[idx], qdd[idx] = qk, qdk, qddk
    # Land exactly on the last waypoint at rest.
    q[-1], qd[-1], qdd[-1] = Q[-1], 0.0, 0.0
    return JointPathResult(t=t, q=q, qd=qd, qdd=qdd, knot_times=knots)
