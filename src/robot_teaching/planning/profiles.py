"""Quintic Hermite segments.

A quintic polynomial is the lowest degree that lets us prescribe position,
velocity and acceleration at both ends. With zero boundary velocity and
acceleration it is the minimum-jerk profile ``10s^3 - 15s^4 + 6s^5``.

Coefficients are computed per joint (vectorized along the last axis).
"""

from __future__ import annotations

import numpy as np

# Peak |velocity| and |acceleration| of a rest-to-rest quintic, in units of (Δq/T) and (Δq/T²).
MIN_JERK_PEAK_VEL = 1.875
MIN_JERK_PEAK_ACC = 10.0 / np.sqrt(3.0)   # ≈ 5.7735


def quintic_coeffs(q0, v0, a0, q1, v1, a1, T: float) -> np.ndarray:
    """Return coefficients ``c`` (6, n) of ``q(t) = Σ c_k t^k`` on ``[0, T]``."""
    if T <= 0.0:
        raise ValueError("segment duration must be > 0")
    q0, v0, a0, q1, v1, a1 = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, dtype=float)) for x in (q0, v0, a0, q1, v1, a1))
    )
    d = q1 - q0
    T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
    c0 = q0
    c1 = v0
    c2 = a0 / 2.0
    c3 = (20.0 * d - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * T2) / (2.0 * T3)
    c4 = (-30.0 * d + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * T2) / (2.0 * T4)
    c5 = (12.0 * d - 6.0 * (v1 + v0) * T + (a1 - a0) * T2) / (2.0 * T5)
    return np.stack([c0, c1, c2, c3, c4, c5])


def quintic_eval(c: np.ndarray, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate position, velocity and acceleration at time(s) ``t``.

    ``t`` scalar → arrays of shape (n,); ``t`` of shape (m,) → arrays of shape (m, n).
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    tt = t.reshape(-1, 1)
    c0, c1, c2, c3, c4, c5 = c
    q = c0 + c1 * tt + c2 * tt**2 + c3 * tt**3 + c4 * tt**4 + c5 * tt**5
    qd = c1 + 2 * c2 * tt + 3 * c3 * tt**2 + 4 * c4 * tt**3 + 5 * c5 * tt**4
    qdd = 2 * c2 + 6 * c3 * tt + 12 * c4 * tt**2 + 20 * c5 * tt**3
    if scalar:
        return q[0], qd[0], qdd[0]
    return q, qd, qdd


def min_jerk_duration(delta, vmax, amax, t_min: float = 0.05) -> float:
    """Shortest rest-to-rest minimum-jerk duration respecting per-joint limits."""
    delta = np.abs(np.atleast_1d(np.asarray(delta, dtype=float)))
    vmax = np.atleast_1d(np.asarray(vmax, dtype=float))
    amax = np.atleast_1d(np.asarray(amax, dtype=float))
    t_v = MIN_JERK_PEAK_VEL * delta / vmax
    t_a = np.sqrt(MIN_JERK_PEAK_ACC * delta / amax)
    return float(max(t_min, np.max(t_v), np.max(t_a)))
