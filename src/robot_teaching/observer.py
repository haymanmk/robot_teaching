"""Disturbance observers: the joint torque the dynamics model does not explain.

With the model the controller has, ``M̂(q)`` (URDF plus ``armature``) and
``N̂(q, q̇) = Ĉ q̇ + ĝ``, the arm obeys

    M̂(q) q̈ + N̂(q, q̇) = τ + τ_d

where ``τ`` is the motor torque and ``τ_d`` lumps everything else: an external push, minus
friction, minus every error of the model. An observer estimates ``τ_d`` from the joint
positions, the joint velocities and the applied torque, without measuring acceleration. Two
forms are here, with the same interface (docs/disturbance_observer.md has the derivation and
the reasons for preferring the first on this arm):

* :class:`MomentumObserver`: the gain follows the pose, so every joint has the bandwidth it
  was given, in every pose (the generalized momentum observer of De Luca et al., 2006).
* :class:`ConstantGainObserver`: the nonlinear disturbance observer of Mohammadi et al.
  (2012) with a constant gain matrix ``Y``, kept for comparison. Its poles are the
  eigenvalues of ``Y M̂(q)⁻¹`` and move with the pose. :func:`inertia_bounds` and
  :func:`paper_gain` give that paper's design constants and its scalar gain.

Call :meth:`update` once per control tick. Its torque argument is the torque the motors
applied *since the previous call*: for a commanded torque, the value sent at the previous
tick; for the motors' own report, the latest reading. The first call only initialises the
observer and returns zeros.

An observer owns no Pinocchio data: it uses the :class:`RobotModel` it is given, which is
not thread safe, so give it the model of the thread that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import RobotModel


class _Observer:
    """State and model terms the two forms share."""

    def __init__(self, model: RobotModel, armature: np.ndarray | float | None = None, max_dt: float = 0.1,
                 gravity_scale: np.ndarray | float | None = None):
        self.model = model
        self.n = model.n
        self.armature = np.zeros(self.n)
        self.gravity_scale = np.ones(self.n)
        self.configure(armature=armature, gravity_scale=gravity_scale)
        self.max_dt = float(max_dt)
        self._z: np.ndarray | None = None
        self._estimate = np.zeros(self.n)

    def configure(self, armature: np.ndarray | float | None = None,
                  gravity_scale: np.ndarray | float | None = None) -> None:
        """Change the model the observer holds; ``None`` keeps a value. ``armature`` is the
        reflected rotor inertia per joint (kg·m²) added to the URDF's ``M(q)``;
        ``gravity_scale`` multiplies the URDF's ``g(q)`` per joint. An observer whose estimate
        is fed back must use the gravity model the controller feeds forward, or it reports the
        difference between the two as a disturbance and compensates gravity twice."""
        if armature is not None:
            a = np.broadcast_to(np.asarray(armature, dtype=float), (self.n,)).copy()
            if np.any(a < 0.0) or not np.all(np.isfinite(a)):
                raise ValueError("armature must be finite and not negative")
            self.armature = a
        if gravity_scale is not None:
            s = np.broadcast_to(np.asarray(gravity_scale, dtype=float), (self.n,)).copy()
            if not np.all(np.isfinite(s)):
                raise ValueError("gravity_scale must be finite")
            self.gravity_scale = s

    @property
    def estimate(self) -> np.ndarray:
        """The latest estimate of τ_d, N·m per joint (zeros before the first update)."""
        return self._estimate.copy()

    def reset(self) -> None:
        """Forget the state: the next update starts again from a zero estimate."""
        self._z = None
        self._estimate = np.zeros(self.n)

    def _mass(self, q: np.ndarray) -> np.ndarray:
        return self.model.mass_matrix(q) + np.diag(self.armature)

    def _gravity(self, q: np.ndarray) -> np.ndarray:
        return self.gravity_scale * self.model.gravity(q)

    def _inputs(self, q, qd, tau) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, qd, tau = (np.asarray(x, dtype=float).reshape(-1)[: self.n] for x in (q, qd, tau))
        if not (q.shape == qd.shape == tau.shape == (self.n,)):
            raise ValueError(f"q, qd and tau need {self.n} entries each")
        return q, qd, tau


class MomentumObserver(_Observer):
    """Disturbance observer on the generalized momentum ``p = M̂(q) q̇``.

    The estimate follows the disturbance through one first-order low-pass per joint,
    ``τ̂_d' = K (τ_d − τ̂_d)``, whatever the pose: ``bandwidth_hz`` (a scalar or one value per
    joint) is that filter's corner frequency, ``K = 2π · bandwidth_hz``. In the auxiliary-state
    form that needs no acceleration:

        z' = −K z + K (ĝ − Ĉᵀ q̇ − τ − K p),     τ̂_d = z + K p

    The update over a tick is the exact solution of that equation for the measured tick
    time, with the pose-dependent terms taken as the mean of the two ends of the interval,
    so neither a high bandwidth nor loop jitter makes it unstable.
    """

    def __init__(self, model: RobotModel, bandwidth_hz: np.ndarray | float = 5.0,
                 armature: np.ndarray | float | None = None, max_dt: float = 0.1,
                 gravity_scale: np.ndarray | float | None = None):
        super().__init__(model, armature, max_dt, gravity_scale)
        self.set_bandwidth(bandwidth_hz)
        self._w_prev: np.ndarray | None = None

    def set_bandwidth(self, bandwidth_hz: np.ndarray | float) -> None:
        """Change the corner frequency. The estimate carries on from where it is."""
        f = np.broadcast_to(np.asarray(bandwidth_hz, dtype=float), (self.n,)).copy()
        if np.any(f <= 0.0):
            raise ValueError("bandwidth_hz must be positive")
        k_old = getattr(self, "k", None)
        self.k = 2.0 * np.pi * f
        if k_old is not None and self._z is not None:
            # z and the stored term both contain K p: keep p and the estimate, swap the gain.
            p = (self._estimate - self._z) / k_old
            self._z = self._estimate - self.k * p
            self._w_prev = self._w_prev + (k_old - self.k) * p

    def reset(self) -> None:
        super().reset()
        self._w_prev = None

    def update(self, q, qd, tau, dt: float) -> np.ndarray:
        """Advance by one tick of ``dt`` seconds; ``tau`` is the torque applied during it."""
        q, qd, tau = self._inputs(q, qd, tau)
        k_p = self.k * (self._mass(q) @ qd)                      # K p
        w = self._gravity(q) - self.model.coriolis_matrix(q, qd).T @ qd - k_p
        if self._z is None or not (0.0 < dt <= self.max_dt):
            # First call, or a gap too long to integrate over: restart from the estimate held.
            self._z = self._estimate - k_p
        else:
            a = np.exp(-self.k * dt)
            self._z = a * self._z + (1.0 - a) * (0.5 * (self._w_prev + w) - tau)
        self._w_prev = w
        self._estimate = self._z + k_p
        return self._estimate.copy()


class ConstantGainObserver(_Observer):
    """The nonlinear disturbance observer of Mohammadi et al. (2012), constant gain ``Y``.

        z' = −L(q) z + L(q) (N̂(q, q̇) − τ − Y q̇),     τ̂_d = z + Y q̇,     L(q) = Y M̂(q)⁻¹

    ``gain`` is ``Y`` in N·m·s/rad: a scalar ``y`` (the paper's analytical design,
    :func:`paper_gain`), one value per joint, or a full matrix. The poles are the eigenvalues
    of ``L(q)`` (:meth:`poles`), which on this arm spread over orders of magnitude; the
    update is implicit (backward Euler) so that poles far above the control rate stay stable.
    """

    def __init__(self, model: RobotModel, gain: np.ndarray | float,
                 armature: np.ndarray | float | None = None, max_dt: float = 0.1,
                 gravity_scale: np.ndarray | float | None = None):
        super().__init__(model, armature, max_dt, gravity_scale)
        g = np.asarray(gain, dtype=float)
        if g.ndim == 0:
            self.Y = float(g) * np.eye(self.n)
        elif g.ndim == 1:
            self.Y = np.diag(np.broadcast_to(g, (self.n,)))
        else:
            self.Y = g.copy()
        if self.Y.shape != (self.n, self.n) or np.linalg.matrix_rank(self.Y) < self.n:
            raise ValueError(f"gain must be an invertible {self.n} × {self.n} matrix, a vector or a scalar")
        self._prev: tuple[np.ndarray, np.ndarray] | None = None   # (L, N̂ − Y q̇) at the last update

    def reset(self) -> None:
        super().reset()
        self._prev = None

    def poles(self, q: np.ndarray) -> np.ndarray:
        """Eigenvalues of ``L(q)`` in rad/s, slowest first: the observer's poles at this pose."""
        ev = np.linalg.eigvals(self.Y @ np.linalg.inv(self._mass(np.asarray(q, dtype=float))))
        return ev[np.argsort(ev.real)]

    def update(self, q, qd, tau, dt: float) -> np.ndarray:
        """Advance by one tick of ``dt`` seconds; ``tau`` is the torque applied during it."""
        q, qd, tau = self._inputs(q, qd, tau)
        yv = self.Y @ qd
        L = self.Y @ np.linalg.inv(self._mass(q))
        s = self.model.coriolis_matrix(q, qd) @ qd + self._gravity(q) - yv
        if self._z is None or not (0.0 < dt <= self.max_dt):
            self._z = self._estimate - yv
        else:
            L0, s0 = self._prev
            Lm = 0.5 * (L0 + L)
            self._z = np.linalg.solve(np.eye(self.n) + dt * Lm, self._z + dt * Lm @ (0.5 * (s0 + s) - tau))
        self._prev = (L, s)
        self._estimate = self._z + yv
        return self._estimate.copy()


# ── design constants of the constant-gain form ─────────────────────────────────────────────

@dataclass
class InertiaBounds:
    """What Mohammadi et al. need to know about ``M̂(q)`` over the workspace."""

    sigma1: float       # smallest eigenvalue of M̂, kg·m²
    sigma2: float       # largest eigenvalue of M̂, kg·m²
    zeta: float         # bound on ‖dM̂/dt‖ at the velocity limits, kg·m²/s
    condition: float    # largest condition number of M̂ met


def inertia_bounds(model: RobotModel, velocity_limits: np.ndarray | float,
                   armature: np.ndarray | float | None = None, samples: int = 5000,
                   seed: int = 0) -> InertiaBounds:
    """Sample the joint range for the extreme eigenvalues of ``M̂`` and the largest ``‖Ṁ‖``.

    ``Ṁ = C + Cᵀ`` is linear in the velocity, so its norm is taken at the corners of the
    velocity box. The bounds are estimates from below: more samples can only widen them.
    """
    n = model.n
    vmax = np.broadcast_to(np.asarray(velocity_limits, dtype=float), (n,))
    arm = np.zeros(n) if armature is None else np.broadcast_to(np.asarray(armature, dtype=float), (n,))
    rng = np.random.default_rng(seed)
    s1, s2, zeta, cond = np.inf, 0.0, 0.0, 0.0
    for _ in range(int(samples)):
        q = rng.uniform(model.lower, model.upper)
        w = np.linalg.eigvalsh(model.mass_matrix(q) + np.diag(arm))
        s1, s2, cond = min(s1, w[0]), max(s2, w[-1]), max(cond, w[-1] / w[0])
        C = model.coriolis_matrix(q, rng.choice([-1.0, 1.0], n) * vmax)
        zeta = max(zeta, float(np.linalg.norm(C + C.T, 2)))
    return InertiaBounds(sigma1=float(s1), sigma2=float(s2), zeta=zeta, condition=float(cond))


def paper_gain(bounds: InertiaBounds, beta: float) -> float:
    """The paper's scalar gain ``y = ζ/2 + β σ₂`` (its eq. 58) for a slowest convergence rate
    ``beta`` in rad/s. The fastest pole is then ``y / σ₁``."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    return 0.5 * bounds.zeta + float(beta) * bounds.sigma2
