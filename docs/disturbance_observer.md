# Disturbance observer for sensorless force estimation: design note

**Status (2026-11-06).** A design note: nothing here is implemented. It records what the
nonlinear disturbance observer (NDOB) of Mohammadi et al. gives on this arm, the concerns
that came out of checking it against the arm's model and control loop, and the form and
order of work suggested instead. The numbers were computed from the URDF with the snippets
below (`uv run python` from the repository root); the two observer forms were checked
against a known disturbance on a simulated plant, not on the arm.

**Why.** The goal is admittance control without a load cell at the end-effector, which the
payload budget rules out. A disturbance observer estimates the torque the model does not
explain, from the joint positions, velocities and the applied torque alone. It also
tolerates the mismatch that will always remain between the model and one particular arm, and
the drift of that arm over time.

## 1. What the observer estimates

Write the model the controller has as $\hat M(q)$ and $\hat N(q,\dot q) = \hat C\,\dot q + \hat g$.
The arm then obeys

$$
\hat M(q)\,\ddot q + \hat N(q,\dot q) = \tau + \tau_d, \qquad
\tau_d = \tau_{ext} - \Delta M\,\ddot q - \Delta N - F(\dot q)
$$

The observer estimates the lumped term $\tau_d$: the external torque, minus the friction
$F$, minus every error of the model ($\Delta M$, $\Delta N$). It cannot tell these apart.
That has two consequences which pull in opposite directions:

- **Disturbance rejection tolerates a crude model.** Feeding $\hat\tau_d$ back cancels the
  whole lump, whatever it is made of.
- **Force sensing does not.** For admittance the wanted signal is $\tau_{ext}$ alone:
  $\hat\tau_{ext} = \hat\tau_d + \hat F(\dot q)$, and every N·m of friction or model error
  that is not subtracted is reported as a push from the user. The estimate is as accurate
  as the model, so identification still matters; what the observer buys is that a slow
  drift can be zeroed whenever the arm is known to be in free space.

The two uses also conflict in the loop. Feeding $\hat\tau_d$ back makes the arm stiffer
against a hand, the opposite of admittance. For admittance the estimate is an input to a
virtual mass–damper that moves the reference; it is not added to the motor torque.

## 2. The paper's observer

Mohammadi, Tavakoli, Marquez and Hashemzadeh (2012) avoid measuring acceleration with an
auxiliary state $z$ (their eq. 25, 27, 28). With a constant matrix $Y = X^{-1}$:

$$
\dot z = -L(q)\,z + L(q)\,\bigl(\hat N(q,\dot q) - \tau - Y\dot q\bigr), \qquad
\hat\tau_d = z + Y\dot q, \qquad L(q) = Y\,\hat M(q)^{-1}
$$

The estimation error $\Delta\tau_d = \tau_d - \hat\tau_d$ then follows
$\Delta\dot\tau_d = \dot\tau_d - L(q)\,\Delta\tau_d$. It converges exponentially for a slowly
varying disturbance, and stays bounded for a fast one, if (their eq. 29)

$$
X + X^T - X^T\,\dot{\hat M}(q)\,X \ge \Gamma > 0
$$

which they turn into a linear matrix inequality in $Y$. For a scalar gain $Y = y\,I$ the
answer is explicit (eq. 57):

$$
y \ge \tfrac12\,\zeta + \beta\,\sigma_2
$$

with $\sigma_2$ the largest eigenvalue of $\hat M$ over the workspace, $\zeta$ a bound on
$\lVert \dot{\hat M} \rVert$, and $\beta$ the slowest convergence rate accepted. $\hat M$ may
be any symmetric positive definite estimate, so the method does not need an exact model.

## 3. The design constants on this arm

```python
import numpy as np, pinocchio as pin
from robot_teaching.model import RobotModel

m = RobotModel(6); n = m.n
vmax = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 2.0])     # limits.joint_velocity
rng = np.random.default_rng(0)
s1, s2, zeta, cond = np.inf, 0.0, 0.0, []
for _ in range(20000):
    q = rng.uniform(m.lower, m.upper)
    w = np.linalg.eigvalsh(m.mass_matrix(q))
    s1, s2 = min(s1, w[0]), max(s2, w[-1]); cond.append(w[-1] / w[0])
    v = rng.choice([-1, 1], n) * vmax
    C = np.array(pin.computeCoriolisMatrix(m.model, m.data, m.pad(q), m.pad(v)))[:n, :n]
    zeta = max(zeta, np.linalg.norm(C + C.T, 2))     # dM/dt = C + C^T
print(s1, s2, np.median(cond), max(cond), zeta)
```

| quantity | value (URDF only, no armature) |
|---|---|
| $\sigma_1$, smallest eigenvalue of $M$ | 3.6 × 10⁻⁴ kg·m² |
| $\sigma_2$, largest eigenvalue of $M$ | 1.04 kg·m² |
| condition number of $M$ | median 1 500, largest 2 800 |
| $\zeta$ at the playback velocity limits | 1.53 kg·m²/s |
| $M_{11}$ over the workspace | 0.005 to 0.84 kg·m² |

The observer's poles are the eigenvalues of $L(q) = Y \hat M(q)^{-1}$, so a scalar $y$ gives
poles from $y/\sigma_2$ to $y/\sigma_1$:

| $\beta$ (rad/s) | $y$ | slowest pole (rad/s) | fastest pole (rad/s) |
|---|---|---|---|
| 6 (the paper's SCARA value) | 7.0 | 6.7 | 19 500 |
| 30 | 31.8 | 30.7 | 89 000 |

**Concern 1: a constant gain spreads the poles by three orders of magnitude.** A 500 Hz
loop integrates a pole explicitly only below about 1 000 rad/s ($\lambda\,\Delta t < 2$); the
wrist poles are far above that while the shoulder answers in 150 ms. An implicit update
keeps it stable (the simulated check below does that), but the wrist estimate is then an
unfiltered copy of the velocity noise. A constant matrix does not repair it: with
$Y = \omega \bar M$ ($\bar M$ the mean inertia, $\omega$ = 2π · 5 Hz) the poles still run from
7 to 1 800 rad/s, because the inertia itself changes that much with the pose.

**Concern 2: the paper's condition is not the binding constraint.** It is a floor: the
observer must be faster than the inertia matrix changes, and $\zeta/2$ = 0.76 is met by any
useful gain. The ceiling comes from what the paper does not model: the sample rate, the
noise and lag of the velocity signal, and how well the applied torque is known (section 5).

## 4. Suggested form: let the gain follow the pose

Choose $Y = K\,\hat M(q)$ with $K = \mathrm{diag}(k_i)$. The paper's theorem does not cover a
pose-dependent $Y$, but none is needed: $L = K$ exactly, the error dynamics are

$$
\Delta\dot\tau_d = \dot\tau_d - K\,\Delta\tau_d
$$

for every pose, and $k_i$ is directly the bandwidth of joint $i$ in rad/s. The price is one
term. The auxiliary vector becomes $K \hat M(q)\,\dot q$, $K$ times the generalized momentum,
and its derivative brings in $\dot{\hat M}\dot q = (\hat C + \hat C^T)\,\dot q$:

$$
\dot z = -K z + K\,\bigl(\hat g(q) - \hat C^T(q,\dot q)\,\dot q - \tau - K \hat M(q)\,\dot q\bigr),
\qquad \hat\tau_d = z + K \hat M(q)\,\dot q
$$

This is the generalized momentum observer of De Luca et al. (2006), the usual choice for
sensorless collision detection; seen from the paper it is the same observer with $\hat N$
replaced by $\hat N - \dot{\hat M}\dot q$. Because $K$ is constant and diagonal the update
over one tick is exact, and it uses the measured tick time, so loop jitter cannot
destabilise it:

```python
# once per tick, before the new torque is sent; u_prev holds last tick's value
a = np.exp(-K * dt)                         # K: per-joint bandwidth, rad/s; dt: measured
z = a * z + (1 - a) * u_prev                # z over the interval the last torque was applied
p = K * (M_hat @ qd)                        # M_hat = model.mass_matrix(q) + diag(armature)
tau_d_hat = z + p
u_prev = g - C.T @ qd - tau_applied - p     # C from pin.computeCoriolisMatrix
```

**Check on a simulated plant.** URDF dynamics plus 0.01 kg·m² armature per joint, PD with
gravity compensation following a 0.7 Hz sine (5.8 rad/s² peak on J1), a step disturbance of
up to 1 N·m at t = 1 s, both observers at 500 Hz with exact $q$ and $\dot q$, $K$ = 2π · 5 Hz,
and $y$ = 7 for the paper's form:

| | error with an exact model | 63 % rise on J2 | 63 % rise on J6 |
|---|---|---|---|
| suggested form | ≤ 0.02 N·m | 32 ms (= $1/K$) | 32 ms |
| paper's form, $Y = 7\,I$ | ≤ 0.08 N·m | 66 to 74 ms | under one tick |

With the armature left out of $\hat M$, both report a phantom 0.06 N·m, which is
$J_a\,\ddot q$ = 0.01 × 5.8.

## 5. Concerns specific to this arm

- **The armature is not identified.** `feedforward.armature` is all zeros. The reflected
  rotor inertia ($N^2 J_{rotor}$, $N$ = 9 on the RS-06 and 10 on the RS-00) probably
  dominates the wrist, where the URDF gives $M_{66}$ = 0.0004 kg·m². Missing, it appears as
  a phantom torque $J_a\,\ddot q$ opposing every acceleration. At the playback limits
  (2 to 4 rad/s²) that is small, 0.1 N·m for 0.05 kg·m²; under a hand, at 20 to 50 rad/s², it
  is not. It also conditions the problem: 0.01 kg·m² per joint lowers the condition number
  of $\hat M$ from 2 800 to about 100. Identification: `docs/natural_frequency.md` section 5
  (ring-down), or the regression of section 6 below, which gives friction in the same pass.
- **The velocity signal caps the bandwidth.** The observer uses $\dot q$ directly, and here
  it is a finite difference of position low-passed with a 30 ms time constant
  (`free_drive.velocity_filter_time_constant`, about 5 Hz), because the RS firmware's
  velocity readout is not in rad/s. An observer faster than that filter estimates the
  filter's lag, $\hat M(\dot q - \dot q_{filt})$, as a disturbance during acceleration.
  Either the observer stays near 5 Hz or the velocity estimate has to improve first.
- **The applied torque is not exactly known.** In MIT mode the firmware adds the PD term
  from its own position and velocity. The host's `tau_cmd` reconstructs it from the host's
  samples, and `tau_meas` is the motor's report, whose freshness depends on which frames
  were decoded. Both should be fed to the observer on recorded data and compared before
  one is chosen.
- **Friction sets the force floor.** After the gravity re-scaling,
  `docs/tracking_error_analysis.md` leaves 0.3 to 0.5 N·m of stiction at the J3 dwells, plus
  a residual gravity-shape error and a position-locked ripple at 19.8 cycles per rad. A
  push below what remains after the friction model is subtracted is invisible, and at rest
  the sign of the friction torque is unknown, so the estimate needs a dead band there.
- **The torque scale.** A torque-constant error scales $\tau$ and so the estimate; the
  per-joint `tau_scale` (1.152 on J3) hints at one. Parameters identified from the same
  torque signal absorb it consistently.
- **Rate and bus.** The observer is one 6 × 6 product per tick and adds no CAN traffic, but
  the bus is already near its limit at 500 Hz, so the rate cannot be raised to buy
  bandwidth.

## 6. Identifying the armature and friction together

Per joint, on a recording of free motion, fit by least squares:

$$
\tau - \tau_{URDF}(q,\dot q,\ddot q) = J_a\,\ddot q + F_c\,\mathrm{sign}(\dot q) + F_v\,\dot q + \tau_0
$$

with $\tau_{URDF}$ from `inertial_torque` plus `gravity`. Offline, $\dot q$ and $\ddot q$ come
from a zero-phase filter, so there is no lag (the paper's Appendix B low-passes both sides
instead). The acceleration has to be large enough to see: a 0.1 rad sine at 2 Hz gives
16 rad/s² at 1.3 rad/s, which needs a higher acceleration limit but not a higher speed
limit. Two or three frequencies separate inertia, which grows with $f^2$, from viscous
friction, which grows with $f$. The scope CSV has no velocity column, which is fine for
this offline use. The result should agree with the ring-down, and afterwards the observer's
estimate in free motion should not correlate with $\ddot q$.

## 7. From the estimate to admittance

1. $\hat\tau_{ext} = \hat\tau_d + \hat F(\dot q)$, with a dead band sized from the free-motion
   noise floor.
2. Joint-space admittance first: $M_v\,\ddot q_r + D_v\,\dot q_r = \hat\tau_{ext}$ moves the
   reference the stiff position loop follows. It needs no Jacobian inverse.
3. Cartesian admittance after that: $\hat F = J^{-T}\hat\tau_{ext}$, which degrades near
   singularities and needs the damped inverse the IK already uses.

## 8. A design tool

The design equation is one line; the tool is worth building for what it shows. It fits as a
page of the pendant next to `plot.html`, not as a separate application:

- **Pole map.** The observer poles over the workspace against the sample-rate limit and
  the velocity-filter corner, for the paper's form and the suggested one.
- **Offline replay.** A scope recording run through the observer. In free motion the
  output is the noise floor, hence the smallest detectable push; it also compares
  `tau_cmd` against `tau_meas` as the torque input.
- **Simulator check.** `SimBackend.push()` applies a known joint torque, and the
  simulator has friction and an armature to hide from the model: ground truth for the
  estimate.

## 9. Suggested order of work

1. Observer module (suggested form, with the paper's form beside it for comparison) and
   tests against `SimBackend.push()`.
2. Offline replay on existing recordings: noise floor, and the choice of torque input.
3. Armature and friction identification (section 6); a better velocity estimate if the
   replay shows the 30 ms filter is the limit.
4. A live $\hat\tau_d$ trace in the scope.
5. The tuning page (section 8).
6. Joint-space admittance, then Cartesian.

## References

- A. Mohammadi, M. Tavakoli, H. J. Marquez, F. Hashemzadeh, "Nonlinear disturbance observer
  design for robotic manipulators", Control Engineering Practice, 2012,
  doi:10.1016/j.conengprac.2012.10.008.
- W.-H. Chen, D. J. Ballance, P. J. Gawthrop, J. O'Reilly, "A nonlinear disturbance observer
  for robotic manipulators", IEEE Trans. Industrial Electronics 47, 2000.
- A. De Luca, A. Albu-Schäffer, S. Haddadin, G. Hirzinger, "Collision detection and safe
  reaction with the DLR-III lightweight manipulator arm", IEEE/RSJ IROS, 2006.
