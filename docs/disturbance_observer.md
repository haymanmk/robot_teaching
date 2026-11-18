# Disturbance observer for sensorless force estimation: design note

**Status (2026-11-17).** The observer exists as a module, `src/robot_teaching/observer.py`,
and the controller runs it every tick: its estimate is a scope trace, and a fourth
feed-forward switch feeds it back in hold and playback (section 4). Both are tested against
the simulator. A first pair of recordings on the arm (J3, section 4) confirms the estimate
and the torque input. They also showed the observer and the friction feed-forward cancelling
friction twice when both were on, which has since been corrected in the controller and
checked on the simulator, not yet on the arm. The rest is a design note. It records what the nonlinear
disturbance observer (NDOB) of Mohammadi et al. gives on this arm, the concerns that came
out of checking it against the arm's model and control loop, and the form and order of work
suggested instead. The numbers were computed from the URDF with the snippets below
(`uv run python` from the repository root).

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
# once per tick; tau_applied is the torque the motors applied since the previous tick
k_p = K * (M_hat @ qd)                      # K: per-joint bandwidth, rad/s; M_hat = M(q) + diag(armature)
w = g - C.T @ qd - k_p                      # C = model.coriolis_matrix(q, qd)
a = np.exp(-K * dt)                         # dt: the measured tick time
z = a * z + (1 - a) * (0.5 * (w_prev + w) - tau_applied)
tau_d_hat = z + k_p
w_prev = w
```

The pose-dependent terms are taken as the mean of the two ends of the interval, which
removes the half-tick lag a value held from the previous tick would add.

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

### The module

`src/robot_teaching/observer.py` has both forms behind one interface:

```python
from robot_teaching.model import RobotModel
from robot_teaching.observer import MomentumObserver, ConstantGainObserver, inertia_bounds, paper_gain

model = RobotModel(6)                                   # the calling thread's own model
obs = MomentumObserver(model, bandwidth_hz=5.0, armature=None)   # scalar or one value per joint
# every control tick, with the torque applied since the previous tick and the measured dt:
#   tau_d_hat = obs.update(q, qd, tau_applied, dt)

bounds = inertia_bounds(model, [1.0, 1.0, 1.0, 1.5, 1.5, 2.0])  # sigma1, sigma2, zeta of section 3
paper = ConstantGainObserver(model, paper_gain(bounds, beta=6.0))
print(paper.poles([0.0, 0.7, 1.1, 0.0, 0.0, 0.0]).real)          # 12 to 16 800 rad/s at the L pose
```

- `update` returns zeros on the first call (the estimate starts from zero wherever the arm
  is and however it moves), and holds the estimate over a gap longer than `max_dt` instead
  of integrating over it. `set_bandwidth` changes the corner without a step in the estimate.
- The torque argument is the torque applied *since the previous call*: the motors' report
  read this tick, or the command sent at the previous tick.
- `ConstantGainObserver` uses a backward-Euler update, so its fast poles stay stable.

`tests/test_observer.py` runs them on `SimBackend`, 500 Hz, about the L pose, with the
simulator's hold gains and gravity fed forward:

| case | result (momentum form, 5 Hz) |
|---|---|
| exact model, 0.3 rad swing at 0.7 Hz | estimate within 0.02 N·m of zero |
| same, torque input = the previous tick's command | within 0.05 N·m: the host does not see the PD term the motor adds inside the tick |
| `push()` of 0.05 to 1 N·m per joint | settles on the push within 0.01 N·m; 63 % rise 32 ms on all six joints |
| per-joint bandwidths of 2 to 20 Hz | each joint rises with its own $1/K$ |
| simulator friction 0.3 N·m Coulomb + 0.2 N·m·s/rad | estimate = −friction through the 5 Hz low-pass, within 0.04 N·m |
| 0.05 kg·m² armature hidden from the model | phantom 0.29 N·m = $J_a\,\ddot q$; gone when the armature is given |
| tick time jittered between 0.6 and 1.8 periods | push still found within 0.06 N·m |
| paper's form, $y$ = 7.0 | settles on the push; 63 % rise 84 ms on J2, one tick on J6 |

The simulator gives the observer exact positions and velocities, so these confirm the
equations and the discretisation. They say nothing yet about the arm's filtered velocity
and its torque signal (section 5).

### In the controller

`TeachController` owns a `MomentumObserver` and advances it at the start of every tick while
the motors are enabled, with the controller's own filtered velocity estimate.

- **Its model is the feed-forward's**: `free_drive.tau_scale × g(q)` and
  `feedforward.armature`. With the URDF's unscaled gravity the observer would report the
  difference between the two gravity models as a disturbance, and feeding that back would
  cancel the calibration.
- **The torque it is given** is the one applied since the previous tick (`observer.torque`):
  `measured`, the motors' report, falling back per joint to the command where a motor
  reports none; or `commanded`, the MIT law the host evaluated at the previous tick.
- **The friction feed-forward is taken out of that torque first.** It is a model of a
  disturbance, like the gravity scale, but one the observer's equations have no place for.
  Left in the applied torque, it does not change what the observer reports (the whole
  friction), so the compensation would cancel friction a second time. Taken out, the
  estimate is what the feed-forward leaves over, and the two add up to the friction. The
  inertia feed-forward stays in: the observer's model already expects the torque an
  acceleration needs.
- **The estimate** $\hat\tau_d$ is the trace field `tau_d`: the scope's "disturbance
  estimate" and the `tau_d_Nm` column of a recording, whatever the switch says.
- **The compensation** is the trace field `tau_o` $= -w\,\mathrm{clip}(\hat\tau_d,
  \pm$`observer.max_torque`$)$, added to the feed-forward. The weight $w$ ramps to 1 over
  `free_drive.transition_duration` while `feedforward.observer` is on and the mode is hold
  or playback, and to 0 otherwise, so it fades out in free drive and never steps.
- **It restarts from zero** after the motors were off (e-stop) and after a control-loop
  error, when the applied torque is not known.

Because the compensation is part of the torque the observer is told about, the estimate
does not depend on whether it is fed back: in the ideal case there is no loop through the
observer, only the low-pass on the true disturbance. What can close a loop is an error in
the two signals it trusts, the velocity and the torque (section 5).

`tests/test_controller.py`, on the simulator with the shipped configuration (0.3 N·m of
Coulomb friction and 0.2 N·m·s/rad viscous per joint; the controller's velocity is the
30 ms filtered finite difference, as on the arm):

| case | observer off | observer on |
|---|---|---|
| playback of the test program, largest tracking error, `measured`, 2 Hz | 9.9 mrad | 4.3 mrad |
| same, `commanded`, 2 Hz | 9.9 mrad | 3.8 mrad |
| same, `measured`, 5 Hz | 9.9 mrad | 4.2 mrad |
| hold, arm 10 % heavier than the model: sag | 4.3 mrad | below 0.1 mrad |
| hold under a push of 0.05 to 1 N·m: deflection | up to 10 mrad | at most 0.1 mrad |
| free drive with the switch on | | `tau_o` is exactly zero after the fade; a push is estimated, not opposed |

With `commanded`, the peak estimate on J2 during playback is 0.86 N·m against 0.39 N·m with
`measured`. The excess is the damping term the host mis-evaluates: it computes
$k_d(\dot q_{cmd} - \dot q)$ with its lagging velocity estimate, while the motor uses its
own. That is the error section 5 warns of, and the reason `measured` is the default; which
of the two is better on the real arm depends on how fresh the motors' torque report is, and
is still to be checked there. One observer update costs about 20 μs on the development
machine, a whole controller tick about 50 μs of the 2 ms period.

**First use on the arm.** Leave the switch off and record a move with the scope: the
estimate should be smooth, about the size of the friction, with the sign opposing the
motion. Compare a recording per `observer.torque` setting. Only then switch it on, in hold
first, starting from a low `observer.max_torque`.

### First recordings on the arm (2026-11-17)

J3 swinging over the short stroke (0.737 to 1.107 rad) at 50 % speed, gravity and friction
feed-forward on (0.37 N·m), observer at 2 Hz with `measured` torque, one recording with the
observer switch off and one with it on (`data/dob_test/`, not under version control).

- **The estimate is consistent.** It is the same in both recordings, as section 4 predicts:
  −0.19 N·m moving up and +0.57 N·m moving down, that is 0.38 N·m of friction against the
  motion and a gravity feed-forward about 0.19 N·m too high. The configured friction is 0.37.
  Above 5 Hz it carries 0.07 N·m RMS of noise.
- **`measured` is the right torque input.** The reported torque minus the host's commanded
  torque is, to about 90 %, $-20\,(\dot q - \dot q_{filt})$: the joint's $k_d$ is 20, so the
  difference is the damping term the host evaluates with its lagging velocity. The motors'
  report is the truthful one, and it leads the host's figure by 15 to 23 ms.
- **Friction is compensated twice when both switches are on.** The observer's model has no
  friction, and the friction feed-forward is part of the torque it is told about, so it
  reports the whole friction and its compensation stacks on the feed-forward. The mean error
  went from +0.03 / +1.34 mrad (moving up / down) to +0.96 / −2.07 mrad: from lagging to
  leading, with the RMS barely changed (2.24 to 1.94 mrad). *Corrected the same day:* the
  controller now takes the friction feed-forward out of the torque the observer is given
  ("In the controller" above). On the simulator, with the feed-forward matched to its
  0.3 N·m of friction, the joints lagged by 4.7 mrad with neither term, led by 4.3 mrad with
  both before the correction, and are within 0.3 mrad with both after it; the largest error
  is then 4.1 mrad against 8.5 mrad with neither, 4.5 with the friction term alone and 4.2
  with the observer alone. The recordings on the arm have not been repeated yet.
- **What the observer can reach.** With it off the error is 1.8 mrad RMS below 1 Hz
  (friction and the gravity offset, which it can remove), 1.0 mrad RMS between 1 and 15 Hz
  (mostly the position-locked ripple near 5.6 Hz at this speed, above its bandwidth) and
  0.14 mrad above.

### Why the observer does not remove the ripple (2026-11-17)

Five recordings of the same J3 stroke (`data/dob_test/`): observer off and on at 50 % speed
with a 2 Hz bandwidth, on with 4 Hz, and off and on at 100 %. The ripple is the
position-locked one of `docs/tracking_error_analysis.md`, 19.8 cycles per rad, so 5.6 Hz at
50 % speed and 11 Hz at 100 %.

- **The torque traces show the reaction, not the ripple.** The motor's reported torque comes
  from its current; cogging and gear mesh act on the rotor after it, so they are in neither
  the commanded nor the reported torque. With the observer off, the reported torque ripple
  is the motor's PD acting on the position ripple: 566 N·m/rad at −103° against 716 at −102°
  from $k_p$ = 150 and $k_d$ = 20. Feedback needs the error to exist before it pushes back.
- **The estimate arrives late.** Against the ripple torque rebuilt without lag from the
  recorded position and torque (URDF inertia for J3, 0.246 kg·m², no armature), the estimate
  is 0.5 times its size and 31 to 39 ms (63° to 79°) late at 2 Hz, 0.6 to 0.7 times and 21 to
  26 ms (42° to 51°) late at 4 Hz, and 0.4 times and 26 ms (104°) late at 100 % speed.
  Subtracting it leaves 90 to 97 %, 76 to 81 % and 116 to 121 % of the ripple torque. A
  first-order estimate with corner $f_c$ leaves $f/\sqrt{f^2+f_c^2}$ of a disturbance at
  $f$: 94 % and 81 % for the first two. Removing 90 % would need $f_c \approx 10 f$, 56 Hz
  at 50 % speed, against a velocity estimate filtered at 5.3 Hz.
- **The position ripple confirms it:** 0.52 / 0.69 mrad RMS (up / down) with the observer
  off, 0.64 / 0.80 at 2 Hz and 0.70 / 0.74 at 4 Hz.

The ripple repeats with the joint angle, so it is fed forward from a table of the angle
instead (`src/robot_teaching/ripple.py`, the fifth feed-forward term; README, "Ripple
feed-forward"). What the recordings say about such a table:

- The position-locked part of the motor torque is repeatable: tables fitted to two 25 %
  recordings made minutes apart agree to within 9 % RMS, and same-day recordings at 50 % and
  at 100 % to about 30 %. A table from the 25 % recordings of 2026-10-29 supplies 40 to 60 %
  of what the feedback applied in the 50 % recordings three weeks later.
- It differs between the two directions of travel and is not a single sine (at 50 %, one
  sine at 19.8 cycles per rad explains 85 % of the detrended error moving up and a third
  moving down), hence one curve per direction and a table, not a harmonic model.
- Adding the URDF's $M_{33}\,\ddot q$ to the motor torque made the curves of different
  speeds and directions agree less, not more, so the table is taken from the motor torque
  alone at low speed, where the inertia term is small, and refined by repeating the
  recording with the table in use: the feedback's remaining share then moves into the table.
  With the arm's gains the share left after one round at 25 % speed should be about 20 %.
- The observer's model has no ripple either, so the table's torque is taken out of the
  torque the observer is given, like the friction term's. Without that the simulator shows
  the two cancelling the ripple twice (0.57 mrad of position ripple against 0.11).

Simulator, 0.5 N·m of ripple on J3: 1.67 mrad of position ripple at the learning speed
becomes 0.69, 0.25 and 0.10 mrad over three rounds; the observer alone leaves 0.96. None of
this has been run on the arm yet.

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
   tests against `SimBackend.push()`. *Done 2026-11-16, section 4.*
2. Offline replay on existing recordings: noise floor, and the choice of torque input.
3. Armature and friction identification (section 6); a better velocity estimate if the
   replay shows the 30 ms filter is the limit.
4. A live $\hat\tau_d$ trace in the scope, and the estimate fed back as a feed-forward term
   in hold and playback. *Done 2026-11-17, section 4. Not yet run on the arm; steps 2 and 3
   are what make the estimate trustworthy there.*
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
