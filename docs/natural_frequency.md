# Natural frequencies of the joint motors under PD control with gravity compensation

What the scope's spectrum shows as a peak is a mode of the closed loop: the arm's inertia
against the stiffness of the MIT position gains, damped by the velocity gains. This
walkthrough derives that mode from the dynamics model, computes it with the repository's own
code, checks it against the simulator, and shows how to turn a measured peak into the one
number the URDF lacks, the reflected rotor inertia. Every snippet runs with `uv run python`
from the repository root, in the order given.

## 1. From the equations of motion to a mass–spring–damper

The arm is a rigid-body chain:

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = \tau
$$

with $M(q)$ the joint-space inertia matrix, $C(q,\dot q)\,\dot q$ the Coriolis and centrifugal
torques and $g(q)$ gravity. The motors run the MIT impedance law with the gravity torque fed
forward:

$$
\tau = K_p\,(q_d - q) + K_d\,(\dot q_d - \dot q) + g(q)
$$

where $K_p = \mathrm{diag}(k_{p,i})$ and $K_d = \mathrm{diag}(k_{d,i})$ are the per-joint gains of
the vendor YAML. Holding a fixed target ($\dot q_d = 0$), write the error $e = q - q_d$. Two
approximations make the loop linear:

- The feed-forward cancels $g(q)$, so gravity drops out of the balance (the 5 % model error is
  discussed in section 6).
- For a small motion about $q_d$, $M(q) \approx M(q_d)$, and $C(q,\dot q)\,\dot q$, quadratic in
  velocity, is negligible.

What is left is a multi-degree-of-freedom mass–spring–damper:

$$
M(q_d)\,\ddot e + K_d\,\dot e + K_p\,e = 0
$$

The inertia matrix and the gains are both available in the code base:

```python
import numpy as np
from robot_teaching.model import RobotModel
from robot_teaching.backend.rebotarm import mit_gains_from_yaml

model = RobotModel(6)
q = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])          # the "L" clearance pose
M = model.mass_matrix(q)                                # 6 × 6, kg·m²
kp, kd = mit_gains_from_yaml(None, model.joint_names)   # MIT gains from the vendor YAML
```

`mass_matrix` wraps Pinocchio's `crba` on the URDF; `mit_gains_from_yaml` reads the
`MIT: kp / kd` entries of the hardware YAML, no arm needed.

## 2. One joint at a time

Keeping only the diagonal of $M$ gives each joint its own second-order system. The inertia
the motor accelerates is the link inertia the URDF knows about plus the rotor inertia
reflected through the gear, which it does not:

$$
J_i = M_{ii}(q) + J_{a,i}, \qquad J_{a,i} = N_i^2\, J_{\mathrm{rotor},i}
$$

($N$ is the reduction ratio, 9 on the RS-06 and 10 on the RS-00.) The equation
$J_i \ddot e_i + k_{d,i} \dot e_i + k_{p,i} e_i = 0$ is the standard form
$\ddot e + 2\zeta\omega_n \dot e + \omega_n^2 e = 0$, so

$$
\omega_{n,i} = \sqrt{\frac{k_{p,i}}{J_i}}, \qquad
f_{n,i} = \frac{\omega_{n,i}}{2\pi}, \qquad
\zeta_i = \frac{k_{d,i}}{2\sqrt{k_{p,i}\,J_i}}, \qquad
f_{d,i} = f_{n,i}\sqrt{1-\zeta_i^2}
$$

$f_n$ is the undamped natural frequency, $\zeta$ the damping ratio and $f_d$ the frequency a
decaying ring actually shows (it exists only for $\zeta < 1$).

```python
armature = np.zeros(6)                 # J_a per joint; 0 until identified (section 5)
J = np.diag(M) + armature
f_n = np.sqrt(kp / J) / (2 * np.pi)
zeta = kd / (2 * np.sqrt(kp * J))
f_d = f_n * np.sqrt(np.clip(1 - zeta**2, 0, None))
```

At the L pose with the shipped gains and no armature:

| joint | $k_p$ / $k_d$ | $M_{ii}$ (kg·m²) | $f_n$ (Hz) | $\zeta$ | $f_d$ (Hz) |
|---|---|---|---|---|---|
| 1 | 50 / 3 | 0.068 | 4.3 | 0.82 | 2.5 |
| 2 | 150 / 10 | 0.381 | 3.2 | 0.66 | 2.4 |
| 3 | 150 / 10 | 0.246 | 3.9 | 0.82 | 2.2 |
| 4 | 50 / 5 | 0.035 | 6.0 | 1.9 | – |
| 5 | 50 / 4 | 0.009 | 11.8 | 3.0 | – |
| 6 | 50 / 4 | 0.0004 | 55 | 14 | – |

Two things to read off. The frequency depends on the pose through $M_{ii}(q)$: joint 2 spans
2.4 Hz folded to 5.8 Hz at the home pose. And the wrist numbers are not credible: $M_{66}$ is so
small that any realistic armature dominates, so $J_a$ must be identified before the wrist
figures mean anything.

## 3. The joints are coupled

$M(q)$ is far from diagonal. At the L pose the shoulder–elbow entry is $M_{23} = -0.22$ against
$M_{22} = 0.38$ and $M_{33} = 0.25$, so the single-joint numbers are only a first guess. The full
system is analysed in state space, $x = (e, \dot e)$:

$$
\dot x = A\,x, \qquad
A = \begin{bmatrix} 0 & I \\ -M^{-1}K_p & -M^{-1}K_d \end{bmatrix}
$$

Each oscillatory mode is a complex pair of eigenvalues $\lambda = -\zeta\omega_n \pm j\,\omega_d$, so

$$
\omega_n = |\lambda|, \qquad \zeta = -\frac{\operatorname{Re}\lambda}{|\lambda|}, \qquad
\omega_d = \operatorname{Im}\lambda
$$

and the first half of the eigenvector is the mode shape: which joints move in that mode.

```python
n = len(kp)
A = np.block([[np.zeros((n, n)), np.eye(n)],
              [-np.linalg.solve(M, np.diag(kp)), -np.linalg.solve(M, np.diag(kd))]])
lam, V = np.linalg.eig(A)
for i in np.where(lam.imag > 1e-9)[0]:                 # one of each complex pair
    w = abs(lam[i])
    shape = np.abs(V[:n, i]) / np.abs(V[:n, i]).max()
    print(f"f_n {w/(2*np.pi):.2f} Hz  zeta {-lam[i].real/w:.2f}  "
          f"f_d {lam[i].imag/(2*np.pi):.2f} Hz  shape {shape.round(2)}")
```

At the L pose with the shipped gains this prints two oscillatory modes:

| mode | $f_n$ (Hz) | $\zeta$ | $f_d$ (Hz) | shape |
|---|---|---|---|---|
| shoulder–elbow–wrist pitch | 2.53 | 0.55 | 2.12 | joints 2 : 3 : 4 = 1.0 : 0.77 : 0.56 |
| base yaw | 4.21 | 0.80 | 2.51 | joint 1 |

The other eight eigenvalues are real, so overdamped. The coupled mode of joints 2–4 sits lower
in both frequency and damping than any of the three single-joint figures. A ring seen "on
joint 3" is this mode, and the spectrum should show it on joints 2 and 4 as well.

## 4. Check it in the simulator

The simulator integrates the full $M(q)$, $C$, $g$ and the MIT law, so it is a fair test of the
two approximations. Kick one joint away from the target, run PD plus gravity compensation,
and measure the decay. From the recorded error, $f_d$ comes from the zero crossings and the
decay rate $\sigma = \zeta\omega_n$ from the envelope of the peaks, $e_{\mathrm{peak}}(t) \propto e^{-\sigma t}$;
then

$$
\omega_n = \sqrt{\omega_d^2 + \sigma^2}, \qquad \zeta = \frac{\sigma}{\omega_n}
$$

`start_loop` and `step` are the simulator's test hooks: with `realtime=False`, each `step`
runs the control callback once and then one physics step, in that order, as on the hardware.

```python
from robot_teaching.backend.sim import SimBackend

def free_decay(joint, kp, kd, armature=0.0, kick=0.05, seconds=4.0, rate=500.0):
    sim = SimBackend(realtime=False, rate=rate, model=model, armature=armature,
                     coulomb_friction=0.0, viscous_friction=0.0)
    sim.connect(); sim.enable()
    q0 = q.copy(); q0[joint] += kick
    sim.set_state(q0)
    n = int(seconds * rate); t = np.arange(n) / rate; err = np.zeros(n); k = 0

    def control(dt):                      # one tick: PD + gravity compensation, as HOLD does
        nonlocal k
        qk, _ = sim.read_positions()
        sim.send_arm_mit(q, np.zeros(6), kp, kd, model.gravity(qk))
        err[k] = qk[joint] - q[joint]; k += 1

    sim.start_loop(control, rate)
    sim.step(n)
    return t, err

def measure(t, err):
    zc = np.where(np.diff(np.sign(err)) != 0)[0]
    f_d = 1.0 / (2 * np.mean(np.diff(t[zc[:6]])))
    pk_t, pk_a = [], []
    for a, b in zip(zc[:5], zc[1:6]):
        i = a + np.argmax(np.abs(err[a:b])); pk_t.append(t[i]); pk_a.append(abs(err[i]))
    sigma = -np.polyfit(pk_t, np.log(pk_a), 1)[0]
    w_n = np.hypot(2 * np.pi * f_d, sigma)
    return f_d, w_n / (2 * np.pi), sigma / w_n          # f_d, f_n, zeta

kd_test = kd.copy(); kd_test[2] = 3.0                   # elbow with a tenth of its damping
print(measure(*free_decay(2, kp, kd_test)))
```

The shipped gains hardly ring ($\zeta \approx 0.8$), so the damping is lowered for the test to
make the oscillation measurable:

| case | prediction | simulated |
|---|---|---|
| joint 1, $k_d$ = 0.5, single-joint formula | $f_n$ 4.33 Hz, $\zeta$ 0.14 | $f_n$ 4.29 Hz, $\zeta$ 0.14 |
| joint 3, $k_d$ = 3, single-joint formula | $f_n$ 3.93 Hz, $\zeta$ 0.25 | $f_n$ 2.81 Hz, $\zeta$ 0.39 |
| joint 3, $k_d$ = 3, coupled, lowest mode | $f_n$ 2.78 Hz, $\zeta$ 0.40 | the same run |

Base yaw is almost uncoupled and the single-joint formula lands within 1 %. The elbow needs
the coupled analysis: the single-joint formula is 40 % high. With the shipped $k_d$ = 10 the
elbow error is down to microradians within a second, with no visible ring.

## 5. Identify the rotor inertia from a measured ring

Inverting the single-joint formula turns a measured natural frequency into the inertia the
motor actually sees, and so into the part the URDF lacks:

$$
J_i = \frac{k_{p,i}}{\omega_{n,i}^2}, \qquad J_{a,i} = J_i - M_{ii}(q)
$$

```python
kd_test = kd.copy(); kd_test[0] = 0.5
f_d, f_n, zeta = measure(*free_decay(0, kp, kd_test, armature=0.05))   # 0.05 kg·m² hidden in the sim
J_est = kp[0] / (2 * np.pi * f_n) ** 2
print(f"J {J_est:.4f}  M_11 {M[0, 0]:.4f}  armature {J_est - M[0, 0]:.4f}")  # ≈ 0.05
```

Tested in the simulator with a hidden armature of 0.05 kg·m² on joint 1: the decay gives
$f_n$ = 3.27 Hz, hence $J$ = 0.118 kg·m² and $J_a$ = 0.118 − 0.068 = 0.051. On the real arm, use a
joint with little coupling (base yaw, or each wrist joint on its own), hold it with a low
$k_d$ at a clearance pose, tap it, and record the scope. The spectrum strip gives $f_d$; the
decay in the recording gives $\sigma$; section 4's formulas give $\omega_n$. The identified value
goes into `feedforward.armature` in the YAML, where the inverse-dynamics feed-forward uses it.

## 6. What the simple model leaves out

- **Gravity stiffness.** Linearising $g(q)$ adds $\partial g/\partial q$ to the stiffness, up to
  −10 N·m/rad on joint 2 at the L pose. The feed-forward evaluated at the measured $q$ cancels
  it; only the model error remains, about 5 % or 0.5 N·m/rad: nothing against $k_p$ = 150 and a
  few percent against the free-drive $k_p$ = 8.
- **Friction.** Coulomb friction of 0.2–0.5 N·m is not linear. It stops a decaying ring early
  and causes stick-slip; it does not shift $f_n$.
- **The motor loop.** The PD runs in the motor firmware on its own velocity estimate, and $k_p$
  is taken as an exact N·m/rad (the vendor's calibration rests on that too). Lag in the
  firmware's velocity estimate lowers the effective damping, which is why a measured $\zeta$
  below the prediction is the more informative mismatch.
- **Model error.** The URDF's gravity term was checked to 5–11 %; its inertia tensors were
  not, and the armature is unknown until identified.
