"""Disturbance observers against the simulator, where the disturbance is known."""

import numpy as np
import pytest

from robot_teaching.backend.sim import SimBackend
from robot_teaching.observer import ConstantGainObserver, MomentumObserver, inertia_bounds, paper_gain

from conftest import L_POSE

RATE = 500.0
PUSH = np.array([0.5, -1.0, 0.8, 0.2, -0.1, 0.05])      # N·m, applied from t = 0.5 s for 1 s
SWING = np.array([1.0, 0.5, 0.5, 1.0, 1.0, 1.0])         # relative amplitude of the reference per joint
OMEGA = 2.0 * np.pi * 0.7                                 # rad/s, reference sine


def run(model, observer, *, amplitude=0.0, push=None, seconds=2.0, torque="measured",
        coulomb=0.0, viscous=0.0, armature=0.0, jitter=False):
    """Hold (or swing about) the L pose under the simulator's MIT law with gravity fed forward,
    and feed the observer once per tick. Returns time, estimate and joint velocity per tick."""
    sim = SimBackend(realtime=False, rate=RATE, viscous_friction=viscous, coulomb_friction=coulomb,
                     armature=armature, q0=L_POSE, model=model)
    sim.connect()
    sim.enable()
    kp, kd = sim.hold_gains()
    last_cmd = np.zeros(model.n)
    log = []

    def tick(dt):
        nonlocal last_cmd
        t = sim.sim_time
        q, _ = sim.read_positions()
        qd = sim.read_velocities()
        # The torque applied since the last tick: the motors' report, or the command sent then.
        est = observer.update(q, qd, sim.read_torques() if torque == "measured" else last_cmd, dt)
        q_cmd = L_POSE + amplitude * np.sin(OMEGA * t) * SWING
        qd_cmd = amplitude * OMEGA * np.cos(OMEGA * t) * SWING
        ff = model.gravity(q)
        sim.send_arm_mit(q_cmd, qd_cmd, kp, kd, ff)
        last_cmd = kp * (q_cmd - q) + kd * (qd_cmd - qd) + ff
        log.append((t, est, qd))

    sim.start_loop(tick, RATE)
    rng = np.random.default_rng(1)
    for k in range(int(seconds * RATE)):
        if push is not None and k == int(0.5 * RATE):
            sim.push(push, 1.0)
        sim.step(1, dt=rng.uniform(0.6, 1.8) / RATE if jitter else None)
    return tuple(np.array(x) for x in zip(*log))


def rise_time(t, est, joint, target, t0=0.5):
    """Time after ``t0`` at which the estimate of ``joint`` first reaches 63 % of ``target``."""
    i0 = np.searchsorted(t, t0)
    k = i0 + np.argmax(np.abs(est[i0:, joint]) >= abs(target) * (1.0 - np.exp(-1.0)))
    return t[k] - t[i0]


def at(t, x, when):
    return x[np.searchsorted(t, when)]


# ── model term the observers add ───────────────────────────────────────────────────────────

def test_coriolis_matrix_gives_the_inertia_rate_and_the_coriolis_torque(model):
    q = L_POSE + np.array([0.3, -0.2, 0.1, 0.4, -0.3, 0.2])
    qd = np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.3])
    C = model.coriolis_matrix(q, qd)
    assert C.shape == (6, 6)
    assert np.allclose(C @ qd + model.gravity(q), model.nonlinear_effects(q, qd))
    h = 1e-6
    M_dot = (model.mass_matrix(q + h * qd) - model.mass_matrix(q - h * qd)) / (2 * h)
    assert np.allclose(C + C.T, M_dot, atol=1e-6), "dM/dt = C + C^T is what the momentum observer rests on"


# ── momentum observer ──────────────────────────────────────────────────────────────────────

def test_first_update_initialises_and_returns_zero(model):
    obs = MomentumObserver(model, 5.0)
    assert np.allclose(obs.estimate, 0.0)
    out = obs.update(L_POSE, np.full(6, 0.5), np.ones(6), 0.002)      # already moving: still zero
    assert np.allclose(out, 0.0) and np.allclose(obs.estimate, 0.0)


def test_exact_model_in_free_motion_reads_zero(model):
    t, est, _ = run(model, MomentumObserver(model, 5.0), amplitude=0.3)
    assert np.abs(est[t > 0.3]).max() < 0.03          # N·m, against joint torques of several N·m


def test_commanded_torque_of_the_previous_tick_works_as_the_input(model):
    # The host does not see the PD term the motor adds inside the tick, so this is coarser.
    t, est, _ = run(model, MomentumObserver(model, 5.0), amplitude=0.3, torque="commanded")
    assert np.abs(est[t > 0.3]).max() < 0.08


def test_push_is_followed_with_the_set_bandwidth_on_every_joint(model):
    f = 5.0
    t, est, _ = run(model, MomentumObserver(model, f), push=PUSH)
    assert np.allclose(at(t, est, 1.4), PUSH, atol=0.01), "settles on the push"
    assert np.allclose(est[-1], 0.0, atol=0.01), "and returns to zero when it ends"
    tau = 1.0 / (2.0 * np.pi * f)
    for j in range(6):                                 # shoulder and wrist alike: one time constant
        assert rise_time(t, est, j, PUSH[j]) == pytest.approx(tau, abs=2.5 / RATE)


def test_push_is_found_while_the_arm_moves(model):
    t, est, _ = run(model, MomentumObserver(model, 5.0), amplitude=0.3, push=PUSH)
    assert np.allclose(at(t, est, 1.4), PUSH, atol=0.03)


def test_bandwidth_per_joint(model):
    f = np.array([2.0, 5.0, 5.0, 10.0, 10.0, 20.0])
    t, est, _ = run(model, MomentumObserver(model, f), push=PUSH)
    for j in range(6):
        assert rise_time(t, est, j, PUSH[j]) == pytest.approx(1.0 / (2.0 * np.pi * f[j]), abs=2.5 / RATE)


def test_friction_is_reported_as_a_low_passed_negative_disturbance(model):
    f, coulomb, viscous = 5.0, 0.3, 0.2
    t, est, qd = run(model, MomentumObserver(model, f), amplitude=0.3, coulomb=coulomb, viscous=viscous)
    friction = coulomb * np.tanh(qd / 0.05) + viscous * qd          # the simulator's law
    expected, a = np.zeros_like(est), np.exp(-2.0 * np.pi * f / RATE)
    for k in range(1, len(t)):                                        # tau_d = -F through the observer's filter
        expected[k] = a * expected[k - 1] + (1.0 - a) * -friction[k - 1]
    assert np.abs(friction).max() > 0.4
    assert np.abs(est - expected)[t > 0.3].max() < 0.04


def test_armature_missing_from_the_model_is_a_phantom_torque(model):
    armature, amplitude = 0.05, 0.3
    t, est, _ = run(model, MomentumObserver(model, 5.0), amplitude=amplitude, armature=armature)
    phantom = armature * amplitude * OMEGA ** 2 * SWING              # J_a · peak acceleration
    assert np.abs(est[t > 0.3]).max(axis=0) == pytest.approx(phantom, rel=0.2)
    t, est, _ = run(model, MomentumObserver(model, 5.0, armature=armature), amplitude=amplitude, armature=armature)
    assert np.abs(est[t > 0.3]).max() < 0.03, "given the armature, the phantom is gone"


def test_loop_jitter_does_not_disturb_the_estimate(model):
    t, est, _ = run(model, MomentumObserver(model, 5.0), amplitude=0.3, push=PUSH, jitter=True)
    assert np.allclose(at(t, est, 1.4), PUSH, atol=0.06)
    assert np.abs(est[t > t[-1] - 0.2]).max() < 0.06, "back near zero after the push"


def test_gap_longer_than_max_dt_holds_the_estimate(model):
    obs = MomentumObserver(model, 5.0, max_dt=0.1)
    g = model.gravity(L_POSE)
    obs.update(L_POSE, np.zeros(6), g, 0.002)
    for _ in range(500):                               # at rest, the motors 1 N·m short of gravity on J2
        obs.update(L_POSE, np.zeros(6), g - np.array([0, 1.0, 0, 0, 0, 0]), 0.002)
    before = obs.estimate
    assert before[1] == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(obs.update(L_POSE, np.zeros(6), np.zeros(6), 5.0), before), "no integration over a stall"
    assert np.allclose(obs.update(L_POSE, np.zeros(6), g - np.array([0, 1.0, 0, 0, 0, 0]), 0.002), before, atol=1e-3)
    obs.reset()
    assert np.allclose(obs.estimate, 0.0)


def test_changing_the_bandwidth_keeps_the_estimate(model):
    obs = MomentumObserver(model, 5.0)
    g, qd = model.gravity(L_POSE), np.full(6, 0.2)
    short = np.array([0.3, 1.0, -0.5, 0.1, 0.1, 0.02])
    for _ in range(400):
        obs.update(L_POSE, qd, g - short, 0.002)
    before = obs.estimate
    obs.set_bandwidth(20.0)
    assert np.allclose(obs.estimate, before)
    assert np.allclose(obs.update(L_POSE, qd, g - short, 0.002), before, atol=0.02), "no step when the gain changes"


def test_bad_arguments_are_rejected(model):
    with pytest.raises(ValueError):
        MomentumObserver(model, 0.0)
    with pytest.raises(ValueError):
        MomentumObserver(model, 5.0, armature=-0.1)
    with pytest.raises(ValueError):
        MomentumObserver(model, 5.0).update(np.zeros(3), np.zeros(6), np.zeros(6), 0.002)
    with pytest.raises(ValueError):
        ConstantGainObserver(model, np.zeros((6, 6)))


# ── constant-gain observer of Mohammadi et al. ─────────────────────────────────────────────

def test_paper_design_constants_of_this_arm(model, limits):
    b = inertia_bounds(model, limits[0], samples=2000)
    assert b.sigma1 < 1e-3 and 0.8 < b.sigma2 < 1.2 and 1.0 < b.zeta < 2.0
    assert b.condition > 1000, "without the armature the inertia matrix is badly conditioned"
    assert paper_gain(b, 6.0) == pytest.approx(0.5 * b.zeta + 6.0 * b.sigma2)
    assert inertia_bounds(model, limits[0], armature=0.01, samples=2000).condition < 150


def test_constant_gain_converges_but_its_poles_spread_with_the_inertia(model, limits):
    y = paper_gain(inertia_bounds(model, limits[0], samples=2000), 6.0)
    obs = ConstantGainObserver(model, y)
    poles = obs.poles(L_POSE).real
    assert poles[0] > 6.0, "the slowest pole keeps the rate asked for"
    assert poles[-1] > 2.0 * RATE, "the fastest is beyond what a 500 Hz loop can integrate explicitly"
    t, est, _ = run(model, obs, push=PUSH)
    assert np.allclose(at(t, est, 1.4), PUSH, atol=0.01), "the implicit update keeps it stable and it settles"
    assert rise_time(t, est, 1, PUSH[1]) > 10 * rise_time(t, est, 5, PUSH[5]), "shoulder far slower than the wrist"
    t, est, _ = run(model, ConstantGainObserver(model, y), amplitude=0.3)
    assert np.abs(est[t > 0.3]).max() < 0.1
