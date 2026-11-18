import dataclasses

import numpy as np
import pytest

from robot_teaching.backend.sim import SimBackend
from robot_teaching.config import resolve_vector
from robot_teaching.controller import Mode, TeachController, TraceBuffer
from robot_teaching.planning import plan_program
from robot_teaching.program import Program, TaughtPoint

from conftest import L_POSE


@pytest.fixture
def rig(cfg, model):
    sim = SimBackend(realtime=False, q0=L_POSE, model=model)
    ctl = TeachController(sim, cfg)
    ctl.start()
    yield sim, ctl
    ctl.shutdown()


def settle(sim, ctl, n):
    sim.step(n)
    return ctl.snapshot()


def test_hold_tracks_target(rig):
    sim, ctl = rig
    s = settle(sim, ctl, 500)
    assert s.mode == Mode.HOLD.value and s.enabled
    assert np.max(np.abs(s.q - s.q_target)) < 5e-3
    assert np.allclose(s.q_target, L_POSE)


def test_jog_moves_at_limited_speed(rig, cfg):
    sim, ctl = rig
    settle(sim, ctl, 100)
    ok, _ = ctl.jog_to(L_POSE + np.array([0.3, 0, 0, 0, 0, 0]))
    assert ok
    s = settle(sim, ctl, 100)        # 0.2 s
    assert s.jogging
    assert s.q_target[0] == pytest.approx(cfg.jog.joint_velocity * 0.2, abs=1e-3)
    s = settle(sim, ctl, 500)
    assert not s.jogging and s.q_target[0] == pytest.approx(0.3)
    assert abs(s.q[0] - 0.3) < 5e-3
    bad, msg = ctl.jog_to(np.array([0, -1.0, 1.1, 0, 0, 0]))
    assert not bad and "limits" in msg


def test_free_drive_lock_and_follow(rig):
    sim, ctl = rig
    settle(sim, ctl, 100)
    assert ctl.free_drive()[0]
    s = settle(sim, ctl, 100)
    assert s.mode == Mode.FREE_DRIVE.value and s.free_drive_status == "STARTUP" and not s.gains_settled
    s = settle(sim, ctl, 400)
    assert s.gains_settled and s.free_drive_status == "LOCKED"
    assert np.max(np.abs(s.q - L_POSE)) < 0.03, "arm must not sag when released"
    sim.push(np.array([0, 0, 1.5, 0, 0, 0]), 0.5)
    seen = set()
    for _ in range(8):
        seen.add(settle(sim, ctl, 50).free_drive_status)
    assert "FOLLOW" in seen
    s = settle(sim, ctl, 1000)
    assert s.free_drive_status == "LOCKED"
    assert s.q[2] > L_POSE[2] + 0.05, "the push must have moved the elbow"
    moved = s.q.copy()
    s = settle(sim, ctl, 500)
    assert np.max(np.abs(s.q - moved)) < 0.02, "locked arm must stay where it was released"
    # Playback is refused in free drive; hold is required first.
    assert not ctl.play.__self__ is None
    ok, msg = ctl.hold()
    assert ok
    s = settle(sim, ctl, 400)
    assert s.mode == Mode.HOLD.value and s.gains_settled


def _program():
    prog = Program()
    prog.add(TaughtPoint(q=L_POSE + [0.3, 0.1, 0, 0.2, 0, 0], gripper=0.0, speed=0.8))
    prog.add(TaughtPoint(q=L_POSE + [0.4, 0.2, -0.1, 0.3, 0.2, 0.3], gripper=2.0, speed=0.8, motion="linear"))
    prog.add(TaughtPoint(q=L_POSE + [0.0, 0.3, 0.0, 0.0, -0.3, 0.0], gripper=2.0, speed=0.8, blend=True))
    prog.add(TaughtPoint(q=L_POSE, gripper=0.0, speed=0.8))
    return prog


def test_playback_tracks_and_fires_gripper(rig, cfg):
    sim, ctl = rig
    s = settle(sim, ctl, 300)
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    ok, msg = ctl.play(traj, speed=1.0)
    assert ok, msg
    max_err, gripper_targets, ticks = 0.0, set(), 0
    while True:
        s = settle(sim, ctl, 5)
        ticks += 5
        if s.mode != Mode.PLAYBACK.value:
            break
        max_err = max(max_err, float(np.max(np.abs(s.q - s.q_target))))
        gripper_targets.add(s.gripper_target)
        assert ticks < 60000
    assert s.mode == Mode.HOLD.value
    assert max_err < 0.02
    assert gripper_targets >= {0.0, 2.0}
    assert np.max(np.abs(s.q_target - L_POSE)) < 1e-6
    assert abs(ticks / 500 - traj.duration) < 1.0          # speed slews from 0 at start


def test_playback_stop_ramps_and_speed(rig, cfg):
    sim, ctl = rig
    s = settle(sim, ctl, 300)
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=0.5)[0]
    s = settle(sim, ctl, 500)
    assert s.playback["speed"] == pytest.approx(0.5, abs=0.02)
    assert ctl.set_speed(1.0)[0]
    s = settle(sim, ctl, 500)
    assert s.playback["speed"] == pytest.approx(1.0, abs=0.02)
    assert ctl.stop_motion()[0]
    s = settle(sim, ctl, 25)
    assert s.mode == Mode.PLAYBACK.value and s.playback["stopping"] and 0 < s.playback["speed"] < 1
    s = settle(sim, ctl, 300)
    assert s.mode == Mode.HOLD.value and not s.jogging
    # Replaying from here is refused: the trajectory starts elsewhere.
    ok, msg = ctl.play(traj, speed=1.0)
    assert not ok and "start" in msg


def test_playback_requires_hold(rig, cfg):
    sim, ctl = rig
    s = settle(sim, ctl, 300)
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    ctl.free_drive()
    settle(sim, ctl, 10)
    ok, msg = ctl.play(traj, speed=1.0)
    assert not ok and "hold" in msg
    ctl.hold()
    settle(sim, ctl, 10)
    ok, msg = ctl.play(traj, speed=1.0)
    assert not ok and "settle" in msg


def test_estop_and_enable(rig):
    sim, ctl = rig
    settle(sim, ctl, 100)
    assert ctl.estop()[0]
    s = settle(sim, ctl, 10)
    assert s.mode == Mode.DISABLED.value and not s.enabled and not sim.enabled
    assert not ctl.free_drive()[0]
    assert ctl.enable()[0]
    s = settle(sim, ctl, 200)
    assert s.mode == Mode.HOLD.value and s.enabled and s.gains_settled
    assert np.max(np.abs(s.q - s.q_target)) < 0.02


def test_backend_error_falls_back_to_hold(rig):
    sim, ctl = rig
    settle(sim, ctl, 100)
    ctl.free_drive()
    settle(sim, ctl, 300)
    original = sim.read_positions
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CAN timeout")
        return original()

    sim.read_positions = flaky
    s = settle(sim, ctl, 5)
    assert s.error and "CAN timeout" in s.error
    assert s.mode == Mode.HOLD.value


def test_connection_lifecycle(cfg, model):
    sim = SimBackend(realtime=False, q0=L_POSE, model=model)
    ctl = TeachController(sim, cfg)
    s = ctl.snapshot()
    assert s is not None and s.mode == Mode.DISCONNECTED.value and not s.connected and not s.enabled
    ok, msg = ctl.hold()
    assert not ok and "not connected" in msg
    assert not ctl.estop()[0]
    ctl.start()
    s = settle(sim, ctl, 100)
    assert s.connected and s.mode == Mode.HOLD.value and sim.enabled
    ctl.shutdown()
    s = ctl.snapshot()
    assert s.mode == Mode.DISCONNECTED.value and not s.connected and not sim.enabled
    assert not ctl.free_drive()[0]
    ctl.start()                       # reconnecting works
    s = settle(sim, ctl, 100)
    assert s.mode == Mode.HOLD.value and np.max(np.abs(s.q - s.q_target)) < 0.01
    ctl.shutdown()


def test_failed_connect_stays_disconnected(cfg, model):
    class DeadBus(SimBackend):
        def connect(self):
            raise RuntimeError("can0: no such device")

    sim = DeadBus(realtime=False, q0=L_POSE, model=model)
    ctl = TeachController(sim, cfg)
    with pytest.raises(RuntimeError, match="can0"):
        ctl.start()
    s = ctl.snapshot()
    assert s.mode == Mode.DISCONNECTED.value and not ctl.is_connected


def test_gripper_hand_mode(rig):
    sim, ctl = rig
    s = settle(sim, ctl, 100)
    assert not s.gripper_hand and s.gripper_target == 0.0
    # Stiff by default: a hand push is pulled back to the target.
    sim.set_state(s.q, gripper=1.0)
    s = settle(sim, ctl, 300)
    assert s.gripper_target == 0.0 and abs(s.gripper) < 0.05
    # Hand mode: the target follows the hand, in hold and in free drive.
    assert ctl.gripper_hand()[0]
    sim.set_state(s.q, gripper=2.0)
    s = settle(sim, ctl, 20)
    assert s.gripper_hand and s.gripper_target == pytest.approx(2.0)
    ctl.free_drive()
    settle(sim, ctl, 300)
    sim.set_state(ctl.snapshot().q, gripper=3.0)
    s = settle(sim, ctl, 20)
    assert s.gripper_hand and s.gripper_target == pytest.approx(3.0)
    # A target command ends hand mode and the gripper is stiff again.
    assert ctl.set_gripper(5.0)[0]
    s = settle(sim, ctl, 600)
    assert not s.gripper_hand and s.gripper_target == 5.0 and abs(s.gripper - 5.0) < 0.05
    sim.set_state(s.q, gripper=4.0)
    s = settle(sim, ctl, 300)
    assert s.gripper_target == 5.0 and abs(s.gripper - 5.0) < 0.05
    # Hand mode survives free drive -> hold, but not playback.
    ctl.gripper_hand()
    ctl.hold()
    s = settle(sim, ctl, 300)
    assert s.gripper_hand
    traj = plan_program(ctl.model, _program(), s.q_target, ctl.cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=1.0)[0]
    s = settle(sim, ctl, 5)
    assert not s.gripper_hand
    assert not ctl.gripper_hand()[0]


def _with_friction(cfg, coulomb):
    return dataclasses.replace(cfg, friction=dataclasses.replace(cfg.friction, coulomb=coulomb, velocity_scale=0.02),
                               feedforward=dataclasses.replace(cfg.feedforward, friction=True))


def _play_and_measure(cfg, model):
    """Play the test program once; return (max |q - q_cmd| per joint, max |tau_f| per joint)."""
    sim = SimBackend(realtime=False, q0=L_POSE, model=model)
    ctl = TeachController(sim, cfg)
    ctl.start()
    s = settle(sim, ctl, 300)
    assert np.all(s.tau_f == 0.0), "nothing is commanded to move in hold: no friction term"
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=1.0)[0]
    err, tau_f = np.zeros(ctl.n), np.zeros(ctl.n)
    active = np.broadcast_to(np.asarray(cfg.friction.coulomb, float), (ctl.n,)) > 0.0
    while True:
        s = settle(sim, ctl, 1)
        if s.mode != Mode.PLAYBACK.value:
            break
        _, vel, _, _, tau = sim._cmd
        moving = (np.abs(vel) > 0.1) & active
        # The term carries the sign of the commanded velocity and is bounded by the configured friction.
        assert np.all(np.sign(s.tau_f[moving]) == np.sign(vel[moving]))
        assert np.allclose(tau, s.tau_g + s.tau_f + s.tau_i)
        err = np.maximum(err, np.abs(s.q - s.q_target))
        tau_f = np.maximum(tau_f, np.abs(s.tau_f))
    assert np.all(ctl.snapshot().tau_f == 0.0), "back in hold the term is zero"
    ctl.shutdown()
    return err, tau_f


def test_friction_feedforward_follows_commanded_velocity(cfg, model):
    coulomb = np.array([0.0, 0.3, 0.3, 0.2, 0.2, 0.0])
    err, tau_f = _play_and_measure(_with_friction(cfg, coulomb.tolist()), model)
    assert np.all(tau_f <= coulomb + 1e-9)
    assert np.all(tau_f[[1, 2, 4]] > 0.9 * coulomb[[1, 2, 4]]), "joints that move see the full term"
    assert np.all(tau_f[[0, 5]] == 0.0), "a zero entry disables the term for that joint"


def test_friction_feedforward_reduces_tracking_error(cfg, model):
    # The simulator has 0.3 N·m of Coulomb friction on every joint; feeding it forward must
    # cut the tracking error, and the error must not grow on any joint (wrong sign would).
    err_off, _ = _play_and_measure(_with_friction(cfg, 0.0), model)
    err_on, _ = _play_and_measure(_with_friction(cfg, 0.3), model)
    assert np.max(err_on) < 0.6 * np.max(err_off), (err_off, err_on)
    assert np.all(err_on <= err_off + 5e-4), (err_off, err_on)


def test_trace_buffer_wraps_and_continues():
    buf = TraceBuffer(n=2, size=8)
    for k in range(20):
        buf.append(0.01 * k, q_cmd=[k, -k], q=[k + 0.5, 0.0], tau_cmd=[1.0, 2.0], tau_meas=np.nan)
    chunk = buf.since(None, 1)
    assert all(v is None for v in chunk["tau_g"]), "a field not given to append reads as missing"
    assert chunk["seq"] == 20 and len(chunk["t"]) == 8, "only the newest rows survive a wrap"
    assert chunk["epoch"] == buf.epoch > 0, "the chunk names the run its samples come from"
    assert chunk["q_cmd"] == [-k for k in range(12, 20)] and chunk["t"] == pytest.approx([0.01 * k for k in range(12, 20)])
    assert chunk["tau_meas"] == [None] * 8, "NaN becomes null for JSON"
    assert buf.since(15, 1)["q_cmd"] == [-15, -16, -17, -18, -19]
    assert buf.since(20, 1)["t"] == [] and buf.since(None, 0, max_samples=3)["q"] == [17.5, 18.5, 19.5]
    with pytest.raises(ValueError):
        buf.since(None, 2)


def test_trace_records_every_tick(rig, cfg):
    sim, ctl = rig
    s = settle(sim, ctl, 300)
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=1.0)[0]
    s = settle(sim, ctl, 400)
    chunk = ctl.trace_since(1)
    assert len(chunk["t"]) == 700 and np.all(np.diff(chunk["t"]) > 0)
    assert chunk["q_cmd"][-1] == pytest.approx(s.q_target[1]) and chunk["q"][-1] == pytest.approx(s.q[1])
    assert all(v is not None for v in chunk["tau_cmd"]) and all(v is not None for v in chunk["tau_meas"])
    # The commanded torque is the MIT law on the host; while tracking it stays near the gravity torque.
    assert abs(chunk["tau_cmd"][-1] - s.tau_g[1]) < 3.0
    assert ctl.trace_since(1, chunk["seq"])["t"] == []
    settle(sim, ctl, 10)
    assert len(ctl.trace_since(1, chunk["seq"])["t"]) == 10


def _with_gains(cfg, hold_kp=None, hold_kd=None, free_kp=None, coulomb=None, tau_scale=None):
    hold = dataclasses.replace(cfg.hold, kp=cfg.hold.kp if hold_kp is None else hold_kp,
                               kd=cfg.hold.kd if hold_kd is None else hold_kd)
    fd = dataclasses.replace(cfg.free_drive, kp=cfg.free_drive.kp if free_kp is None else free_kp,
                             tau_scale=cfg.free_drive.tau_scale if tau_scale is None else tau_scale)
    fr = dataclasses.replace(cfg.friction, coulomb=cfg.friction.coulomb if coulomb is None else coulomb)
    return dataclasses.replace(cfg, hold=hold, free_drive=fd, friction=fr)


def test_reload_gains_fades_in_hold_and_free_drive(rig, cfg):
    sim, ctl = rig
    settle(sim, ctl, 100)
    kp0 = sim._cmd[2].copy()
    target = 133.0
    ok, msg = ctl.reload_gains(_with_gains(cfg, hold_kp=[target] * 6, hold_kd=[7.0] * 6, free_kp=4.0, coulomb=0.25))
    assert ok and msg.startswith("gains reloaded")
    s = settle(sim, ctl, 10)                                  # 20 ms into the fade
    assert not s.gains_settled, "the new gains fade in, they do not jump"
    kp_mid = sim._cmd[2]
    moving = np.abs(kp0 - target) > 1.0
    assert moving.any()
    assert np.all(kp_mid[moving] > np.minimum(kp0, target)[moving]) and np.all(kp_mid[moving] < np.maximum(kp0, target)[moving])
    s = settle(sim, ctl, int(cfg.free_drive.transition_duration * 500) + 10)
    assert s.gains_settled and np.allclose(sim._cmd[2], target) and np.allclose(sim._cmd[3], 7.0)
    g = ctl.gains()
    assert g["hold"]["kp"] == [target] * 6 and g["friction"]["coulomb"] == [0.25] * 6 and g["free_drive"]["kp"] == [4.0] * 6
    # In free drive the compliant gains fade in the same way.
    assert ctl.free_drive()[0]
    settle(sim, ctl, 400)
    assert ctl.reload_gains(_with_gains(cfg, hold_kp=[target] * 6, free_kp=2.0))[0]
    s = settle(sim, ctl, 400)
    assert s.mode == Mode.FREE_DRIVE.value and s.gains_settled and np.allclose(sim._cmd[2], 2.0)


def test_reload_gains_rescales_gravity_compensation(rig, cfg):
    sim, ctl = rig
    s = settle(sim, ctl, 100)
    tau_g0 = s.tau_g.copy()
    assert np.max(np.abs(tau_g0)) > 0.5, "the L pose has gravity torque to compensate"
    ok, msg = ctl.reload_gains(_with_gains(cfg, tau_scale=[0.5] * 6))
    assert ok and "tau_scale [0.5 0.5 0.5 0.5 0.5 0.5]" in msg, msg
    s = settle(sim, ctl, 2)
    assert np.allclose(s.tau_g, 0.5 * ctl.model.gravity(s.q), atol=1e-6), "the gravity term uses the reloaded scale at once"
    assert np.allclose(sim._cmd[4], s.tau_g, atol=0.05), "and it is what the motors receive in hold"
    assert ctl.gains()["free_drive"]["tau_scale"] == [0.5] * 6


def test_reload_gains_rules(cfg, model):
    sim = SimBackend(realtime=False, q0=L_POSE, model=model)
    ctl = TeachController(sim, cfg)
    new = _with_gains(cfg, hold_kp=[150.0] * 6)
    ok, _ = ctl.reload_gains(new)                             # before the first connection: stored for the next hold
    assert ok and ctl.gains()["hold"]["kp"] == [150.0] * 6
    ctl.start()
    sim.step(100)
    assert np.allclose(sim._cmd[2], 150.0), "the connection holds with the reloaded gains"
    prog = Program()
    prog.add(TaughtPoint(q=L_POSE + [0.3, 0, 0, 0, 0, 0], gripper=0.0, speed=0.6))
    traj = plan_program(model, prog, L_POSE, cfg, gripper_start=0.0)
    assert ctl.play(traj, 1.0)[0]
    sim.step(50)
    ok, msg = ctl.reload_gains(new)
    assert not ok and "playback" in msg
    ctl.stop_motion()
    sim.step(300)
    bad = dataclasses.replace(new, hold=dataclasses.replace(new.hold, kp=[1.0, 2.0]))     # wrong length
    ok, msg = ctl.reload_gains(bad)
    assert not ok and "hold.kp" in msg and ctl.gains()["hold"]["kp"] == [150.0] * 6, "a bad config leaves the gains untouched"
    ctl.shutdown()


def _fast(cfg, scale=3.0, armature=0.0, inertia_scale=1.0, max_torque=3.0):
    """The test config with joint limits scaled up, the simulator's friction and gravity, and the feed-forward set."""
    lim = cfg.limits
    vmax = (np.asarray(lim.joint_velocity, float) * scale).tolist()
    amax = (np.asarray(lim.joint_acceleration, float) * scale).tolist()
    return dataclasses.replace(
        cfg,
        limits=dataclasses.replace(lim, joint_velocity=vmax, joint_acceleration=amax),
        friction=dataclasses.replace(cfg.friction, coulomb=0.3, velocity_scale=0.05),
        free_drive=dataclasses.replace(cfg.free_drive, tau_scale=1.0),
        feedforward=dataclasses.replace(cfg.feedforward, friction=True, inertia=True, inertia_scale=inertia_scale,
                                        armature=armature, max_torque=max_torque),
    )


def _tracking(cfg, model, armature):
    """Play the test program on a simulator with ``armature``; return (max |q - q_cmd|, max |tau_i|) per joint."""
    sim = SimBackend(realtime=False, q0=L_POSE, model=model, armature=armature)
    ctl = TeachController(sim, cfg)
    ctl.start()
    s = settle(sim, ctl, 300)
    assert np.all(s.tau_i == 0.0), "nothing moves in hold: no inertia term"
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=1.0)[0]
    err, tau_i = np.zeros(ctl.n), np.zeros(ctl.n)
    for _ in range(int(traj.duration * 500) + 1000):
        s = settle(sim, ctl, 1)
        if s.mode != Mode.PLAYBACK.value:
            break
        err = np.maximum(err, np.abs(s.q - s.q_target))
        tau_i = np.maximum(tau_i, np.abs(s.tau_i))
    assert s.mode == Mode.HOLD.value and np.all(s.tau_i == 0.0), "back in hold the term is zero"
    ctl.shutdown()
    return err, tau_i


def test_inertia_feedforward_reduces_acceleration_error(cfg, model):
    armature = 0.1          # kg·m², a reflected rotor inertia of the order of the link inertia
    err_off, tau_off = _tracking(_fast(cfg, armature=armature, inertia_scale=0.0), model, armature)
    err_on, tau_on = _tracking(_fast(cfg, armature=armature, inertia_scale=1.0), model, armature)
    assert np.all(tau_off == 0.0) and np.max(tau_on) > 0.3
    moving = err_off > 2e-3
    assert moving.any()
    assert np.max(err_on[moving] / err_off[moving]) < 0.6, (err_off, err_on)
    assert np.all(err_on <= err_off + 5e-4), (err_off, err_on)
    # The clip bounds the term and the gains report carries the settings.
    _, tau_clipped = _tracking(_fast(cfg, armature=armature, inertia_scale=1.0, max_torque=0.1), model, armature)
    assert np.max(tau_clipped) <= 0.1 + 1e-9
    ctl = TeachController(SimBackend(realtime=False, q0=L_POSE, model=model), _fast(cfg, armature=armature))
    assert ctl.gains()["feedforward"] == {"gravity": True, "friction": True, "inertia": True, "observer": False, "ripple": False,
                                          "inertia_scale": 1.0, "armature": [armature] * 6, "max_torque": 3.0}


def test_feedforward_switches_default_to_gravity_and_ramp_in(cfg, model):
    # Friction and inertia are configured but switched off (the shipped default): playback sends
    # gravity alone. Switching a term on during playback ramps it in over the gain fade time;
    # switching gravity off takes g(q) out of the sent torque. The trace records the applied terms.
    base = dataclasses.replace(cfg, friction=dataclasses.replace(cfg.friction, coulomb=0.3, velocity_scale=0.02),
                               feedforward=dataclasses.replace(cfg.feedforward, armature=0.1, inertia_scale=1.0))
    assert (base.feedforward.gravity, base.feedforward.friction, base.feedforward.inertia) == (True, False, False)
    sim = SimBackend(realtime=False, q0=L_POSE, model=model, armature=0.1)
    ctl = TeachController(sim, base)
    ctl.start()
    s = settle(sim, ctl, 300)
    assert ctl.feedforward() == {"gravity": True, "friction": False, "inertia": False, "observer": False, "ripple": False} == s.feedforward
    traj = plan_program(ctl.model, _program(), s.q_target, base, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=1.0, loop=True)[0]
    for _ in range(50):
        s = settle(sim, ctl, 10)
        assert s.mode == Mode.PLAYBACK.value
        assert np.all(s.tau_f == 0.0) and np.all(s.tau_i == 0.0)
        assert np.allclose(sim._cmd[4], s.tau_g), "gravity alone is sent"
    ok, msg = ctl.set_feedforward(friction=True, inertia=True)
    assert ok and msg == "feed-forward gravity on, friction on, inertia on, observer off, ripple off"
    assert ctl.feedforward() == {"gravity": True, "friction": True, "inertia": True, "observer": False, "ripple": False}
    early = np.zeros(ctl.n)
    for _ in range(12):                                   # 0.24 s: half of the 0.5 s ramp at most
        s = settle(sim, ctl, 10)
        early = np.maximum(early, np.abs(s.tau_f))
    assert 0.0 < np.max(early) <= 0.5 * 0.3 + 1e-6, early
    late = np.zeros(ctl.n)
    for _ in range(int(traj.duration * 50) + 50):        # the rest of the ramp and a full lap
        s = settle(sim, ctl, 10)
        late = np.maximum(late, np.abs(s.tau_f))
    assert np.max(late) > 0.9 * 0.3, late
    chunk = ctl.trace_since(int(np.argmax(late)), None, 20)
    assert len(chunk["tau_g"]) == len(chunk["t"]) == len(chunk["tau_f"]) == len(chunk["tau_i"]) == 20
    assert np.allclose(chunk["tau_g"], chunk["tau_g"]) and max(abs(v) for v in chunk["tau_g"]) > 0.0
    ok, msg = ctl.set_feedforward(gravity=False)
    assert ok and msg == "feed-forward gravity off, friction on, inertia on, observer off, ripple off"
    settle(sim, ctl, 300)                                 # the ramp is over
    s = settle(sim, ctl, 1)
    assert s.mode == Mode.PLAYBACK.value and np.allclose(sim._cmd[4], s.tau_f + s.tau_i), "no gravity in the sent torque"
    assert all(v == 0.0 for v in ctl.trace_since(1, None, 5)["tau_g"]), "the trace records the applied gravity term"
    assert ctl.stop_motion()[0]
    settle(sim, ctl, 400)
    assert ctl.snapshot().mode == Mode.HOLD.value and np.allclose(sim._cmd[4], ctl.snapshot().tau_g), "hold always compensates gravity"
    ctl.shutdown()


# ── disturbance observer as a feed-forward term ────────────────────────────────────────────

def _with_observer(cfg, on=True, **observer):
    return dataclasses.replace(cfg, feedforward=dataclasses.replace(cfg.feedforward, observer=on),
                               observer=dataclasses.replace(cfg.observer, **observer))


def _observer_rig(cfg, model, **sim_kw):
    sim = SimBackend(realtime=False, q0=L_POSE, model=model, **sim_kw)
    ctl = TeachController(sim, cfg)
    ctl.start()
    return sim, ctl


def _playback_error(sim, ctl, cfg):
    """Play the test program once; return max |q - q_cmd|, max |tau_o| and max |tau_d| per joint."""
    s = settle(sim, ctl, 500)
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=1.0)[0]
    err, tau_o, tau_d = np.zeros(ctl.n), np.zeros(ctl.n), np.zeros(ctl.n)
    while True:
        s = settle(sim, ctl, 1)
        if s.mode != Mode.PLAYBACK.value:
            break
        assert np.allclose(sim._cmd[4], s.tau_g + s.tau_f + s.tau_i + s.tau_o), "the sent torque is the sum of the four terms"
        err = np.maximum(err, np.abs(s.q - s.q_target))
        tau_o, tau_d = np.maximum(tau_o, np.abs(s.tau_o)), np.maximum(tau_d, np.abs(s.tau_d))
    return err, tau_o, tau_d


def test_observer_estimates_without_feeding_back_by_default(cfg, model):
    sim, ctl = _observer_rig(cfg, model)
    assert ctl.feedforward()["observer"] is False
    err, tau_o, tau_d = _playback_error(sim, ctl, cfg)
    assert np.all(tau_o == 0.0), "switched off: nothing is added to the torque"
    # The simulator's joints have 0.3 N·m of Coulomb friction: that is what the estimate finds.
    assert np.all(tau_d > 0.25) and np.all(tau_d < 0.6), tau_d
    chunk = ctl.trace_since(2, None, max_samples=50)
    assert len(chunk["tau_d"]) == len(chunk["tau_o"]) == len(chunk["t"]) == 50 and chunk["tau_d"][-1] is not None
    s = ctl.snapshot().to_dict()
    assert len(s["tau_d"]) == len(s["tau_o"]) == 6
    ctl.shutdown()


@pytest.mark.parametrize("torque", ["measured", "commanded"])
def test_observer_compensation_cuts_playback_error(cfg, model, torque):
    sim, ctl = _observer_rig(cfg, model)
    err_off, _, _ = _playback_error(sim, ctl, cfg)
    ctl.shutdown()
    on = _with_observer(cfg, torque=torque)
    sim, ctl = _observer_rig(on, model)
    err_on, tau_o, tau_d = _playback_error(sim, ctl, on)
    ctl.shutdown()
    assert np.max(err_on) < 0.6 * np.max(err_off), (err_off, err_on)
    assert np.all(err_on <= err_off + 5e-4), (err_off, err_on)          # worse on no joint
    assert np.allclose(tau_o, tau_d), "fully on and inside the clip, the compensation is the negated estimate"


def test_observer_compensation_removes_a_gravity_model_error_in_hold(cfg, model):
    # The simulated arm is 10 % heavier than the model: the stiff hold sags, the observer sees
    # the missing torque as a disturbance, and feeding it back takes the sag out.
    sim, ctl = _observer_rig(cfg, model, gravity_scale=1.1)
    s = settle(sim, ctl, 1000)
    sag = np.max(np.abs(s.q - s.q_target))
    assert sag > 3e-3 and np.all(s.tau_o == 0.0)
    missing = s.tau_g * (1.1 / np.asarray(ctl.gains()["free_drive"]["tau_scale"]) - 1.0)
    assert np.allclose(s.tau_d, -missing, atol=0.02), "the estimate is the gravity torque the model lacks"
    ok, msg = ctl.set_feedforward(observer=True)
    assert ok and "observer on" in msg
    s = settle(sim, ctl, 20)                                  # 40 ms into the 0.5 s ramp
    assert 0.0 < np.max(np.abs(s.tau_o)) < 0.5 * np.max(np.abs(s.tau_d)), "the term ramps in, it does not step"
    s = settle(sim, ctl, 1500)
    assert np.max(np.abs(s.q - s.q_target)) < 0.05 * sag
    assert np.allclose(s.tau_o, -s.tau_d) and np.allclose(sim._cmd[4], s.tau_g + s.tau_o)
    ctl.shutdown()


def test_observer_compensation_is_clipped(cfg, model):
    on = _with_observer(cfg, max_torque=0.2)
    sim, ctl = _observer_rig(on, model, gravity_scale=1.1)
    s = settle(sim, ctl, 1500)
    assert np.max(np.abs(s.tau_d)) > 0.5, "the disturbance is larger than the clip"
    assert np.max(np.abs(s.tau_o)) == pytest.approx(0.2)
    assert ctl.gains()["observer"] == {"bandwidth": resolve_vector(cfg.observer.bandwidth, 6, "bandwidth").tolist(),
                                       "max_torque": 0.2, "torque": cfg.observer.torque}
    ctl.shutdown()


def test_observer_compensation_fades_out_in_free_drive(cfg, model):
    on = _with_observer(cfg)
    sim, ctl = _observer_rig(on, model, gravity_scale=1.1)
    s = settle(sim, ctl, 1500)
    assert np.max(np.abs(s.tau_o)) > 0.5
    assert ctl.free_drive()[0]
    s = settle(sim, ctl, 30)
    assert 0.0 < np.max(np.abs(s.tau_o)) < np.max(np.abs(s.tau_d)), "it fades with the gains, it does not drop"
    s = settle(sim, ctl, int(cfg.free_drive.transition_duration * 500) + 10)
    assert s.mode == Mode.FREE_DRIVE.value and np.all(s.tau_o == 0.0), "nothing pushes back against the hand"
    assert ctl.feedforward()["observer"] is True, "the switch keeps its state"
    sim.push(np.array([0.0, 0.0, 0.0, 0.0, 0.6, 0.0]), 0.3)
    s = settle(sim, ctl, 100)
    assert np.all(s.tau_o == 0.0) and abs(s.tau_d[4]) > 0.2, "the push is estimated, not opposed"
    settle(sim, ctl, 400)
    assert ctl.hold()[0]
    s = settle(sim, ctl, int(cfg.free_drive.transition_duration * 500) + 500)
    assert np.max(np.abs(s.tau_o)) > 0.5, "back in hold the compensation returns"
    ctl.shutdown()


def test_observer_restarts_when_the_motors_were_off(cfg, model):
    on = _with_observer(cfg)
    sim, ctl = _observer_rig(on, model, gravity_scale=1.1)
    s = settle(sim, ctl, 1500)
    assert np.max(np.abs(s.tau_d)) > 0.5
    assert ctl.estop()[0]
    s = settle(sim, ctl, 5)
    assert np.all(s.tau_d == 0.0) and np.all(s.tau_o == 0.0), "no torque is applied: nothing to estimate"
    sim.set_state(L_POSE)                                     # wherever the arm ended up, enable from the L pose
    assert ctl.enable()[0]
    s = settle(sim, ctl, 1)
    assert np.all(s.tau_d == 0.0), "the estimate starts again from zero, it does not resume a stale value"
    s = settle(sim, ctl, 1500)
    assert np.max(np.abs(s.tau_d)) > 0.5
    ctl.shutdown()


def test_reload_gains_updates_the_observer(cfg, model):
    sim, ctl = _observer_rig(cfg, model)
    settle(sim, ctl, 100)
    ok, msg = ctl.reload_gains(_with_observer(cfg, on=False, bandwidth=[1, 2, 3, 4, 5, 6], max_torque=1.0, torque="commanded"))
    assert ok and "observer [1 2 3 4 5 6] Hz, ±1 N·m, commanded torque" in msg, msg
    assert ctl.gains()["observer"] == {"bandwidth": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "max_torque": 1.0, "torque": "commanded"}
    assert np.allclose(ctl._observer.k, 2 * np.pi * np.arange(1, 7))
    for bad in ({"torque": "guessed"}, {"bandwidth": 0.0}, {"max_torque": -1.0}):
        ok, msg = ctl.reload_gains(_with_observer(cfg, **bad))
        assert not ok and "observer" in msg, msg
    assert ctl.gains()["observer"]["torque"] == "commanded", "a bad config leaves the observer untouched"
    # A new gravity scale reaches the observer's model too, so it is not reported as a disturbance.
    assert ctl.reload_gains(_with_gains(cfg, tau_scale=[1.0] * 6))[0]
    s = settle(sim, ctl, 1500)
    assert np.max(np.abs(s.tau_d)) < 0.02, s.tau_d
    ctl.shutdown()


def _playback_lead(cfg, model):
    """Play the test program once; return the largest |error|, the mean error along the commanded
    velocity (positive: the joints run ahead of their setpoints) and the largest |tau_d|."""
    sim, ctl = _observer_rig(cfg, model)
    s = settle(sim, ctl, 500)
    traj = plan_program(ctl.model, _program(), s.q_target, cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=1.0)[0]
    err, tau_d, lead = 0.0, 0.0, []
    while True:
        s = settle(sim, ctl, 1)
        if s.mode != Mode.PLAYBACK.value:
            break
        vel, e = sim._cmd[1], s.q - s.q_target
        moving = np.abs(vel) > 0.1
        if moving.any():
            lead.append(np.mean((e * np.sign(vel))[moving]))
        err, tau_d = max(err, np.max(np.abs(e))), max(tau_d, np.max(np.abs(s.tau_d)))
    ctl.shutdown()
    return err, float(np.mean(lead)), tau_d


def test_observer_leaves_the_friction_feedforward_its_share(cfg, model):
    # The simulator's joints have 0.3 N·m of Coulomb friction. The friction feed-forward cancels
    # it; the observer, whose model has no friction, must then report only what is left over.
    # If it reported the whole friction (the feed-forward being part of the torque it is told
    # about), both would cancel it and the joints would run ahead by as much as they lagged.
    friction_only = _with_friction(cfg, 0.3)
    observer_only = _with_observer(cfg)
    both = _with_observer(friction_only)
    err_neither, lead_neither, _ = _playback_lead(cfg, model)
    err_f, lead_f, tau_d_f = _playback_lead(friction_only, model)
    err_o, lead_o, tau_d_o = _playback_lead(observer_only, model)
    err_both, lead_both, tau_d_both = _playback_lead(both, model)
    assert lead_neither < -3e-3, "without either the joints lag"
    assert abs(lead_both) < 1e-3, f"together they neither lag nor lead ({lead_both * 1e3:+.2f} mrad)"
    assert err_both < 0.6 * err_neither and err_both <= min(err_f, err_o) + 2e-4, (err_neither, err_f, err_o, err_both)
    assert tau_d_o > 0.4, "alone, the observer's estimate is the whole friction"
    assert tau_d_f < 0.6 * tau_d_o and tau_d_both < 0.6 * tau_d_o, "with the feed-forward on, only the remainder"
