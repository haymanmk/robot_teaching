import dataclasses

import numpy as np
import pytest

from robot_teaching.backend.sim import SimBackend
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
    return dataclasses.replace(cfg, friction=dataclasses.replace(cfg.friction, coulomb=coulomb, velocity_scale=0.02))


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


def _with_gains(cfg, hold_kp=None, hold_kd=None, free_kp=None, coulomb=None):
    hold = dataclasses.replace(cfg.hold, kp=cfg.hold.kp if hold_kp is None else hold_kp,
                               kd=cfg.hold.kd if hold_kd is None else hold_kd)
    fd = dataclasses.replace(cfg.free_drive, kp=cfg.free_drive.kp if free_kp is None else free_kp)
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
        feedforward=dataclasses.replace(cfg.feedforward, inertia_scale=inertia_scale, armature=armature,
                                        max_torque=max_torque),
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
    assert ctl.gains()["feedforward"] == {"inertia_scale": 1.0, "armature": [armature] * 6, "max_torque": 3.0}
