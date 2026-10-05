import numpy as np
import pytest

from robot_teaching.backend.sim import SimBackend
from robot_teaching.controller import Mode, TeachController
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
