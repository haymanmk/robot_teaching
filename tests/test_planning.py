import numpy as np
import pytest

import pinocchio as pin

from robot_teaching.planning import (
    PlanningError,
    Trajectory,
    min_jerk_duration,
    plan_joint_path,
    plan_linear_block,
    plan_linear_segment,
    plan_program,
    quintic_coeffs,
    quintic_eval,
)
from robot_teaching.program import Program, TaughtPoint

from conftest import L_POSE


def test_quintic_boundary_conditions():
    c = quintic_coeffs([0.0, 1.0], [0.2, 0.0], [0.0, 0.5], [1.0, -1.0], [-0.1, 0.3], [0.0, 0.0], 2.0)
    q, qd, qdd = quintic_eval(c, 0.0)
    assert np.allclose(q, [0.0, 1.0]) and np.allclose(qd, [0.2, 0.0]) and np.allclose(qdd, [0.0, 0.5])
    q, qd, qdd = quintic_eval(c, 2.0)
    assert np.allclose(q, [1.0, -1.0]) and np.allclose(qd, [-0.1, 0.3]) and np.allclose(qdd, [0.0, 0.0])


def test_min_jerk_duration_matches_peaks():
    T = min_jerk_duration([1.0], [1.0], [10.0])
    c = quintic_coeffs([0.0], [0.0], [0.0], [1.0], [0.0], [0.0], T)
    _, qd, qdd = quintic_eval(c, np.linspace(0, T, 1001))
    assert np.max(np.abs(qd)) == pytest.approx(1.0, rel=1e-3)
    assert np.max(np.abs(qdd)) <= 10.0 + 1e-6


def test_joint_path_limits_and_waypoints(limits):
    vmax, amax = limits
    W = np.array([L_POSE, L_POSE + [0.5, 0.2, 0.2, 0.2, 0, 0], L_POSE + [0.8, -0.1, -0.1, 0.1, 0.3, 0], L_POSE + [0.2, 0.1, 0.1, 0, 0, 0]])
    res = plan_joint_path(W, [True, False, False, True], vmax, amax, 0.01)
    assert res.t[0] == 0.0 and np.all(np.diff(res.t) > 0)
    assert np.max(np.abs(res.qd), axis=0).max() <= vmax.max() * 1.001
    assert np.all(np.max(np.abs(res.qd), axis=0) <= vmax * 1.001)
    assert np.all(np.max(np.abs(res.qdd), axis=0) <= amax * 1.001)
    for k, w in zip(res.knot_times, W):
        i = int(np.argmin(np.abs(res.t - k)))
        assert np.max(np.abs(res.q[i] - w)) < 2e-3
    assert np.allclose(res.qd[0], 0) and np.allclose(res.qd[-1], 0)
    # The via points carry velocity (not a stop-and-go interpolation).
    i1 = int(np.argmin(np.abs(res.t - res.knot_times[1])))
    assert np.max(np.abs(res.qd[i1])) > 0.1
    # Velocity is consistent with the position samples (C1 continuity).
    fd = np.diff(res.q, axis=0) / np.diff(res.t)[:, None]
    assert np.max(np.abs(fd - 0.5 * (res.qd[1:] + res.qd[:-1]))) < 0.02


def test_joint_path_stop_flags_give_zero_velocity(limits):
    vmax, amax = limits
    W = np.array([L_POSE, L_POSE + 0.3, L_POSE + 0.6])
    res = plan_joint_path(W, [True, True, True], vmax, amax, 0.01)
    i1 = int(np.argmin(np.abs(res.t - res.knot_times[1])))
    assert np.max(np.abs(res.qd[i1])) < 1e-2


def test_joint_path_segment_speed_scales_time(limits):
    vmax, amax = limits
    W = np.array([L_POSE, L_POSE + 0.5])
    fast = plan_joint_path(W, [True, True], vmax, amax, 0.01, [1.0])
    slow = plan_joint_path(W, [True, True], vmax, amax, 0.01, [0.25])
    assert slow.t[-1] > 2.0 * fast.t[-1]


def test_linear_segment_is_straight(model, limits):
    vmax, amax = limits
    qb = L_POSE + np.array([0.3, 0.1, -0.1, 0.05, 0.1, 0.2])
    tr = plan_linear_segment(model, L_POSE, qb, vmax, amax, 0.15, 0.3, 0.8, 2.0, 0.01, speed=1.0, margin=0.02)
    pa, pb = model.fk(L_POSE).translation, model.fk(qb).translation
    for s in np.linspace(0, tr.duration, 40):
        pm = model.fk(tr.sample(s)[0]).translation
        assert np.linalg.norm(np.cross(pm - pa, pb - pa)) / np.linalg.norm(pb - pa) < 1e-4
    assert np.max(np.abs(tr.q[-1] - qb)) < 1e-3
    lin = max(model.ee_velocity(tr.q[i], tr.qd[i])[0] for i in range(len(tr.t)))
    assert lin <= 0.15 * 1.05


def test_linear_unreachable_raises(model, limits):
    vmax, amax = limits
    far = np.array([0.0, 1.57, 0.0, 0.0, 0.0, 0.0])        # arm stretched horizontally
    # Target far outside the reach: fold the arm so the straight line passes through the base.
    behind = np.array([3.0, 0.3, 0.3, 0.0, 0.0, 0.0])
    with pytest.raises(PlanningError):
        plan_linear_segment(model, far, behind, vmax, amax, 0.15, 0.3, 0.8, 2.0, 0.01, speed=1.0, margin=0.02)


def test_plan_program_structure(model, cfg):
    prog = Program()
    prog.add(TaughtPoint(q=L_POSE + [0.3, 0, 0, 0, 0, 0], gripper=0.0, speed=0.6))
    prog.add(TaughtPoint(q=L_POSE + [0.5, 0.2, 0, 0, 0, 0], gripper=0.0, speed=0.6, blend=True))
    prog.add(TaughtPoint(q=L_POSE + [0.2, 0.3, 0.1, 0, 0, 0], gripper=2.0, speed=0.6))
    prog.add(TaughtPoint(q=L_POSE + [0.2, 0.2, 0.2, 0.1, 0, 0], gripper=2.0, motion="linear", speed=0.5, dwell=0.4))
    traj = plan_program(model, prog, L_POSE, cfg, gripper_start=0.0, loop=True)
    assert traj.t[0] == 0.0 and np.all(np.diff(traj.t) > 0)
    assert len(traj.point_times) == 5                      # 4 points + closing move
    assert traj.loop_start_time == pytest.approx(traj.point_times[0])
    kinds = [(e.kind, e.value) for e in traj.events]
    assert ("gripper", 2.0) in kinds and ("gripper", 0.0) in kinds
    gripper_t = [e.time for e in traj.events if e.kind == "gripper" and e.value == 2.0][0]
    assert gripper_t == pytest.approx(traj.point_times[2])
    # dwell + gripper settle hold after point 4 (linear point).
    q_at_arrival, _ = traj.sample(traj.point_times[3])
    q_after, _ = traj.sample(traj.point_times[3] + 0.3)
    assert np.allclose(q_at_arrival, q_after, atol=1e-9)
    # Blend point is passed through with velocity; stop points are at rest.
    _, qd_blend = traj.sample(traj.point_times[1])
    _, qd_stop = traj.sample(traj.point_times[2])
    assert np.max(np.abs(qd_blend)) > 0.05 and np.max(np.abs(qd_stop)) < 1e-2
    assert np.max(np.abs(traj.q[-1] - prog.points[0].q_array())) < 1e-9
    assert np.max(np.abs(traj.q[0] - L_POSE)) < 1e-9


def test_plan_program_rejects_bad_points(model, cfg):
    prog = Program()
    prog.add(TaughtPoint(q=[0, -0.3, 1.1, 0, 0, 0]))
    with pytest.raises(PlanningError):
        plan_program(model, prog, L_POSE, cfg)
    with pytest.raises(PlanningError):
        plan_program(model, Program(), L_POSE, cfg)


def test_trajectory_sampling_and_concat():
    t = np.array([0.0, 1.0, 2.0])
    q = np.array([[0.0], [1.0], [1.0]])
    qd = np.array([[0.0], [1.0], [0.0]])
    a = Trajectory(t, q, qd)
    assert a.sample(0.5)[0][0] == pytest.approx(0.5)
    assert a.sample(5.0)[0][0] == 1.0
    b = Trajectory(np.array([0.0, 1.0]), np.array([[1.0], [2.0]]), np.zeros((2, 1)))
    c = Trajectory.concatenate([a, b])
    assert c.duration == 3.0 and len(c.t) == 4 and c.q[-1, 0] == 2.0
    with pytest.raises(ValueError):
        Trajectory(np.array([0.0, 0.0]), np.zeros((2, 1)), np.zeros((2, 1)))


def test_linear_move_may_start_on_a_joint_limit(model, cfg):
    """The home pose (URDF zero) has joints 2 and 3 on their lower limit; a linear move away from it must plan."""
    home = np.zeros(6)
    prog = Program()
    prog.add(TaughtPoint(q=[0.0, 0.25, 0.35, 0.0, 0.0, 0.0], motion="linear", speed=0.5))
    traj = plan_program(model, prog, home, cfg, gripper_start=0.0)
    assert traj.duration > 0 and np.min(traj.q[:, 1]) >= -1e-9 and np.min(traj.q[:, 2]) >= -1e-9
    pa, pb = model.fk(home).translation, model.fk(np.array(prog.points[0].q)).translation
    for s in np.linspace(0, traj.duration, 30):
        pm = model.fk(traj.sample(s)[0]).translation
        assert np.linalg.norm(np.cross(pm - pa, pb - pa)) / np.linalg.norm(pb - pa) < 1e-4


def _shifted(model, q_seed, dxyz):
    """IK solution for the pose of q_seed translated by dxyz (base axes)."""
    T = model.fk(q_seed)
    res = model.ik(pin.SE3(T.rotation, T.translation + np.asarray(dxyz)), q_seed)
    assert res.success
    return np.asarray(res.q)


def test_linear_block_blends_the_corner(model, cfg, limits):
    vmax, amax = limits
    q1 = _shifted(model, L_POSE, [0.08, 0.0, 0.0])       # 8 cm forward
    q2 = _shifted(model, q1, [0.0, 0.0, 0.08])           # then 8 cm up: an L-shaped path
    r = 0.02
    res = plan_linear_block(model, L_POSE, [q1, q2], vmax, amax, 0.15, 0.3, 0.8, 2.0, 0.01,
                            speed=1.0, blend_radius=r)
    traj = res.trajectory
    P0, P1, P2 = (model.fk(q).translation for q in (L_POSE, q1, q2))
    pos = np.array([model.fk(traj.sample(s)[0]).translation for s in np.linspace(0, traj.duration, 200)])
    # The corner is cut: the path never visits P1 but comes within the blend radius of it.
    d_corner = np.min(np.linalg.norm(pos - P1, axis=1))
    assert 0.3 * r < d_corner < r
    # Outside the blend zone the path is on the two straight lines.
    for p in pos:
        on1 = np.linalg.norm(np.cross(p - P0, P1 - P0)) / np.linalg.norm(P1 - P0)
        on2 = np.linalg.norm(np.cross(p - P1, P2 - P1)) / np.linalg.norm(P2 - P1)
        if np.linalg.norm(p - P1) > 1.5 * r:
            assert min(on1, on2) < 2e-4
    # Velocity is continuous through the corner: the arm does not stop at the pass-through point.
    t1 = res.arrival_times[0]
    assert 0.0 < t1 < traj.duration
    lin = model.ee_velocity(*traj.sample(t1))[0]
    assert lin > 0.05
    assert np.max(np.abs(traj.q[-1] - q2)) < 1e-3
    assert np.all(traj.max_abs_velocity() <= vmax * 1.001)


def test_plan_program_stops_at_unblended_linear_points(model, cfg):
    q1 = _shifted(model, L_POSE, [0.08, 0.0, 0.0])
    q2 = _shifted(model, q1, [0.0, 0.0, 0.08])
    prog = Program()
    prog.add(TaughtPoint(q=q1, motion="linear", speed=1.0, blend=False))
    prog.add(TaughtPoint(q=q2, motion="linear", speed=1.0))
    traj = plan_program(model, prog, L_POSE, cfg, gripper_start=0.0)
    assert np.max(np.abs(traj.sample(traj.point_times[0])[1])) < 1e-2        # stopped at the corner
    prog.points[0].blend = True
    traj_b = plan_program(model, prog, L_POSE, cfg, gripper_start=0.0)
    assert np.max(np.abs(traj_b.sample(traj_b.point_times[0])[1])) > 0.05   # passed through
    assert traj_b.duration < traj.duration


def test_linear_pure_rotation_has_a_duration(model, limits):
    vmax, amax = limits
    q_rot = L_POSE + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    T0, T1 = model.fk(L_POSE), model.fk(q_rot)
    if np.linalg.norm(T1.translation - T0.translation) > 1e-6:
        pytest.skip("joint 6 does not rotate about the tool point in this model")
    tr = plan_linear_segment(model, L_POSE, q_rot, vmax, amax, 0.15, 0.3, 0.8, 2.0, 0.01)
    assert tr.duration > 0.5 and np.max(np.abs(tr.q[-1] - q_rot)) < 1e-3
    assert np.max(np.abs(tr.qd[:, 5])) <= 0.8 * 1.05
