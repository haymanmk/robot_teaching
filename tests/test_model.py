import numpy as np
import pinocchio as pin
import pytest

from robot_teaching.model import Pose

from conftest import L_POSE


def test_relative_pose_is_zero_at_reference(model):
    T = model.fk(L_POSE)
    rel = Pose.relative(T, T)
    assert np.allclose(rel.xyz, 0.0) and np.allclose(rel.rpy, 0.0, atol=1e-12)


def test_relative_pose_offsets_along_base_axes(model):
    T_home = model.fk(np.zeros(6))
    # Shift the home frame 50 mm up in the base frame and yaw it 10 deg about the base vertical.
    Rz = pin.rpy.rpyToMatrix(0.0, 0.0, np.deg2rad(10.0))
    T = pin.SE3(Rz @ T_home.rotation, T_home.translation + np.array([0.0, 0.0, 0.05]))
    rel = Pose.relative(T_home, T)
    assert np.allclose(rel.xyz, [0.0, 0.0, 0.05])
    assert rel.rpy[2] == pytest.approx(np.deg2rad(10.0)) and abs(rel.rpy[0]) < 1e-9 and abs(rel.rpy[1]) < 1e-9
    # The reading does not depend on how the tool is oriented at home: a tilted home frame gives the same offsets.
    T_home_tilted = pin.SE3(pin.rpy.rpyToMatrix(np.pi, 0.3, -0.7) @ T_home.rotation, T_home.translation)
    T2 = pin.SE3(Rz @ T_home_tilted.rotation, T_home_tilted.translation + np.array([0.0, 0.0, 0.05]))
    rel2 = Pose.relative(T_home_tilted, T2)
    assert np.allclose(rel2.xyz, [0.0, 0.0, 0.05]) and rel2.rpy[2] == pytest.approx(np.deg2rad(10.0))
