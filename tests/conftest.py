import dataclasses

import numpy as np
import pytest

import robot_teaching  # noqa: F401  (puts the upstream checkout on sys.path)
from robot_teaching.config import load_config
from robot_teaching.model import RobotModel

L_POSE = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])   # elbow-up clearance pose


@pytest.fixture(scope="session")
def model():
    return RobotModel()


@pytest.fixture(scope="session")
def cfg():
    # The shipped configuration alone: a machine's teaching.local.yaml holds gains tuned for the
    # real arm (a gravity tau_scale, say), which the simulator does not share. The same goes for
    # its ripple tables (config/ripple.local.yaml): no file here, a test names its own.
    cfg = load_config(use_local=False)
    return dataclasses.replace(cfg, ripple=dataclasses.replace(cfg.ripple, file=""))


@pytest.fixture
def limits(cfg, model):
    from robot_teaching.config import resolve_vector
    n = model.n
    return (resolve_vector(cfg.limits.joint_velocity, n, "v"), resolve_vector(cfg.limits.joint_acceleration, n, "a"))
