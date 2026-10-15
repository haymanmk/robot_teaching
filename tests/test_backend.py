import pytest

from robot_teaching.backend.base import BackendError


def test_hardware_gains_are_read_from_the_yaml_each_time(tmp_path):
    pytest.importorskip("motorbridge")          # the vendor actuator package imports it at module level
    from robot_teaching.backend.rebotarm import mit_gains_from_yaml

    y = tmp_path / "hw.yaml"
    y.write_text("joints:\n"
                 "  - {name: j1, motor_id: 1, feedback_id: 253, MIT: {kp: 10, kd: 1}}\n"
                 "  - {name: j2, motor_id: 2, feedback_id: 253, MIT: {kp: 20, kd: 2}}\n")
    kp, kd = mit_gains_from_yaml(str(y), ["j2", "j1"])
    assert kp.tolist() == [20.0, 10.0] and kd.tolist() == [2.0, 1.0]
    y.write_text("joints:\n  - {name: j1, motor_id: 1, feedback_id: 253, MIT: {kp: 15, kd: 1.5}}\n")
    kp, kd = mit_gains_from_yaml(str(y), ["j1"])
    assert kp.tolist() == [15.0] and kd.tolist() == [1.5], "an edit is seen on the next call"
    with pytest.raises(BackendError):
        mit_gains_from_yaml(str(y), ["j2"])
