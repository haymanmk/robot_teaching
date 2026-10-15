import subprocess

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


# ── SocketCAN bring-up (no privileges needed: sysfs and `ip` are faked) ───────────────────

def _sysfs(tmp_path, interface="can0", flags=None):
    root = tmp_path / "sys_class_net"
    if flags is not None:
        d = root / interface
        d.mkdir(parents=True)
        (d / "flags").write_text(flags + "\n")
    return root


def test_can_bring_up_runs_ip_link_when_down(tmp_path):
    from robot_teaching.backend.socketcan import ensure_can_up

    sysfs = _sysfs(tmp_path, flags="0x1002")          # IFF_UP clear
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        (sysfs / "can0" / "flags").write_text("0x1003\n")   # the interface comes up
        return subprocess.CompletedProcess(cmd, 0, "", "")

    msg = ensure_can_up("can0", 1_000_000, sysfs=sysfs, run=fake_run, as_root=False)
    assert msg == "can0 brought up at 1000000 bit/s"
    assert calls == [["sudo", "-n", "ip", "link", "set", "can0", "up", "type", "can", "bitrate", "1000000"]]
    # As root the command runs directly.
    (sysfs / "can0" / "flags").write_text("0x1002\n")
    calls.clear()
    ensure_can_up("can0", 500_000, sysfs=sysfs, run=fake_run, as_root=True)
    assert calls == [["ip", "link", "set", "can0", "up", "type", "can", "bitrate", "500000"]]


def test_can_bring_up_leaves_an_up_interface_alone(tmp_path):
    from robot_teaching.backend.socketcan import ensure_can_up

    sysfs = _sysfs(tmp_path, flags="0x1003")

    def fail_run(cmd, **kw):
        raise AssertionError("must not run ip link on an interface that is up")

    assert ensure_can_up("can0", 1_000_000, sysfs=sysfs, run=fail_run, as_root=False) == "can0 is up"


def test_can_bring_up_errors_are_explained(tmp_path):
    from robot_teaching.backend.socketcan import ensure_can_up, is_socketcan_channel

    with pytest.raises(BackendError, match="no such network interface"):
        ensure_can_up("can0", 1_000_000, sysfs=_sysfs(tmp_path), run=None, as_root=False)
    sysfs = _sysfs(tmp_path / "b", flags="0x1002")

    def denied(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "sudo: a password is required")

    with pytest.raises(BackendError) as e:
        ensure_can_up("can0", 1_000_000, sysfs=sysfs, run=denied, as_root=False)
    assert "a password is required" in str(e.value) and "sudoers.d/robot-teaching-can" in str(e.value)
    with pytest.raises(BackendError) as e:
        ensure_can_up("can0", 1_000_000, sysfs=sysfs, run=denied, as_root=True)
    assert "sudoers" not in str(e.value), "no sudo hint when already root"
    assert is_socketcan_channel("can0") and not is_socketcan_channel("/dev/ttyACM0")
