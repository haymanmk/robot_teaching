from robot_teaching.config import load_config, local_override_path


def test_local_override_is_merged_section_by_section(tmp_path):
    base = tmp_path / "teaching.yaml"
    base.write_text("hold:\n  kp: 100.0\n  kd: 5.0\njog:\n  joint_velocity: 0.7\nfriction:\n  coulomb: [1, 1, 1, 1, 1, 1]\n")
    cfg = load_config(base)
    assert cfg.local_path is None and cfg.hold.kp == 100.0 and cfg.hardware_yaml is None
    local = local_override_path(base)
    assert local == tmp_path / "teaching.local.yaml"
    local.write_text("hold:\n  kp: 90.0\nfriction:\n  coulomb: 0.2\nprograms_dir: elsewhere\n")
    cfg = load_config(base)
    assert cfg.local_path == local
    assert cfg.hold.kp == 90.0 and cfg.hold.kd == 5.0, "a local key replaces that key, the rest stays"
    assert cfg.jog.joint_velocity == 0.7, "untouched sections keep the base values"
    assert cfg.friction.coulomb == 0.2 and cfg.programs_dir == "elsewhere"
    assert load_config(base, use_local=False).hold.kp == 100.0
    other = tmp_path / "site.yaml"
    other.write_text("hold:\n  kp: 70.0\n")
    assert load_config(base, local=other).hold.kp == 70.0


def test_hardware_yaml_resolves_relative_to_the_config_file(tmp_path):
    base = tmp_path / "teaching.yaml"
    hw = tmp_path / "rebotarm_rs.local.yaml"
    hw.write_text("joints: []\n")
    base.write_text("hardware_yaml: rebotarm_rs.local.yaml\n")
    assert load_config(base).hardware_yaml == str(hw.resolve())
    base.write_text("hardware_yaml: rebotarm_rs.yaml\n")          # not next to the config: left for the vendor to resolve
    assert load_config(base).hardware_yaml == "rebotarm_rs.yaml"
    base.write_text("hardware_yaml: null\n")
    assert load_config(base).hardware_yaml is None
