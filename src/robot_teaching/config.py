"""Application configuration loaded from ``config/teaching.yaml``.

Every section is a dataclass with defaults, so a missing key in the YAML falls
back to a safe value. Per-joint quantities accept a scalar or a list and are
resolved to arrays by :func:`resolve_vector` once the joint count is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "teaching.yaml"


def resolve_vector(value: float | Sequence[float] | np.ndarray, n: int, label: str) -> np.ndarray:
    """Expand a scalar or an ``n``-length sequence to a float array of length ``n``."""
    if isinstance(value, (int, float)):
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"{label} must be a scalar or {n} values, got {arr.size}")
    return arr


@dataclass
class ControlConfig:
    rate: float = 500.0
    state_stream_rate: float = 30.0


@dataclass
class LimitsConfig:
    joint_velocity: Any = 1.0
    joint_acceleration: Any = 2.0
    cartesian_linear_velocity: float = 0.15
    cartesian_linear_acceleration: float = 0.3
    cartesian_angular_velocity: float = 0.8
    cartesian_angular_acceleration: float = 2.0
    joint_position_margin: float = 0.02
    planning_dt: float = 0.01


@dataclass
class FreeDriveConfig:
    kp: float = 8.0
    kd: float = 1.0
    ki: float = 1.0
    integral_limit: float = 0.3
    tau_scale: Any = 1.0
    transition_duration: float = 0.5
    linear_release_threshold: float = 0.04
    angular_release_threshold: float = 0.08
    linear_lock_threshold: float = 0.02
    angular_lock_threshold: float = 0.04
    lock_settle_duration: float = 0.15
    velocity_filter_time_constant: float = 0.03
    max_valid_sample_dt: float = 0.1


@dataclass
class HoldConfig:
    kp: Any = None
    kd: Any = None


@dataclass
class JogConfig:
    joint_steps_deg: list[float] = field(default_factory=lambda: [0.5, 1.0, 5.0])
    cartesian_steps_m: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.02])
    cartesian_steps_deg: list[float] = field(default_factory=lambda: [0.5, 2.0, 5.0])
    joint_velocity: float = 0.5
    max_ik_joint_jump: float = 0.5


@dataclass
class GripperConfig:
    closed_position: float = 0.0
    open_position: float = 5.0
    settle_time: float = 0.6
    kp: Any = None
    kd: Any = None


@dataclass
class ParkConfig:
    pose: Any = 0.0        # rad, scalar or one value per joint; the URDF zero is the extended rest pose
    speed: float = 0.3


@dataclass
class PlaybackConfig:
    default_speed: float = 0.5
    dwell_default: float = 0.0


@dataclass
class TeachingConfig:
    control: ControlConfig = field(default_factory=ControlConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    free_drive: FreeDriveConfig = field(default_factory=FreeDriveConfig)
    hold: HoldConfig = field(default_factory=HoldConfig)
    jog: JogConfig = field(default_factory=JogConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    park: ParkConfig = field(default_factory=ParkConfig)
    programs_dir: str = "programs"
    source_path: Path | None = None

    def programs_path(self) -> Path:
        p = Path(self.programs_dir)
        if not p.is_absolute():
            base = self.source_path.parent.parent if self.source_path else PROJECT_ROOT
            p = base / p
        return p


def _build(cls, data: dict | None):
    """Instantiate dataclass ``cls`` from a mapping, ignoring unknown keys."""
    data = data or {}
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            value = _build(f.type, value)
        kwargs[f.name] = value
    return cls(**kwargs)


_SECTIONS = {
    "control": ControlConfig,
    "limits": LimitsConfig,
    "free_drive": FreeDriveConfig,
    "hold": HoldConfig,
    "jog": JogConfig,
    "gripper": GripperConfig,
    "playback": PlaybackConfig,
    "park": ParkConfig,
}


def load_config(path: str | Path | None = None) -> TeachingConfig:
    """Load the YAML configuration; a missing file yields the defaults."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
    cfg = TeachingConfig()
    for key, cls in _SECTIONS.items():
        setattr(cfg, key, _build(cls, data.get(key)))
    if "programs_dir" in data:
        cfg.programs_dir = str(data["programs_dir"])
    cfg.source_path = cfg_path if cfg_path.exists() else None
    return cfg
