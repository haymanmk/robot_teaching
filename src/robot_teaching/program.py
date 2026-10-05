"""Taught points and programs, with versioned JSON persistence.

A :class:`TaughtPoint` stores everything needed to reproduce a pose on the arm
and how to get there from the previous point. The joint configuration ``q`` is
the source of truth; the cached ``pose`` is for display and is refreshed by the
server whenever ``q`` changes.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

PROGRAM_FORMAT_VERSION = 1
MotionType = Literal["joint", "linear"]
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\- ]{0,63}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TaughtPoint:
    q: list[float]
    gripper: float = 0.0
    name: str = ""
    motion: MotionType = "joint"
    speed: float = 0.5
    blend: bool = False
    dwell: float = 0.0
    pose: dict | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        self.q = [float(v) for v in self.q]
        self.gripper = float(self.gripper)
        self.speed = float(self.speed)
        self.dwell = float(self.dwell)
        if self.motion not in ("joint", "linear"):
            raise ValueError(f"motion must be 'joint' or 'linear', got {self.motion!r}")
        if not (0.0 < self.speed <= 1.0):
            raise ValueError(f"speed must be in (0, 1], got {self.speed}")
        if self.dwell < 0.0:
            raise ValueError("dwell must be >= 0")
        if self.motion == "linear" and self.blend:
            # Linear segments always stop at both ends in this version.
            self.blend = False

    def q_array(self) -> np.ndarray:
        return np.asarray(self.q, dtype=float)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "q": list(self.q),
            "gripper": self.gripper,
            "motion": self.motion,
            "speed": self.speed,
            "blend": self.blend,
            "dwell": self.dwell,
            "pose": self.pose,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaughtPoint":
        return cls(
            q=d["q"],
            gripper=d.get("gripper", 0.0),
            name=d.get("name", ""),
            motion=d.get("motion", "joint"),
            speed=d.get("speed", 0.5),
            blend=bool(d.get("blend", False)),
            dwell=d.get("dwell", 0.0),
            pose=d.get("pose"),
            id=d.get("id") or uuid.uuid4().hex[:8],
        )


@dataclass
class Program:
    name: str = "untitled"
    points: list[TaughtPoint] = field(default_factory=list)
    robot: str = "reBot Arm B601-RS"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    # ── editing ───────────────────────────────────────────────────────────

    def touch(self) -> None:
        self.updated_at = _now()

    def add(self, point: TaughtPoint, index: int | None = None) -> TaughtPoint:
        if not point.name:
            point.name = f"P{len(self.points) + 1}"
        if index is None or index >= len(self.points):
            self.points.append(point)
        else:
            self.points.insert(max(0, index), point)
        self.touch()
        return point

    def find(self, point_id: str) -> TaughtPoint:
        for p in self.points:
            if p.id == point_id:
                return p
        raise KeyError(point_id)

    def index_of(self, point_id: str) -> int:
        for i, p in enumerate(self.points):
            if p.id == point_id:
                return i
        raise KeyError(point_id)

    def remove(self, point_id: str) -> TaughtPoint:
        i = self.index_of(point_id)
        p = self.points.pop(i)
        self.touch()
        return p

    def reorder(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if sorted(ids) != sorted(p.id for p in self.points):
            raise ValueError("reorder must contain exactly the current point ids")
        by_id = {p.id: p for p in self.points}
        self.points = [by_id[i] for i in ids]
        self.touch()

    def move(self, point_id: str, new_index: int) -> None:
        i = self.index_of(point_id)
        p = self.points.pop(i)
        new_index = int(np.clip(new_index, 0, len(self.points)))
        self.points.insert(new_index, p)
        self.touch()

    # ── validation ────────────────────────────────────────────────────────

    def validate(self, lower: np.ndarray, upper: np.ndarray, margin: float = 0.0) -> list[str]:
        """Return human-readable problems (empty list means the program is valid)."""
        problems: list[str] = []
        n = len(lower)
        for i, p in enumerate(self.points):
            label = p.name or p.id
            if len(p.q) != n:
                problems.append(f"point {i + 1} ({label}): expected {n} joint values, got {len(p.q)}")
                continue
            q = p.q_array()
            low = q < lower + margin - 1e-9
            high = q > upper - margin + 1e-9
            for j in np.flatnonzero(low | high):
                problems.append(
                    f"point {i + 1} ({label}): joint {j + 1} = {q[j]:.3f} rad outside "
                    f"[{lower[j] + margin:.3f}, {upper[j] - margin:.3f}]"
                )
        return problems

    # ── persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "format_version": PROGRAM_FORMAT_VERSION,
            "name": self.name,
            "robot": self.robot,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Program":
        version = int(d.get("format_version", 1))
        if version > PROGRAM_FORMAT_VERSION:
            raise ValueError(f"program format {version} is newer than supported {PROGRAM_FORMAT_VERSION}")
        return cls(
            name=d.get("name", "untitled"),
            points=[TaughtPoint.from_dict(p) for p in d.get("points", [])],
            robot=d.get("robot", "reBot Arm B601-RS"),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Program":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Program":
        return cls.from_json(Path(path).read_text())


class ProgramStore:
    """Directory of ``<name>.json`` programs."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_name(name: str) -> str:
        name = name.strip()
        if not _NAME_RE.match(name):
            raise ValueError("program name: 1-64 letters, digits, space, '_' or '-', starting with a letter or digit")
        return name

    def path(self, name: str) -> Path:
        return self.dir / f"{self.check_name(name)}.json"

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                out.append({
                    "name": p.stem,
                    "points": len(d.get("points", [])),
                    "updated_at": d.get("updated_at"),
                })
            except (OSError, ValueError):
                continue
        return out

    def save(self, program: Program, name: str | None = None) -> Path:
        name = self.check_name(name or program.name)
        program.name = name
        return program.save(self.path(name))

    def load(self, name: str) -> Program:
        path = self.path(name)
        if not path.exists():
            raise FileNotFoundError(name)
        program = Program.load(path)
        program.name = name
        return program

    def delete(self, name: str) -> None:
        path = self.path(name)
        if not path.exists():
            raise FileNotFoundError(name)
        path.unlink()
