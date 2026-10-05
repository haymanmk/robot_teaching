"""Sampled, time-parameterized joint trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class PlanningError(RuntimeError):
    """Raised when a program cannot be turned into a valid trajectory."""


@dataclass
class TrajectoryEvent:
    time: float
    kind: str          # "point" (arrival at program point) | "gripper" (new gripper target)
    value: float | int


@dataclass
class Trajectory:
    t: np.ndarray                     # (N,) seconds, strictly increasing from 0
    q: np.ndarray                     # (N, n) rad
    qd: np.ndarray                    # (N, n) rad/s
    events: list[TrajectoryEvent] = field(default_factory=list)
    point_times: list[float] = field(default_factory=list)   # arrival time of each planned program point
    loop_start_time: float = 0.0      # where a looping playback restarts (arrival at the first point)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.q = np.asarray(self.q, dtype=float)
        self.qd = np.asarray(self.qd, dtype=float)
        if self.q.ndim != 2 or self.qd.shape != self.q.shape or len(self.t) != self.q.shape[0]:
            raise ValueError("inconsistent trajectory array shapes")
        if len(self.t) and (self.t[0] != 0.0 or np.any(np.diff(self.t) <= 0.0)):
            raise ValueError("trajectory time must start at 0 and be strictly increasing")

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def n_joints(self) -> int:
        return int(self.q.shape[1])

    def sample(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Linear interpolation of (q, qd) at time ``s``; clamps to the ends."""
        if s <= 0.0:
            return self.q[0].copy(), self.qd[0].copy()
        if s >= self.duration:
            return self.q[-1].copy(), self.qd[-1].copy()
        i = int(np.searchsorted(self.t, s, side="right")) - 1
        t0, t1 = self.t[i], self.t[i + 1]
        a = (s - t0) / (t1 - t0)
        return (1 - a) * self.q[i] + a * self.q[i + 1], (1 - a) * self.qd[i] + a * self.qd[i + 1]

    def max_abs_velocity(self) -> np.ndarray:
        return np.max(np.abs(self.qd), axis=0)

    def max_abs_acceleration(self) -> np.ndarray:
        if len(self.t) < 2:
            return np.zeros(self.n_joints)
        return np.max(np.abs(np.diff(self.qd, axis=0) / np.diff(self.t)[:, None]), axis=0)

    def summary(self) -> dict:
        return {
            "duration": self.duration,
            "samples": int(len(self.t)),
            "points": len(self.point_times),
            "point_times": [float(v) for v in self.point_times],
            "max_joint_velocity": [float(v) for v in self.max_abs_velocity()],
            "loop_start_time": float(self.loop_start_time),
        }

    @staticmethod
    def concatenate(pieces: list["Trajectory"]) -> "Trajectory":
        """Join trajectories end to start, shifting times and merging duplicate knots."""
        pieces = [p for p in pieces if len(p.t)]
        if not pieces:
            raise ValueError("nothing to concatenate")
        ts, qs, qds, events, point_times = [], [], [], [], []
        offset = 0.0
        for k, p in enumerate(pieces):
            t = p.t + offset
            if k > 0:
                # The piece starts where the previous one ended; drop the duplicate sample.
                t, q, qd = t[1:], p.q[1:], p.qd[1:]
            else:
                q, qd = p.q, p.qd
            ts.append(t)
            qs.append(q)
            qds.append(qd)
            events.extend(TrajectoryEvent(e.time + offset, e.kind, e.value) for e in p.events)
            point_times.extend(pt + offset for pt in p.point_times)
            offset += p.duration
        return Trajectory(np.concatenate(ts), np.vstack(qs), np.vstack(qds), events, point_times)
