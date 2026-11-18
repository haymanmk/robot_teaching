"""Position-indexed ripple feed-forward: the torque a joint needs that depends on its angle.

A geared motor does not turn smoothly: cogging and the gear mesh add a torque that repeats
with the joint angle (on J3 of this arm about 20 cycles per radian, 0.3 to 0.7 N·m). The
position loop only pushes back once the error has grown, and the disturbance observer answers
tens of milliseconds late, so at the ripple's frequency neither removes it. A torque read
from a table of the joint angle has no such delay.

The table holds, per joint and per direction of travel, the torque to *add* at each angle
(:class:`RippleTable`, a :class:`RippleMap` for the arm, stored as YAML). It is learned from
scope recordings (:func:`fit_table`): while a joint moves slowly, the position-locked part of
the torque its motor applied is what the feedback had to supply against the ripple, so that
part becomes the feed-forward. A recording taken with a table in use gives the next, better
table in the same way, because the motor torque then contains the table's own contribution
plus whatever the feedback still had to add.

Only what changes quickly with the angle goes into the table (faster than about one cycle per
``4 × smooth`` rad); the slow part (gravity error, friction) is left to the other terms.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

DIRECTIONS = ("forward", "reverse")      # the joint moving in the positive / the negative direction


@dataclass
class RippleTable:
    """The ripple torque of one joint against its angle, one curve per direction of travel.

    ``forward[k]`` and ``reverse[k]`` are the torque to add (N·m) at ``start + k · step`` rad;
    between two values it is interpolated, outside the table it is zero.
    """

    start: float
    step: float
    forward: np.ndarray
    reverse: np.ndarray

    def __post_init__(self) -> None:
        self.start, self.step = float(self.start), float(self.step)
        self.forward = np.asarray(self.forward, dtype=float).reshape(-1)
        self.reverse = np.asarray(self.reverse, dtype=float).reshape(-1)
        if not (self.step > 0.0 and np.isfinite(self.start)):
            raise ValueError("ripple table: step must be positive and start finite")
        if len(self.forward) != len(self.reverse) or len(self.forward) < 2:
            raise ValueError("ripple table: forward and reverse need the same number of values, two or more")
        if not (np.all(np.isfinite(self.forward)) and np.all(np.isfinite(self.reverse))):
            raise ValueError("ripple table: values must be finite")
        self.angles = self.start + self.step * np.arange(len(self.forward))

    @property
    def stop(self) -> float:
        """The angle of the last value, rad."""
        return float(self.angles[-1])

    def torque(self, q: float, direction: float = 0.0) -> float:
        """The torque to add at angle ``q``. ``direction`` runs from −1 (moving in the negative
        direction) to +1 and blends the two curves; 0, at rest, is their mean."""
        fwd = np.interp(q, self.angles, self.forward, left=0.0, right=0.0)
        rev = np.interp(q, self.angles, self.reverse, left=0.0, right=0.0)
        return float(0.5 * ((1.0 + direction) * fwd + (1.0 - direction) * rev))


class RippleMap:
    """The ripple tables of the arm: joint index → :class:`RippleTable`. A joint without a
    table gets no torque."""

    def __init__(self, n: int, tables: dict[int, RippleTable] | None = None) -> None:
        self.n = int(n)
        self.tables: dict[int, RippleTable] = {}
        for joint, table in (tables or {}).items():
            if not (0 <= int(joint) < self.n):
                raise ValueError(f"ripple map: no joint J{int(joint) + 1} on an arm of {self.n} joints")
            self.tables[int(joint)] = table

    def __bool__(self) -> bool:
        return bool(self.tables)

    def torque(self, q: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """The torque to add per joint, N·m; ``direction`` per joint as in :meth:`RippleTable.torque`."""
        out = np.zeros(self.n)
        for joint, table in self.tables.items():
            out[joint] = table.torque(float(q[joint]), float(direction[joint]))
        return out

    def describe(self) -> str:
        """One line for the log: which joints have a table and over which angles."""
        if not self.tables:
            return "no table"
        return ", ".join(f"J{j + 1} {t.start:.3f}..{t.stop:.3f} rad" for j, t in sorted(self.tables.items()))

    @classmethod
    def load(cls, path: str | Path | None, n: int) -> "RippleMap":
        """Read the YAML written by :meth:`save`. No path or no file gives an empty map."""
        if not path or not Path(path).exists():
            return cls(n)
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected one entry per joint (J1, J2, …)")
        tables = {}
        for key, entry in data.items():
            joint = _joint_index(key)
            try:
                tables[joint] = RippleTable(entry["start"], entry["step"], entry["forward"], entry["reverse"])
            except (KeyError, TypeError) as e:
                raise ValueError(f"{path}: {key} needs start, step, forward and reverse") from e
        return cls(n, tables)

    def save(self, path: str | Path) -> None:
        """Write the tables as YAML, one entry per joint (``J1`` is the first joint)."""
        data = {f"J{j + 1}": {"start": round(t.start, 6), "step": round(t.step, 6),
                              "forward": [round(float(v), 4) + 0.0 for v in t.forward],      # + 0.0: no "-0.0"
                              "reverse": [round(float(v), 4) + 0.0 for v in t.reverse]}
                for j, t in sorted(self.tables.items())}
        header = ("# Ripple feed-forward tables, written by `robot-teaching ripple-fit` (see README).\n"
                  "# Per joint: the torque to add (N·m) at start + k·step rad, for the joint moving in the\n"
                  "# positive (forward) and the negative (reverse) direction. Measured on one arm.\n")
        Path(path).write_text(header + yaml.safe_dump(data, default_flow_style=None, sort_keys=False, width=110))


def _joint_index(label) -> int:
    text = str(label).strip().upper()
    if text.startswith("J") and text[1:].isdigit() and int(text[1:]) >= 1:
        return int(text[1:]) - 1
    raise ValueError(f"'{label}' is not a joint label (J1, J2, …)")


# ── learning a table from recordings ───────────────────────────────────────────────────────

@dataclass
class Recording:
    """One joint of a scope recording: time, commanded and measured angle, and the torque the
    motor applied (its own report; the commanded torque where it reported none)."""

    joint: int
    t: np.ndarray
    q_cmd: np.ndarray
    q: np.ndarray
    tau: np.ndarray
    name: str = ""


def read_recording(path: str | Path) -> Recording:
    """Read a CSV written by the scope's Record button (any version of its columns)."""
    cols: dict[str, list[float]] = {k: [] for k in ("t_s", "q_cmd_rad", "q_rad", "tau_cmd_Nm", "tau_meas_Nm")}
    joint = None
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if joint is None:
                joint = _joint_index(row.get("joint", ""))
            for k in cols:
                text = row.get(k) or ""
                try:
                    cols[k].append(float(text))
                except ValueError:
                    cols[k].append(np.nan)
    if joint is None:
        raise ValueError(f"{path}: no samples")
    a = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    tau = np.where(np.isfinite(a["tau_meas_Nm"]), a["tau_meas_Nm"], a["tau_cmd_Nm"])
    return Recording(joint, a["t_s"], a["q_cmd_rad"], a["q_rad"], tau, name=Path(path).name)


@dataclass
class RippleFit:
    """A table and how well it describes the recordings it came from, per direction."""

    joint: int
    table: RippleTable
    rms: dict[str, float]            # N·m, RMS of the table over the angles it covers
    explained: dict[str, float]      # share of the motor torque's fast variation the table reproduces
    samples: dict[str, int]          # samples used
    speed: float                     # rad/s, the fastest cruise speed among the recordings


def _trend(x: np.ndarray, y: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """``y`` smoothed along ``x``: a straight line fitted around each point with Gaussian
    weights of width ``sigma`` (times ``w``). A slope does not bias it, at the ends either."""
    d = x[None, :] - x[:, None]
    k = w[None, :] * np.exp(-0.5 * (d / sigma) ** 2)
    s0, s1, s2 = k.sum(1), (k * d).sum(1), (k * d * d).sum(1)
    t0, t1 = (k * y).sum(1), (k * d * y).sum(1)
    den = s0 * s2 - s1 * s1
    flat = den <= 1e-12 * np.maximum(s0 * s2, 1e-300)
    return np.where(flat, t0 / s0, (s2 * t0 - s1 * t1) / np.where(flat, 1.0, den))


def _fill_and_taper(values: np.ndarray, valid: np.ndarray, taper_bins: int, max_gap: int = 3) -> np.ndarray:
    """Interpolate across gaps of a few bins, set the rest of the unknown bins to zero and
    bring the curve to zero smoothly where it meets them (and at both ends)."""
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return np.zeros(len(values))
    out, known = np.where(valid, values, 0.0), valid.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [values[a], values[b]])
            known[a + 1:b] = True
    # Bins between each known bin and the nearest unknown one (beyond either end is unknown):
    # none for the outermost known bin, which gets zero, the full value from taper_bins on.
    edge = np.concatenate(([False], known, [False]))
    pos = np.arange(len(edge))
    unknown = pos[~edge]
    between = np.abs(pos[:, None] - unknown[None, :]).min(axis=1)[1:-1] - 1
    if taper_bins < 1:
        return out * known
    window = 0.5 * (1.0 - np.cos(np.pi * np.clip(between, 0, taper_bins) / taper_bins))
    return out * np.where(known, window, 0.0)


def fit_table(recordings: Iterable[Recording], step: float = 0.002, smooth: float = 0.05,
              speed_fraction: float = 0.5, taper: float = 0.01, min_count: int = 5) -> RippleFit:
    """Learn the ripple table of one joint from recordings of it moving back and forth.

    Per recording and direction, the motor torque of the samples moving faster than
    ``speed_fraction`` of the recording's top commanded speed is averaged in bins of ``step``
    rad of measured angle. A smooth trend along the angle (width ``smooth`` rad) is removed,
    which leaves the part that follows the angle quickly; the recordings are then averaged.
    Bins with fewer than ``min_count`` samples are unknown: the table is zero there and
    comes to zero over ``taper`` rad next to them.

    Record slowly (a quarter speed or less): the faster the joint moves, the more the
    feedback torque lags the ripple it answers and the less of it one pass captures.
    """
    recs = list(recordings)
    if not recs:
        raise ValueError("no recordings")
    joint = recs[0].joint
    if any(r.joint != joint for r in recs):
        raise ValueError("the recordings are of different joints: fit one joint at a time")
    step, smooth = float(step), float(smooth)
    if step <= 0.0 or smooth <= 0.0:
        raise ValueError("step and smooth must be positive")

    parts: list[list[tuple[np.ndarray, np.ndarray]]] = [[], []]
    top = 0.0
    for r in recs:
        t, q_cmd = np.asarray(r.t, float), np.asarray(r.q_cmd, float)
        ok = np.isfinite(t) & np.isfinite(q_cmd)
        v = np.zeros(len(t))
        if ok.sum() > 2:
            v[ok] = np.gradient(q_cmd[ok], t[ok])
        v = np.where(np.isfinite(v), v, 0.0)
        vmax = float(np.percentile(np.abs(v), 99)) if len(v) else 0.0
        if vmax < 1e-6:                                   # rad/s: this recording stands still
            continue
        top = max(top, vmax)
        for d, sign in enumerate((1.0, -1.0)):
            m = ok & (sign * v > speed_fraction * vmax) & np.isfinite(r.q) & np.isfinite(r.tau)
            if m.any():
                parts[d].append((np.asarray(r.q, float)[m], np.asarray(r.tau, float)[m]))
    if not (parts[0] or parts[1]):
        raise ValueError("the joint does not move in these recordings")

    every_q = np.concatenate([q for d in parts for q, _ in d])
    i0, i1 = int(np.rint(every_q.min() / step)), int(np.rint(every_q.max() / step))
    nb = i1 - i0 + 1
    angles = (i0 + np.arange(nb)) * step
    curves, valids = [], []
    rms, explained, samples = {}, {}, {}
    for d, name in enumerate(DIRECTIONS):
        num, den, resid = np.zeros(nb), np.zeros(nb), []
        for q, tau in parts[d]:
            idx = np.rint(q / step).astype(int) - i0
            count = np.bincount(idx, minlength=nb).astype(float)
            ok = count >= min_count
            if ok.sum() < 3:
                continue
            mean = np.bincount(idx, weights=tau, minlength=nb)[ok] / count[ok]
            trend = _trend(angles[ok], mean, count[ok], smooth)
            num[ok] += count[ok] * (mean - trend)
            den[ok] += count[ok]
            used = ok[idx]
            resid.append((idx[used], q[used], tau[used] - np.interp(q[used], angles[ok], trend)))
        valid = den > 0.0
        curve = _fill_and_taper(np.where(valid, num / np.where(valid, den, 1.0), 0.0), valid,
                                int(round(taper / step)))
        curves.append(curve)
        valids.append(valid)
        samples[name] = int(sum(len(r[0]) for r in resid))
        rms[name] = float(np.sqrt(np.mean(curve[valid] ** 2))) if valid.any() else 0.0
        if resid:
            fast = np.concatenate([r[2] for r in resid])
            left = fast - np.interp(np.concatenate([r[1] for r in resid]), angles, curve)
            explained[name] = float(1.0 - np.sum(left ** 2) / max(np.sum(fast ** 2), 1e-300))
        else:
            explained[name] = 0.0
    covered = np.flatnonzero(valids[0] | valids[1])
    if len(covered) < 2:
        raise ValueError("too few samples to build a table: record more strokes, or lower min_count")
    a, b = covered[0], covered[-1] + 1
    table = RippleTable(angles[a], step, curves[0][a:b], curves[1][a:b])
    return RippleFit(joint, table, rms, explained, samples, top)
