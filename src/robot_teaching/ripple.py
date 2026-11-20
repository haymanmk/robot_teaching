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

A table only knows the angles the joint crossed while it was recorded, and the arm cannot
always be swept over a joint's whole range. The ripple is close to one sinusoid of the joint
angle, though (on J2 and J3 about 20.05 cycles per rad, 126 per revolution, which reproduces
70 to 85 % of their tables), so the fit also finds that sinusoid (:class:`RippleSine`) and
the table continues as it wherever nothing was recorded: outside its range and in its gaps.
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
class RippleSine:
    """The ripple as one sinusoid of the joint angle, per direction of travel:
    ``a · sin(2π · cycles · q) + b · cos(2π · cycles · q)`` N·m, with ``(a, b)`` in
    ``forward`` and ``reverse`` and ``cycles`` in cycles per rad."""

    cycles: float
    forward: np.ndarray
    reverse: np.ndarray

    def __post_init__(self) -> None:
        self.cycles = float(self.cycles)
        self.forward = np.asarray(self.forward, dtype=float).reshape(-1)
        self.reverse = np.asarray(self.reverse, dtype=float).reshape(-1)
        if not (np.isfinite(self.cycles) and self.cycles > 0.0):
            raise ValueError("ripple sinusoid: cycles_per_rad must be positive")
        if len(self.forward) != 2 or len(self.reverse) != 2:
            raise ValueError("ripple sinusoid: forward and reverse are two values each, [sin, cos]")
        if not (np.all(np.isfinite(self.forward)) and np.all(np.isfinite(self.reverse))):
            raise ValueError("ripple sinusoid: values must be finite")

    @property
    def per_revolution(self) -> float:
        """Cycles per revolution of the joint."""
        return 2.0 * np.pi * self.cycles

    def amplitude(self, direction: str) -> float:
        """The amplitude of one direction's sinusoid, N·m."""
        return float(np.hypot(*getattr(self, direction)))

    def curves(self, q):
        """The forward and the reverse torque at angle(s) ``q``."""
        phase = 2.0 * np.pi * self.cycles * np.asarray(q, dtype=float)
        s, c = np.sin(phase), np.cos(phase)
        return self.forward[0] * s + self.forward[1] * c, self.reverse[0] * s + self.reverse[1] * c


@dataclass
class RippleTable:
    """The ripple torque of one joint against its angle, one curve per direction of travel.

    ``forward[k]`` and ``reverse[k]`` are the torque to add (N·m) at ``start + k · step`` rad;
    between two values it is interpolated. Outside the table the torque is that of ``sine``,
    and zero without one.
    """

    start: float
    step: float
    forward: np.ndarray
    reverse: np.ndarray
    sine: RippleSine | None = None

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
        if self.sine is not None and not (self.angles[0] <= q <= self.angles[-1]):
            fwd, rev = self.sine.curves(q)
        else:
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
        return ", ".join(f"J{j + 1} {t.start:.3f}..{t.stop:.3f} rad"
                         + (f" (beyond: sinusoid, {t.sine.cycles:.2f} cycles/rad)" if t.sine else "")
                         for j, t in sorted(self.tables.items()))

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
                sine = entry.get("sine")
                if sine is not None:
                    sine = RippleSine(sine["cycles_per_rad"], sine["forward"], sine["reverse"])
                tables[joint] = RippleTable(entry["start"], entry["step"], entry["forward"], entry["reverse"], sine)
            except (KeyError, TypeError, AttributeError) as e:
                raise ValueError(f"{path}: {key} needs start, step, forward and reverse "
                                 "(and its sine, if any, cycles_per_rad, forward and reverse)") from e
        return cls(n, tables)

    def save(self, path: str | Path) -> None:
        """Write the tables as YAML, one entry per joint (``J1`` is the first joint)."""
        def values(v):
            return [round(float(x), 4) + 0.0 for x in v]                                    # + 0.0: no "-0.0"

        data = {}
        for j, t in sorted(self.tables.items()):
            entry = {"start": round(t.start, 6), "step": round(t.step, 6),
                     "forward": values(t.forward), "reverse": values(t.reverse)}
            if t.sine is not None:
                entry["sine"] = {"cycles_per_rad": round(t.sine.cycles, 6),
                                 "forward": values(t.sine.forward), "reverse": values(t.sine.reverse)}
            data[f"J{j + 1}"] = entry
        header = ("# Ripple feed-forward tables, written by `robot-teaching ripple-fit` (see README).\n"
                  "# Per joint: the torque to add (N·m) at start + k·step rad, for the joint moving in the\n"
                  "# positive (forward) and the negative (reverse) direction. Measured on one arm.\n"
                  "# sine, where present, is the torque outside the table: per direction\n"
                  "# a·sin(2π·cycles_per_rad·q) + b·cos(2π·cycles_per_rad·q) with [a, b].\n")
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
    sine: RippleSine | None = None   # the best sinusoid found; the table uses it only if table.sine is set
    sine_share: float = 0.0          # share of the measured ripple (both directions) that sinusoid reproduces


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


def _fit_sine(angles: np.ndarray, values: list[np.ndarray], weights: list[np.ndarray],
              cycles: float | None, lowest: float, highest: float) -> tuple[RippleSine, float] | None:
    """The sinusoid of the angle closest to ``values`` (one curve per direction, ``weights``
    zero where a bin is unknown): one frequency for both directions, searched between
    ``lowest`` and ``highest`` cycles per rad unless ``cycles`` gives it, and an amplitude and
    phase per direction. Returns it with the share of the curves' weighted power it
    reproduces, or None where there is too little to fit one (fewer than three cycles)."""
    known = [w > 0.0 for w in weights]
    total = sum(float(np.sum(w[m] * y[m] ** 2)) for y, w, m in zip(values, weights, known))
    seen = angles[known[0] | known[1]]
    span = float(np.ptp(seen)) if len(seen) else 0.0
    if total <= 0.0 or span <= 0.0:
        return None

    def solve(f: float) -> tuple[list[np.ndarray], float]:
        phase = 2.0 * np.pi * f * angles
        basis = np.column_stack([np.sin(phase), np.cos(phase)])
        coefs, left = [], 0.0
        for y, w, m in zip(values, weights, known):
            if m.sum() < 3:
                coefs.append(np.zeros(2))
                continue
            root = np.sqrt(w[m])
            c = np.linalg.lstsq(basis[m] * root[:, None], y[m] * root, rcond=None)[0]
            coefs.append(c)
            left += float(np.sum(w[m] * (y[m] - basis[m] @ c) ** 2))
        return coefs, left

    if cycles is None:
        if highest <= lowest:
            return None
        fine = 0.1 / span                     # cycles per rad: a tenth of the width of a spectral line
        cycles = min(np.arange(lowest, highest, fine), key=lambda f: solve(f)[1])
        cycles = min(np.linspace(cycles - fine, cycles + fine, 81), key=lambda f: solve(f)[1])
    if cycles * span < 3.0:
        return None
    coefs, left = solve(float(cycles))
    return RippleSine(float(cycles), coefs[0], coefs[1]), 1.0 - left / total


def fit_table(recordings: Iterable[Recording], step: float = 0.002, smooth: float = 0.05,
              speed_fraction: float = 0.5, taper: float = 0.01, min_count: int = 5,
              sine: bool = True, sine_cycles: float | None = None, sine_share: float = 0.5) -> RippleFit:
    """Learn the ripple table of one joint from recordings of it moving back and forth.

    Per recording and direction, the motor torque of the samples moving faster than
    ``speed_fraction`` of the recording's top commanded speed is averaged in bins of ``step``
    rad of measured angle. A smooth trend along the angle (width ``smooth`` rad) is removed,
    which leaves the part that follows the angle quickly; the recordings are then averaged.
    Bins with fewer than ``min_count`` samples are unknown.

    With ``sine``, the sinusoid of the angle closest to what was measured is fitted as well
    (its frequency searched, or ``sine_cycles`` cycles per rad). If it reproduces at least
    ``sine_share`` of the measured ripple, the table takes its value in the unknown bins,
    passes into it over ``taper`` rad next to them, and continues as it outside its range.
    Otherwise, and without ``sine``, the table is zero there and comes to zero over ``taper``.

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
    means, counts, resids = [], [], []
    for d in range(2):
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
        means.append(np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0))
        counts.append(den)
        resids.append(resid)

    best, share = None, 0.0
    if sine:
        if sine_cycles is not None and not (np.isfinite(sine_cycles) and sine_cycles > 0.0):
            raise ValueError("sine_cycles must be positive")
        found = _fit_sine(angles, means, counts, sine_cycles, 1.0 / (4.0 * smooth), 1.0 / (8.0 * step))
        if found is not None:
            best, share = found
    used = best if best is not None and share >= sine_share else None
    base = used.curves(angles) if used is not None else (np.zeros(nb), np.zeros(nb))

    curves, valids = [], []
    rms, explained, samples = {}, {}, {}
    for d, name in enumerate(DIRECTIONS):
        valid, resid = counts[d] > 0.0, resids[d]
        # What the sinusoid leaves over goes to zero next to the unknown bins: the curve itself
        # then meets the sinusoid there (zero, without one).
        curve = base[d] + _fill_and_taper(np.where(valid, means[d] - base[d], 0.0), valid, int(round(taper / step)))
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
    table = RippleTable(angles[a], step, curves[0][a:b], curves[1][a:b], used)
    return RippleFit(joint, table, rms, explained, samples, top, best, float(share))
