#!/usr/bin/env python3
"""Offline validation of the trajectory planner.

Plans a saved program exactly the way the Play button does, then checks the
sampled trajectory and plots it, so the planner can be judged without the arm:

* timing      samples start at 0 and are strictly increasing;
* arrival     the trajectory passes through every taught joint configuration at
              the reported point time;
* consistency the stored joint velocity equals the finite difference of the
              stored positions;
* continuity  no jump in position or velocity between samples (a jump is a step
              that would need more than 1.5x the velocity or acceleration limit);
* limits      per-segment peak velocity and acceleration against the configured
              limits scaled by the point speed;
* linear      tool position stays on the chord and the orientation on the
              geodesic during linear moves;
* stream      the 500 Hz command stream (the controller's linear interpolation of
              the plan) against a plan sampled directly at the control rate.

Usage:
    uv run --with matplotlib python tools/validate_plan.py programs/example_pick_place.json --out /tmp/plan
    uv run python tools/validate_plan.py programs/my_program.json        # checks only, no plots
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pinocchio as pin  # noqa: E402

import robot_teaching  # noqa: E402,F401  (puts the vendor checkout on sys.path)
from robot_teaching.config import load_config, resolve_vector  # noqa: E402
from robot_teaching.model import RobotModel  # noqa: E402
from robot_teaching.planning import PlanningError, plan_program  # noqa: E402
from robot_teaching.program import Program  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-4, text and surface tokens.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#f0efec"


def moving_average(x: np.ndarray, w: int = 5) -> np.ndarray:
    k = np.ones(w) / w
    return np.column_stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])])


def zero_crossings(x: np.ndarray, deadband: float) -> int:
    """Sign changes of ``x`` ignoring excursions smaller than ``deadband``."""
    s = np.sign(np.where(np.abs(x) < deadband, 0.0, x))
    s = s[s != 0]
    return int(np.sum(s[1:] * s[:-1] < 0)) if len(s) > 1 else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program", help="program JSON file")
    ap.add_argument("--start", default=None, help="start configuration, comma separated (rad); default: first point")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--config", default=None, help="path to teaching.yaml")
    ap.add_argument("--out", default=None, help="prefix for the PNG plots (needs matplotlib); omit for checks only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = RobotModel()
    program = Program.load(args.program)
    n = model.n
    vmax = resolve_vector(cfg.limits.joint_velocity, n, "v")
    amax = resolve_vector(cfg.limits.joint_acceleration, n, "a")
    rate = float(cfg.control.rate)
    q_start = np.array([float(v) for v in args.start.split(",")]) if args.start else program.points[0].q_array()

    try:
        traj = plan_program(model, program, q_start, cfg, gripper_start=None, loop=args.loop)
    except PlanningError as e:
        print(f"planning failed: {e}")
        return 1

    points = list(program.points)
    if args.loop and len(points) > 1:
        points.append(dataclasses.replace(points[0], name=f"{points[0].name} (loop)", motion="joint"))
    t, q, qd = traj.t, traj.q, traj.qd
    dt_plan = float(cfg.limits.planning_dt)
    qd_fd = np.gradient(q, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)
    jerk = np.gradient(qdd, t, axis=0)
    bounds = [0.0] + list(traj.point_times)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))

    # ── timing ────────────────────────────────────────────────────────────
    steps = np.diff(t)
    check("timing", t[0] == 0.0 and np.all(steps > 0),
          f"{len(t)} samples over {traj.duration:.3f} s, spacing {steps.min()*1e3:.2f}-{steps.max()*1e3:.2f} ms "
          f"(planning_dt {dt_plan*1e3:.0f} ms; shorter steps close each piece)")

    # ── arrival at the taught configurations ──────────────────────────────
    arrival_err = [float(np.max(np.abs(traj.sample(pt)[0] - p.q_array()))) for pt, p in zip(traj.point_times, points)]
    check("arrival", max(arrival_err) < 1e-3,
          "max |q(arrival) - q_taught| = " + ", ".join(f"{e:.1e}" for e in arrival_err) + " rad")
    check("start", float(np.max(np.abs(q[0] - q_start))) < 1e-9, f"first sample at the start configuration, |dq| = {np.max(np.abs(q[0]-q_start)):.1e}")

    # ── velocity consistency ──────────────────────────────────────────────
    dv = np.abs(qd - qd_fd)
    check("consistency", float(np.max(dv)) < 0.05 * vmax.min(),
          f"max |qd - d(q)/dt| = {np.max(dv):.2e} rad/s, 99th percentile {np.percentile(dv, 99):.2e} rad/s "
          f"(limit for PASS: 5% of the slowest joint limit = {0.05*vmax.min():.3f})")

    # ── continuity: jumps between samples ─────────────────────────────────
    v_step = np.abs(np.diff(q, axis=0)) / steps[:, None]          # velocity implied by consecutive positions
    a_step = np.abs(np.diff(qd, axis=0)) / steps[:, None]         # acceleration implied by consecutive velocities
    pos_jumps = int(np.sum(np.any(v_step > 1.5 * vmax, axis=1)))
    vel_jumps = int(np.sum(np.any(a_step > 1.5 * amax, axis=1)))
    check("continuity", pos_jumps == 0 and vel_jumps == 0,
          f"{pos_jumps} position jumps, {vel_jumps} velocity jumps (steps needing >1.5x the limits); "
          f"largest implied velocity {np.max(v_step/vmax):.2f}x vmax, acceleration {np.max(a_step/amax):.2f}x amax")

    # ── per-segment limits, smoothness, straightness ──────────────────────
    qdd_s = moving_average(qdd, 5)
    rows = []
    straight_ok, straight_detail = True, []
    seg_masks = []
    for k, p in enumerate(points):
        a, b = bounds[k], bounds[k + 1]
        m = (t >= a - 1e-9) & (t <= b + 1e-9)
        seg_masks.append(m)
        if m.sum() < 3:
            rows.append((k, p, b - a, 0.0, 0.0, 0.0, 0, 0.0))
            continue
        vl, al = vmax * p.speed, amax * p.speed
        rv = float(np.max(np.abs(qd[m]) / vl))
        ra = float(np.max(np.abs(qdd[m]) / al))
        ras = float(np.max(np.abs(qdd_s[m]) / al))
        flips = max(zero_crossings(qdd_s[m][:, j], 0.05 * al[j]) for j in range(n))
        pj = float(np.percentile(np.abs(jerk[m]), 95))
        rows.append((k, p, b - a, rv, ra, ras, flips, pj))
        if p.motion == "linear":
            q_prev = points[k - 1].q_array() if k > 0 else q_start
            T0, T1 = model.fk(q_prev), model.fk(p.q_array())
            p0, p1 = T0.translation, T1.translation
            chord = p1 - p0
            L = float(np.linalg.norm(chord))
            w = pin.log3(T0.rotation.T @ T1.rotation)
            dev, ang = [], []
            for row in q[m]:
                T = model.fk(row)
                d = T.translation - p0
                s = float(np.clip(d @ chord / (L * L), 0.0, 1.0)) if L > 1e-9 else 0.0
                dev.append(np.linalg.norm(d - s * chord))
                R_geo = T0.rotation @ pin.exp3(w * s)
                ang.append(np.linalg.norm(pin.log3(R_geo.T @ T.rotation)))
            dev_mm, ang_deg = 1e3 * max(dev), np.degrees(max(ang))
            straight_ok &= dev_mm < 1.0 and ang_deg < 0.5
            straight_detail.append(f"{p.name}: {dev_mm:.3f} mm off the chord, {ang_deg:.3f} deg off the geodesic")
    lim_ok = all(r[3] <= 1.02 and r[5] <= 1.02 for r in rows)
    check("limits", lim_ok, "per-segment peaks are in the table below (PASS: velocity and smoothed acceleration <= 1.02x)")
    if straight_detail:
        check("linear", straight_ok, "; ".join(straight_detail))

    # ── the command stream: interpolation error bound and IK staircase ──────
    # The controller streams the plan at the control rate by linear interpolation. For a
    # smooth curve the interpolation error is at most max|qdd| * dt^2 / 8.
    interp_bound = float(np.max(np.abs(qdd_s))) * dt_plan ** 2 / 8.0
    # Linear moves re-solve IK at every sample and stop iterating below the IK tolerance,
    # which leaves small steps in q. Their size is the second difference of q.
    lin_mask = np.zeros(len(t), dtype=bool)
    for k, p in enumerate(points):
        if p.motion == "linear":
            lin_mask |= seg_masks[k]
    # The second difference of q minus the part explained by the (smoothed) acceleration is the
    # step left by the IK; the same residual on joint moves is the noise floor of the measure.
    d2 = np.diff(q, 2, axis=0)
    expected = qdd_s[1:-1] * (steps[1:, None] * steps[:-1, None])
    resid = np.abs(d2 - expected)
    stair = float(np.max(resid[lin_mask[1:-1]])) if lin_mask.any() else 0.0
    stair_joint = float(np.max(resid[~lin_mask[1:-1]])) if (~lin_mask).any() else 0.0
    mit_quantum = 8 * np.pi / 65535          # MIT frame position field: 16 bits over [-4pi, 4pi]
    check("stream", interp_bound < 1e-4 and stair < mit_quantum,
          f"linear interpolation of the {1/dt_plan:.0f} Hz plan at {rate:.0f} Hz is off by at most {interp_bound*1e6:.1f} urad; "
          f"IK steps left in the samples: up to {stair*1e6:.0f} urad on linear moves "
          f"({stair_joint*1e6:.0f} urad on joint moves, the floor of the measure); the MIT frame quantises position to {mit_quantum*1e6:.0f} urad")

    # ── robustness to the sample spacing: the same plan at the control rate ──
    cfg_fine = dataclasses.replace(cfg, limits=dataclasses.replace(cfg.limits, planning_dt=1.0 / rate))
    try:
        fine = plan_program(model, program, q_start, cfg_fine, gripper_start=None, loop=args.loop)
        dts = float(np.max(np.abs(np.array(fine.point_times) - np.array(traj.point_times))))
        check("dt-robust", dts < 0.05,
              f"planned again with planning_dt = 1/{rate:.0f} s: point times differ by at most {dts:.3f} s")
    except PlanningError as e:
        check("dt-robust", False,
              f"planned again with planning_dt = 1/{rate:.0f} s the planner refuses: '{e}'. The linear-move re-timing "
              f"differentiates the IK output twice, so the IK steps above look like acceleration (step/dt^2), and at a finer "
              f"spacing they exceed the limit")

    # ── report ────────────────────────────────────────────────────────────
    print(f"program {program.name!r}: {len(points)} points, planned duration {traj.duration:.3f} s, {len(t)} samples\n")
    for name, ok, det in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<12} {det}")
    print("\nper segment (ratios are relative to the limits scaled by the point speed):")
    print(f"{'#':>2} {'point':<14} {'motion':<7} {'speed':>5} {'T [s]':>6} {'vel':>6} {'acc':>6} {'acc smoothed':>12} {'acc flips':>9} {'jerk p95':>9}")
    for k, p, T, rv, ra, ras, flips, pj in rows:
        print(f"{k:>2} {p.name:<14} {p.motion:<7} {p.speed:>5.2f} {T:>6.2f} {rv:>6.3f} {ra:>6.3f} {ras:>12.3f} {flips:>9d} {pj:>9.2f}")
    print("\nacc flips = sign changes of the smoothed acceleration per joint (max over joints); a rest-to-rest move has 1.")

    if args.out is None:
        print("\nno --out given: checks only. Add --out PREFIX (with matplotlib available) for the plots.")
        return 0 if all(ok for _, ok, _ in results) else 2

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; run with 'uv run --with matplotlib ...' for the plots")
        return 0 if all(ok for _, ok, _ in results) else 2

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titlelocation": "left", "legend.frameon": False,
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def decorate(ax, names=False):
        for k, p in enumerate(points):
            if p.motion == "linear":
                ax.axvspan(bounds[k], bounds[k + 1], color=SHADE, lw=0, zorder=0)
            ax.axvline(bounds[k + 1], color=GRID, lw=0.8, zorder=1)
            if names:
                # Alternate two rows so neighbouring names do not collide.
                ax.text(bounds[k + 1], 1.02 + 0.12 * (k % 2), p.name, transform=ax.get_xaxis_transform(),
                        ha="right", va="bottom", fontsize=7, color=INK2)

    # Figure A: joint space, one row per joint (small multiples), columns q / qd / qdd / jerk.
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.0 * n), sharex=True)
    cols = [(q, "position [rad]"), (qd, "velocity [rad/s]"), (qdd, "acceleration [rad/s²]"), (jerk, "jerk [rad/s³]")]
    for j in range(n):
        for c, (arr, label) in enumerate(cols):
            ax = axes[j, c]
            decorate(ax, names=(j == 0))
            if c == 1:
                ax.plot(t, qd_fd[:, j], color=C2, lw=2.2, alpha=0.9, label="finite difference of q")
            if c == 2:
                ax.axhline(amax[j], color=C4, lw=0.8); ax.axhline(-amax[j], color=C4, lw=0.8)
            if c == 1:
                ax.axhline(vmax[j], color=C4, lw=0.8); ax.axhline(-vmax[j], color=C4, lw=0.8)
            ax.plot(t, arr[:, j], color=C1, lw=1.3, label="planned" if c == 1 else None)
            if j == 0:
                ax.set_title(label + ("  (yellow: limit at speed 1.0)" if c in (1, 2) else ""), pad=30)
            if c == 0:
                ax.set_ylabel(f"joint {j+1}", color=INK)
            if j == n - 1:
                ax.set_xlabel("time [s]")
    axes[0, 1].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Planned joint trajectory: {program.name}   (grey bands: linear moves; vertical lines: point arrivals)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6)
    fa = out.with_name(out.name + "_joints.png"); fig.savefig(fa, dpi=130); plt.close(fig)

    # Figure B: task space.
    P = np.array([model.fk(row).translation for row in q])
    ee_lin = np.array([model.ee_velocity(q[i], qd[i])[0] for i in range(len(t))])
    ee_ang = np.array([model.ee_velocity(q[i], qd[i])[1] for i in range(len(t))])
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    ax = axes[0, 0]; decorate(ax, names=True)
    for i, (lab, col) in enumerate(zip("xyz", (C1, C2, C3))):
        ax.plot(t, P[:, i] * 1e3, color=col, lw=1.6, label=lab)
        ax.text(t[-1], P[-1, i] * 1e3, f" {lab}", color=INK2, va="center", fontsize=8)
    ax.set_title("tool position [mm, base frame]", pad=30); ax.legend(loc="upper left", fontsize=8); ax.set_xlabel("time [s]")
    ax = axes[0, 1]; decorate(ax)
    ax.plot(t, ee_lin * 1e3, color=C1, lw=1.6)
    ax.axhline(cfg.limits.cartesian_linear_velocity * 1e3, color=C4, lw=0.8)
    ax.set_title("tool linear speed [mm/s]  (yellow: Cartesian limit at speed 1.0)"); ax.set_xlabel("time [s]")
    ax = axes[1, 0]; decorate(ax)
    ax.plot(t, np.degrees(ee_ang), color=C1, lw=1.6)
    ax.axhline(np.degrees(cfg.limits.cartesian_angular_velocity), color=C4, lw=0.8)
    ax.set_title("tool angular speed [deg/s]  (yellow: Cartesian limit at speed 1.0)"); ax.set_xlabel("time [s]")
    ax = axes[1, 1]
    lin_cols = [C1, C2, C3, C4]
    li = 0
    for k, p in enumerate(points):
        if p.motion != "linear":
            continue
        m = seg_masks[k]
        q_prev = points[k - 1].q_array() if k > 0 else q_start
        p0, p1 = model.fk(q_prev).translation, model.fk(p.q_array()).translation
        chord = p1 - p0; L = np.linalg.norm(chord)
        d = P[m] - p0
        s = np.clip(d @ chord / (L * L), 0, 1)
        dev = np.linalg.norm(d - s[:, None] * chord, axis=1) * 1e3
        col = lin_cols[li % 4]; li += 1
        ax.plot(s * 100, dev, color=col, lw=1.6, label=p.name)
    ax.set_title("linear moves: distance from the straight chord [mm] vs progress [%]"); ax.set_xlabel("progress along the chord [%]")
    ax.set_xlim(0, 100)
    if li:
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Planned tool motion: {program.name}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fb = out.with_name(out.name + "_tool.png"); fig.savefig(fb, dpi=130); plt.close(fig)

    # Figure C: what the controller streams at the control rate.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    ax = axes[0]
    lin_idx = [k for k, p in enumerate(points) if p.motion == "linear"]
    if lin_idx:
        k = lin_idx[0]; a = bounds[k]
        # the first 0.4 s of the first linear move, where the Cartesian step per sample is smallest
        m_plan = (t >= a - 1e-9) & (t <= a + 0.4)
        ts = np.arange(a, a + 0.4, 1.0 / rate)
        qs = np.array([traj.sample(s)[0] for s in ts])
        j = int(np.argmax(np.max(np.abs(qd[m_plan]), axis=0)))
        q_ref = q[m_plan][0, j]
        ax.plot(ts, (qs[:, j] - q_ref) * 1e3, color=C2, lw=1.4, label=f"{rate:.0f} Hz stream sent to the motor")
        ax.plot(t[m_plan], (q[m_plan][:, j] - q_ref) * 1e3, "o", color=C1, ms=4, label=f"{1/dt_plan:.0f} Hz planned samples")
        ax.set_title(f"start of the linear move '{points[k].name}', joint {j+1}: position change [mrad]")
        ax.set_xlabel("time [s]"); ax.legend(fontsize=8, loc="lower left")
    else:
        ax.set_title("no linear move in this program"); ax.axis("off")
    ax = axes[1]; decorate(ax, names=True)
    rel_step = np.max(a_step / amax, axis=1)
    ax.plot(t[1:], rel_step, color=C1, lw=1.0)
    ax.axhline(1.0, color=C4, lw=0.8)
    ax.set_title("step in velocity between samples, as acceleration / limit (yellow = 1.0)", pad=44)
    ax.set_xlabel("time [s]")
    fig.suptitle(f"Command stream: {program.name}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fc = out.with_name(out.name + "_stream.png"); fig.savefig(fc, dpi=130); plt.close(fig)
    print(f"\nplots: {fa}\n       {fb}\n       {fc}")
    return 0 if all(ok for _, ok, _ in results) else 2


if __name__ == "__main__":
    sys.exit(main())
