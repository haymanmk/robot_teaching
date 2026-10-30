"""Tracking-error and ripple analysis of the scope recordings in data/.

Reads every data/scope_J3_*.csv, works out the playback speed and the feed-forward
combination from the file name, and writes to analysis/figures/:

  error_grid.png         mean ± std and RMS of the tracking error, feed-forward on x, one series per speed
  error_cycles_grid.png  the error of every motion cycle overlaid, one panel per condition
  error_breakdown.png    RMS of the dwell error, the stroke lag and the ripple, feed-forward x speed
  stroke_lag_explained.png  one 25 % cycle split into raw error, stroke lag, dwell error and ripple
  ripple_vs_angle.png    the ripple (error minus the per-stroke lag) against absolute joint angle
  ripple_spectrum.png    angle-domain (order) spectrum, ripple frequency vs. velocity, ripple RMS
  window_length.png      why a time-domain window smears the ripple and the angle domain does not
  summary.csv            the numbers behind the figures

When data/20261029/short/ exists (recordings after the cogging calibration and the gravity
re-scaling), it also writes the before/after comparison on the cells both sets share:

  after_error.png        mean / median / std and RMS, before beside after
  after_breakdown.png    dwell, stroke lag and ripple RMS, before beside after
  after_ripple.png       the ripple against joint angle and its order spectrum, before and after

Run with:  uv run --with scipy --with matplotlib python analysis/scope_analysis.py
"""
from __future__ import annotations

import csv
import warnings
import glob
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402
from scipy.signal import butter, find_peaks, sosfiltfilt, spectrogram, welch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
OUT = os.path.join(HERE, "figures")
AFTER = os.path.join(DATA, "20261029", "short")
BEFORE_COLOR, AFTER_COLOR = "#8a8984", "#2a78d6"

SPEEDS = [100, 50, 25]                      # rows of the grid, fastest first
FFS = ["g", "gf", "gfi"]                    # columns of the grid
FF_LABEL = {"g": "gravity", "gf": "gravity + friction", "gfi": "gravity + friction + inertia"}
SPEED_COLOR = {100: "#2a78d6", 50: "#eb6834", 25: "#1baf7a"}   # categorical slots 1-3
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 1.6,
})

DQ = 0.0005          # rad, angle-domain sample spacing (ripple period is ~0.05 rad)
V_FRACTION = 0.3     # a "stroke" is where |v_cmd| exceeds this fraction of its peak


# ----------------------------------------------------------------------------- loading

def load(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    hdr, rows = rows[0], rows[1:]
    col = {h: i for i, h in enumerate(hdr)}
    d = {h: np.array([float(r[col[h]]) for r in rows]) for h in hdr if h not in ("joint", "run")}
    d["name"] = os.path.basename(path)
    return d


def parse(name):
    """-> (speed %, feed-forward code, stroke label) from the file name."""
    speed = int(re.search(r"(\d+)%", name).group(1))
    tail = name.split("%", 1)[1].replace(".csv", "")
    ff = ""
    if "grav" in tail or re.search(r"_g", tail): ff += "g"
    if "fric" in tail or re.search(r"_g?f", tail): ff += "f"
    if "iner" in tail or re.search(r"_g?f?i", tail): ff += "i"
    stroke = "short" if "short" in name else ("long" if "long" in name else "medium")
    return speed, ff, stroke


def prepare(d):
    """Resample onto a uniform time base, add the filtered commanded velocity and the strokes."""
    t = d["t_s"] - d["t_s"][0]
    fs = 1 / np.median(np.diff(t))
    tu = np.arange(0, t[-1], 1 / fs)
    r = {k: np.interp(tu, t, d[k]) for k in ("err_rad", "q_rad", "q_cmd_rad")}
    r.update(t=tu, fs=fs, name=d["name"])
    r["v"] = sosfiltfilt(butter(2, 5, fs=fs, output="sos"), np.gradient(r["q_cmd_rad"], tu))
    vth = V_FRACTION * np.abs(r["v"]).max()
    moving = np.abs(r["v"]) > vth
    cuts = np.where(np.diff(moving.astype(int)) != 0)[0] + 1
    r["strokes"] = [s for s in np.split(np.arange(len(tu)), cuts) if moving[s[0]] and len(s) > 50]
    r["v_max"] = np.abs(r["v"]).max()
    return r


def stroke_residuals(r, qgrid):
    """Ripple of every stroke on a common angle grid: the error minus a cubic fit (the lag).

    Returns (forward list, backward list) of arrays over qgrid (NaN outside the stroke)."""
    fwd, bwd = [], []
    for s in r["strokes"]:
        q, e = r["q_rad"][s], r["err_rad"][s]
        forward = q[-1] > q[0]
        if not forward:
            q, e = q[::-1], e[::-1]
        eu = np.interp(qgrid, q, e, left=np.nan, right=np.nan)
        ok = ~np.isnan(eu)
        res = eu - np.polyval(np.polyfit(qgrid[ok], eu[ok], 3), qgrid)
        (fwd if forward else bwd).append(res)
    return fwd, bwd


def order_spectrum(r):
    """Angle-domain Welch spectrum of the ripple, one window per stroke, averaged.

    Returns (orders in cycles/rad, power), plus the window length in rad."""
    spectra, lengths = [], []
    for s in r["strokes"]:
        q, e = r["q_cmd_rad"][s], r["err_rad"][s]
        if q[-1] < q[0]:
            q, e = q[::-1], e[::-1]
        qu = np.arange(q[0], q[-1], DQ)
        if len(qu) < 200:
            continue
        eu = np.interp(qu, q, e)
        eu -= np.polyval(np.polyfit(qu, eu, 3), qu)
        k, p = welch(eu, fs=1 / DQ, nperseg=len(eu), nfft=8192, window="hann")
        spectra.append(p); lengths.append(len(eu) * DQ)
    return k, np.mean(spectra, axis=0), float(np.mean(lengths))


def ripple_series(r):
    """The error with the per-stroke lag removed, back in the time domain (NaN between strokes)."""
    out = np.full(len(r["t"]), np.nan)
    for s in r["strokes"]:
        q, e = r["q_cmd_rad"][s], r["err_rad"][s]
        out[s] = e - np.polyval(np.polyfit(q, e, 3), q)
    return out


def cycles(r):
    """Split the record into motion cycles (forward stroke start to the next one) and resample
    each over normalised time 0..1."""
    starts = [s[0] for s in r["strokes"] if r["v"][s[0]] > 0]
    grid = np.linspace(0, 1, 600)
    cyc = []
    for a, b in zip(starts[:-1], starts[1:]):
        seg = r["err_rad"][a:b]
        cyc.append(np.interp(grid, np.linspace(0, 1, len(seg)), seg))
    return grid, np.array(cyc)


# ----------------------------------------------------------------------------- figures

def heat(ax, M, title, unit, fmt="%.2f"):
    A = np.abs(M)
    im = ax.imshow(A, cmap=BLUES, aspect="auto", vmin=0)
    ax.set_xticks(range(3), [FF_LABEL[f].replace(" + ", "\n+ ") for f in FFS])
    ax.set_yticks(range(3), [f"{s}%" for s in SPEEDS])
    ax.grid(False); ax.tick_params(length=0)
    for (i, j), val in np.ndenumerate(M):
        ax.text(j, i, fmt % val, ha="center", va="center", fontsize=10,
                color="#fcfcfb" if abs(val) > 0.6 * A.max() + 0.4 * A.min() else INK)
    ax.set_title(f"{title} ({unit})", loc="left")
    return im


def fig_error_grid(short):
    """Point-range chart: feed-forward on x, one dodged series per speed.

    Left: mean error with whiskers at +-1 std (the variation). Right: RMS error."""
    stats = {}
    for (speed, ff), r in short.items():
        e = r["err_rad"] * 1e3
        stats[speed, ff] = dict(rms=np.sqrt(np.mean(e ** 2)), std=e.std(), mean=e.mean(), median=np.median(e),
                                p99=np.percentile(np.abs(e), 99), max=np.abs(e).max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    from matplotlib.lines import Line2D
    offsets = {100: -0.22, 50: 0.0, 25: 0.22}
    x = np.arange(len(FFS))
    for speed in SPEEDS:
        c = SPEED_COLOR[speed]; xs = x + offsets[speed]
        m = np.array([stats[speed, f]["mean"] for f in FFS]); sd = np.array([stats[speed, f]["std"] for f in FFS])
        rms = np.array([stats[speed, f]["rms"] for f in FFS])
        med = np.array([stats[speed, f]["median"] for f in FFS])
        axes[0].errorbar(xs, m, yerr=sd, fmt="o", color=c, ms=6, capsize=7, lw=1.6, label=f"{speed}% speed")
        axes[0].plot(xs, med, "D", color=c, ms=7, markerfacecolor="#fcfcfb", markeredgewidth=1.6)
        axes[1].plot(xs, rms, "o", color=c, ms=7, label=f"{speed}% speed")
        for xi, v in zip(xs, rms):
            axes[1].text(xi, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    axes[0].axhline(0, color=MUTED, lw=0.6)
    axes[0].set_title("Mean (dot), median (diamond), whiskers at ±1 std", loc="left")
    axes[0].legend(handles=[Line2D([], [], marker="o", color=INK2, ls="", ms=6, label="mean"),
                            Line2D([], [], marker="D", color=INK2, ls="", ms=7, markerfacecolor="#fcfcfb",
                                   markeredgewidth=1.6, label="median")], loc="lower right")
    axes[1].set_title("RMS tracking error", loc="left")
    for ax in axes:
        ax.set_xticks(x, [FF_LABEL[f].replace(" + ", "\n+ ") for f in FFS])
        ax.set_xlim(-0.6, len(FFS) - 0.4); ax.set_xlabel("feed-forward terms"); ax.set_ylabel("error (mrad)")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, axis="y", which="minor", color=GRID, linestyle="--", linewidth=0.6)
        ax.tick_params(axis="y", which="minor", length=2)
    axes[1].set_ylim(0, 5.2)
    axes[1].legend(loc="lower right")
    fig.suptitle("J3 tracking error over feed-forward and playback speed (short stroke, 0.37 rad)",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "error_grid.png"), dpi=150); plt.close(fig)
    return stats


def fig_cycles(short):
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    for i, speed in enumerate(SPEEDS):
        for j, ff in enumerate(FFS):
            ax, r = axes[i, j], short[speed, ff]
            grid, cyc = cycles(r)
            c = SPEED_COLOR[speed]
            for row in cyc:
                ax.plot(grid, row * 1e3, color=c, lw=0.5, alpha=0.25)
            ax.plot(grid, cyc.mean(axis=0) * 1e3, color=c, lw=1.8)
            ax.axhline(0, color=MUTED, lw=0.6)
            e = r["err_rad"] * 1e3
            ax.text(0.02, 0.96, f"RMS {np.sqrt(np.mean(e**2)):.2f} mrad   {len(cyc)} cycles",
                    transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
            if i == 0: ax.set_title(FF_LABEL[ff], loc="left")
            if j == 0: ax.set_ylabel(f"{speed}% speed\nerror (mrad)")
            if i == 2: ax.set_xlabel("position in the motion cycle (0 = start of forward stroke)")
    fig.suptitle("Tracking error of every motion cycle, overlaid (thin: single cycles, bold: mean)",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "error_cycles_grid.png"), dpi=150); plt.close(fig)


def error_breakdown(r):
    """RMS (mrad) of the error at the dwells, of the stroke-scale lag while moving, and of the ripple."""
    e = r["err_rad"] * 1e3
    moving = np.zeros(len(e), bool)
    for seg in r["strokes"]:
        moving[seg] = True
    dwell = np.abs(r["v"]) < 0.02 * r["v_max"]
    rip = ripple_series(r) * 1e3
    lag = e - np.nan_to_num(rip)
    return dict(dwell=np.sqrt(np.mean(e[dwell] ** 2)), lag=np.sqrt(np.mean(lag[moving] ** 2)),
                ripple=np.sqrt(np.nanmean(rip[moving] ** 2)))


def fig_breakdown(short):
    parts = {key: error_breakdown(r) for key, r in short.items()}
    titles = [("dwell", "At the dwells (|v| < 2 % of peak)"), ("lag", "Stroke lag while moving (ripple removed)"),
              ("ripple", "Ripple while moving")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    offsets = {100: -0.22, 50: 0.0, 25: 0.22}
    x = np.arange(len(FFS))
    for ax, (key, title) in zip(axes, titles):
        for speed in SPEEDS:
            xs = x + offsets[speed]; vals = np.array([parts[speed, f][key] for f in FFS])
            ax.plot(xs, vals, "o", color=SPEED_COLOR[speed], ms=7, label=f"{speed}% speed")
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xticks(x, [FF_LABEL[f].replace(" + ", "\n+ ") for f in FFS])
        ax.set_xlim(-0.6, len(FFS) - 0.4); ax.set_xlabel("feed-forward terms")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, axis="y", which="minor", color=GRID, linestyle="--", linewidth=0.6)
        ax.tick_params(axis="y", which="minor", length=2)
    axes[0].set_ylabel("RMS error (mrad)"); axes[0].set_ylim(0, 6)
    axes[2].legend(loc="upper right")
    fig.suptitle("Where the tracking error comes from: RMS of each part, feed-forward x playback speed",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "error_breakdown.png"), dpi=150); plt.close(fig)
    return parts


def fig_stroke_lag(short):
    """One 25 % cycle, gravity-only beside gravity + friction: raw error with the lag drawn over it,
    then lag and dwell error alone, then the ripple that remains."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col", sharey="row")
    for col, ff in enumerate(("g", "gf")):
        r = short[25, ff]; t = r["t"]; e = r["err_rad"] * 1e3
        rip = ripple_series(r) * 1e3; lag = np.where(np.isnan(rip), np.nan, e - rip)
        starts = [s[0] for s in r["strokes"] if r["v"][s[0]] > 0]
        a, b = starts[1], starts[2]; sl = slice(a - 150, b + 150); t0 = t[a]
        dwell = np.abs(r["v"]) < 0.02 * r["v_max"]
        for ax in axes[:, col]:
            ax.fill_between(t[sl] - t0, -10, 10, where=dwell[sl], color="#e6e5e1", alpha=0.8, lw=0, label="dwell")
            ax.axhline(0, color=MUTED, lw=0.6); ax.set_xlim(t[a - 150] - t0, t[b + 150] - t0)
        ax = axes[0, col]; ax.plot(t[sl] - t0, e[sl], color=INK, lw=1, label="raw error")
        ax.plot(t[sl] - t0, lag[sl], color="#eb6834", lw=2.2, label="stroke lag (cubic fit per stroke)")
        ax.set_title(f"25 % speed, {FF_LABEL[ff]}", loc="left"); ax.set_ylim(-9.5, 6.5)
        ax = axes[1, col]; ax.plot(t[sl] - t0, lag[sl], color="#eb6834", lw=2.2, label="stroke lag")
        ax.plot(t[sl] - t0, np.where(dwell[sl], e[sl], np.nan), color="#6250d6", lw=2.2, label="dwell error")
        ax = axes[2, col]; ax.plot(t[sl] - t0, rip[sl], color="#2a78d6", lw=1.2, label="ripple = raw − lag")
        ax.set_ylim(-4.5, 4.5); ax.set_xlabel("time (s)")
    for ax in axes[:, 0]: ax.set_ylabel("error (mrad)")
    for ax in axes[:, 1]: ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("One motion cycle split into its three parts (grey bands: dwells)", x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "stroke_lag_explained.png"), dpi=150); plt.close(fig)


def fig_ripple_vs_angle(short):
    qgrid = np.arange(0.745, 1.10, DQ)
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    ripple_rms = {}
    for i, speed in enumerate(SPEEDS):
        for j, ff in enumerate(FFS):
            ax, r = axes[i, j], short[speed, ff]
            fwd, bwd = stroke_residuals(r, qgrid)
            both = np.array(fwd + bwd)
            ripple_rms[speed, ff] = np.sqrt(np.nanmean(both ** 2)) * 1e3
            for arr, color, lab in ((fwd, "#2a78d6", "forward"), (bwd, "#eb6834", "backward")):
                for row in arr:
                    ax.plot(qgrid, row * 1e3, color=color, lw=0.4, alpha=0.2)
                ax.plot(qgrid, np.nanmean(arr, axis=0) * 1e3, color=color, lw=1.6, label=f"{lab} ({len(arr)} strokes)")
            ax.text(0.02, 0.96, f"ripple RMS {ripple_rms[speed, ff]:.2f} mrad", transform=ax.transAxes,
                    va="top", fontsize=8.5, color=INK2)
            if i == 0: ax.set_title(FF_LABEL[ff], loc="left")
            if j == 0: ax.set_ylabel(f"{speed}% speed\nripple (mrad)")
            if i == 2: ax.set_xlabel("measured joint angle q (rad)")
            if i == 0 and j == 2: ax.legend(loc="lower right", fontsize=8)
    axes[0, 0].set_ylim(-4.5, 4.5)
    fig.suptitle("Ripple against absolute joint angle: error minus each stroke's cubic lag trend, every stroke overlaid",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "ripple_vs_angle.png"), dpi=150); plt.close(fig)
    return ripple_rms


def ripple_spectrogram(r, seconds):
    """Spectrogram of the ripple (zero between strokes), zero-padded to a fine frequency grid."""
    x = ripple_series(r); gaps = np.isnan(x); x = np.where(gaps, 0, x)
    fs = r["fs"]; nper = int(seconds * fs)
    F, T, S = spectrogram(x, fs=fs, nperseg=nper, noverlap=nper * 7 // 8, nfft=16384, window="hann")
    # columns whose window overlaps a dwell are not trusted
    gap_frac = np.convolve(gaps.astype(float), np.ones(nper) / nper, mode="same")
    clean = np.interp(T, r["t"], gap_frac) < 0.02
    return F, T, S, clean


def ridge_points(r, fmin=1.5):
    """(|v|, dominant ripple frequency) from a 0.5 s spectrogram, one point per clean column."""
    F, T, S, clean = ripple_spectrogram(r, 0.5)
    S = S.copy(); S[F < fmin] = 0
    pk = F[np.argmax(S, axis=0)]
    v = np.interp(T, r["t"], np.abs(r["v"]))
    keep = clean & (v > 0.5 * r["v_max"])
    return v[keep], pk[keep]


def fig_spectrum(short, ripple_rms):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    peak_order = {}
    for speed in SPEEDS:
        for n, ff in enumerate(FFS):
            k, p, L = order_spectrum(short[speed, ff])
            lo = k > 10
            peak_order[speed, ff] = k[lo][np.argmax(p[lo])]
            ax.semilogy(k, p * 1e6, color=SPEED_COLOR[speed], lw=1.4, alpha=1.0 if n == 0 else 0.45,
                        label=f"{speed}% speed" if n == 0 else None)
    ax.set_xlim(0, 60); ax.set_ylim(1e-4, 1e1)
    ax.set_xlabel("order (cycles per rad of joint travel)"); ax.set_ylabel("ripple power (mrad²/(cycle/rad))")
    ax.set_title("Angle-domain spectrum, one window per stroke (0.37 rad)", loc="left")
    ax.legend(loc="upper right")
    k0 = np.median(list(peak_order.values()))
    ax.axvline(k0, color=MUTED, lw=0.8, ls="--"); ax.text(k0 + 0.8, 4, f"{k0:.1f} cyc/rad", color=INK2)
    ax.axvline(2 * k0, color=MUTED, lw=0.8, ls=":")

    ax = axes[1]
    for speed in SPEEDS:
        for n, ff in enumerate(FFS):
            v, f = ridge_points(short[speed, ff])
            # a window whose strongest line sits far below the expected ripple frequency is
            # dominated by what is left of the stroke lag, not by the ripple
            lagres = f < 0.65 * k0 * v
            ax.scatter(v[~lagres], f[~lagres], s=7, color=SPEED_COLOR[speed], alpha=0.35, edgecolors="none",
                       label=f"{speed}% speed" if n == 0 else None)
            ax.scatter(v[lagres], f[lagres], s=18, marker="x", color=SPEED_COLOR[speed], alpha=0.5, lw=0.8)
    vv = np.linspace(0, 0.6, 2)
    ax.plot(vv, k0 * vv, color=INK, lw=1, ls="--", label=f"{k0:.1f} cycles/rad × velocity")
    ax.scatter([], [], marker="x", color=INK2, s=18, lw=0.8, label="stroke-lag residual, not ripple")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 25)
    ax.set_xlabel("|commanded velocity| (rad/s)"); ax.set_ylabel("dominant ripple frequency (Hz)")
    ax.set_title("Ripple frequency follows the velocity (0.5 s windows)", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    M = np.array([[ripple_rms[s, f] for f in FFS] for s in SPEEDS])
    heat(ax, M, "Ripple RMS", "mrad")
    ax.set_xlabel("feed-forward terms"); ax.set_ylabel("playback speed")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ripple_spectrum.png"), dpi=150); plt.close(fig)
    return peak_order, k0


def fig_window(r, k0):
    """The 50 % gravity-only record: time-frequency view vs. windows of three lengths vs. the order spectrum."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = ripple_series(r); x = np.where(np.isnan(x), 0, x); fs = r["fs"]
    F, T, S, clean = ripple_spectrogram(r, 0.5)
    dB = 10 * np.log10(S * 1e6 + 1e-12)
    ax = axes[0]
    ax.pcolormesh(T, F, dB, cmap=BLUES, shading="auto", vmin=np.nanpercentile(dB, 70), vmax=np.nanpercentile(dB, 99.7))
    ax.plot(r["t"], k0 * np.abs(r["v"]), color="#eb6834", lw=1.0, label=f"{k0:.1f} cyc/rad × |v|")
    ax.set_xlim(0, 20); ax.set_ylim(0, 25); ax.grid(False)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Ripple spectrogram (0.5 s windows): the tone sweeps with |v|", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    for n, (sec, alpha) in enumerate(((0.25, 1.0), (1.0, 0.7), (4.0, 0.45))):
        f, p = welch(x, fs=fs, nperseg=int(sec * fs), window="hann")
        ax.semilogy(f, p * 1e6, color="#2a78d6", alpha=alpha, lw=1.4, label=f"{sec:g} s window (Δf = {1/sec:g} Hz)")
    ax.axvline(k0 * r["v_max"], color=MUTED, lw=0.8, ls="--")
    ax.text(k0 * r["v_max"] + 0.4, 1e-1, f"{k0:.1f} cyc/rad × v_max", color=INK2, fontsize=8)
    ax.set_xlim(0, 25); ax.set_ylim(1e-4, 1e0)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("ripple power (mrad²/Hz)")
    ax.set_title("Time-domain spectrum: smeared up to the top speed", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    k, p, L = order_spectrum(r)
    ax.semilogy(k, p * 1e6, color="#2a78d6", lw=1.4, label=f"one window per stroke ({L:.2f} rad, Δk = {1/L:.1f} cyc/rad)")
    ax.axvline(k0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, 60); ax.set_ylim(1e-4, 1e1)
    ax.set_xlabel("order (cycles per rad of joint travel)"); ax.set_ylabel("ripple power (mrad²/(cycle/rad))")
    ax.set_title("Angle domain: the same data, one sharp line", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Choosing the analysis window, {r['name']}", x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "window_length.png"), dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- before / after

def cond_label(speed, ff):
    terms = FF_LABEL[ff].replace(" + ", " +\n")
    return f"{speed}%\n" + terms


def fig_after_error(before, after, cells):
    x = np.arange(len(cells))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for off, sets, color, lab in ((-0.16, before, BEFORE_COLOR, "before (2026-10-20)"), (0.16, after, AFTER_COLOR, "after (2026-10-29)")):
        e = [sets[c]["err_rad"] * 1e3 for c in cells]
        m = np.array([v.mean() for v in e]); sd = np.array([v.std() for v in e])
        med = np.array([np.median(v) for v in e]); rms = np.array([np.sqrt(np.mean(v ** 2)) for v in e])
        axes[0].errorbar(x + off, m, yerr=sd, fmt="o", color=color, ms=6, capsize=7, lw=1.6, label=lab)
        axes[0].plot(x + off, med, "D", color=color, ms=7, markerfacecolor="#fcfcfb", markeredgewidth=1.6)
        axes[1].plot(x + off, rms, "o", color=color, ms=7, label=lab)
        for xi, v in zip(x + off, rms):
            axes[1].text(xi, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    axes[0].axhline(0, color=MUTED, lw=0.6)
    axes[0].set_title("Mean (dot), median (diamond), whiskers at ±1 std", loc="left")
    axes[1].set_title("RMS tracking error", loc="left")
    for ax in axes:
        ax.set_xticks(x, [cond_label(*c) for c in cells]); ax.set_xlim(-0.6, len(cells) - 0.4)
        ax.set_ylabel("error (mrad)")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, axis="y", which="minor", color=GRID, linestyle="--", linewidth=0.6)
        ax.tick_params(axis="y", which="minor", length=2)
    axes[1].set_ylim(0, 5.2); axes[1].legend(loc="upper right")
    fig.suptitle("J3 tracking error before and after the cogging calibration and gravity re-scaling (short stroke)",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "after_error.png"), dpi=150); plt.close(fig)


def fig_after_breakdown(before, after, cells):
    x = np.arange(len(cells))
    titles = [("dwell", "At the dwells"), ("lag", "Stroke lag while moving"), ("ripple", "Ripple while moving")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (key, title) in zip(axes, titles):
        for off, sets, color, lab in ((-0.16, before, BEFORE_COLOR, "before"), (0.16, after, AFTER_COLOR, "after")):
            vals = np.array([error_breakdown(sets[c])[key] for c in cells])
            ax.plot(x + off, vals, "o", color=color, ms=7, label=lab)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xticks(x, [cond_label(*c) for c in cells]); ax.set_xlim(-0.6, len(cells) - 0.4)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, axis="y", which="minor", color=GRID, linestyle="--", linewidth=0.6)
        ax.tick_params(axis="y", which="minor", length=2)
    axes[0].set_ylabel("RMS error (mrad)"); axes[0].set_ylim(0, 6); axes[2].legend(loc="upper right")
    fig.suptitle("Where the error comes from, before and after", x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "after_breakdown.png"), dpi=150); plt.close(fig)


def fig_after_ripple(before, after, cells):
    qgrid = np.arange(0.745, 1.10, DQ)
    show = [c for c in cells if c[1] == "gf"]
    fig, axes = plt.subplots(2, len(show), figsize=(7 * len(show), 8.5))
    for j, c in enumerate(show):
        ax = axes[0, j]
        for sets, color, lab in ((before, BEFORE_COLOR, "before"), (after, AFTER_COLOR, "after")):
            fwd, bwd = stroke_residuals(sets[c], qgrid)
            rms = np.sqrt(np.nanmean(np.array(fwd + bwd) ** 2)) * 1e3
            for row in fwd:
                ax.plot(qgrid, row * 1e3, color=color, lw=0.4, alpha=0.2)
            ax.plot(qgrid, np.nanmean(fwd, axis=0) * 1e3, color=color, lw=1.6, label=f"{lab}, forward strokes (ripple RMS {rms:.2f} mrad)")
        ax.set_title(f"{c[0]}% speed, {FF_LABEL[c[1]]}: ripple against joint angle", loc="left")
        ax.set_xlabel("measured joint angle q (rad)"); ax.set_ylabel("ripple (mrad)"); ax.set_ylim(-4.5, 4.5)
        ax.legend(loc="lower left", fontsize=8)
        ax = axes[1, j]
        for sets, color, lab in ((before, BEFORE_COLOR, "before"), (after, AFTER_COLOR, "after")):
            k, p, L = order_spectrum(sets[c])
            ax.semilogy(k, p * 1e6, color=color, lw=1.4, label=lab)
        ax.set_xlim(0, 60); ax.set_ylim(1e-4, 1e1)
        ax.set_xlabel("order (cycles per rad of joint travel)"); ax.set_ylabel("ripple power (mrad²/(cycle/rad))")
        ax.set_title("Angle-domain spectrum", loc="left"); ax.legend(loc="upper right")
    fig.suptitle("The ripple before and after the cogging calibration", x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "after_ripple.png"), dpi=150); plt.close(fig)


def compare_after(short, recs):
    if not os.path.isdir(AFTER):
        return
    after = {}
    for path in sorted(glob.glob(os.path.join(AFTER, "scope_J3_*.csv"))):
        speed, ff, _ = parse(os.path.basename(path))
        r = prepare(load(path)); after[speed, ff] = r
        recs[path] = (speed, ff, "short (after)", r)
    cells = [c for c in [(100, "g"), (100, "gf"), (25, "g"), (25, "gf")] if c in after and c in short]
    if not cells:
        return
    fig_after_error(short, after, cells)
    fig_after_breakdown(short, after, cells)
    fig_after_ripple(short, after, cells)


# ----------------------------------------------------------------------------- main

def main():
    warnings.filterwarnings("ignore", "Mean of empty slice")
    os.makedirs(OUT, exist_ok=True)
    recs = {}
    for path in sorted(glob.glob(os.path.join(DATA, "scope_J3_*.csv"))):
        speed, ff, stroke = parse(os.path.basename(path))
        recs[path] = (speed, ff, stroke, prepare(load(path)))
    short = {(s, f): r for s, f, st, r in recs.values() if st == "short"}
    missing = [(s, f) for s in SPEEDS for f in FFS if (s, f) not in short]
    if missing:
        raise SystemExit(f"short-stroke grid incomplete, missing {missing}")

    stats = fig_error_grid(short)
    fig_cycles(short)
    fig_breakdown(short)
    fig_stroke_lag(short)
    ripple_rms = fig_ripple_vs_angle(short)
    peak_order, k0 = fig_spectrum(short, ripple_rms)
    fig_window(short[50, "g"], k0)
    compare_after(short, recs)

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "stroke", "speed_%", "feedforward", "stroke_rad", "v_max_rad_s", "strokes",
                    "err_rms_mrad", "err_mean_mrad", "err_std_mrad", "err_p99_mrad", "err_max_mrad",
                    "ripple_rms_mrad", "ripple_order_cyc_per_rad", "ripple_hz_at_vmax"])
        for path, (speed, ff, stroke, r) in recs.items():
            e = r["err_rad"] * 1e3
            k, p, L = order_spectrum(r); lo = k > 10; ko = k[lo][np.argmax(p[lo])]
            qgrid = np.arange(r["q_rad"].min() + 0.01, r["q_rad"].max() - 0.01, DQ)
            fwd, bwd = stroke_residuals(r, qgrid)
            rr = np.sqrt(np.nanmean(np.array(fwd + bwd) ** 2)) * 1e3
            w.writerow([r["name"], stroke, speed, ff or "g (default)", f"{np.ptp(r['q_cmd_rad']):.2f}",
                        f"{r['v_max']:.3f}", len(r["strokes"]),
                        f"{np.sqrt(np.mean(e**2)):.3f}", f"{e.mean():.3f}", f"{e.std():.3f}",
                        f"{np.percentile(np.abs(e), 99):.3f}", f"{np.abs(e).max():.3f}",
                        f"{rr:.3f}", f"{ko:.1f}", f"{ko * r['v_max']:.1f}"])
    print(open(os.path.join(OUT, "summary.csv")).read())


if __name__ == "__main__":
    main()
