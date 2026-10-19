/* Scope drawing shared by the teach pendant (app.js, live) and the offline plotter (plot.js, a CSV
   recording): time strips with a min–max column per pixel when there are more samples than pixels,
   and the spectrum strip. The axis helpers have no DOM access, so Node can check them
   (tests/record_check.js); in the browser the module lives on window.scopeDraw. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api; else root.scopeDraw = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const PLOT_LEFT = 52, PLOT_RIGHT = 8;                           // px, the value labels / the right margin
  const GAP = 0.25;                                               // s without samples: lift the pen instead of bridging it

  function niceStep(span, want) {
    const raw = span / want, p = Math.pow(10, Math.floor(Math.log10(raw)));
    return [1, 2, 5, 10].map((m) => m * p).find((s) => span / s <= want) || 10 * p;
  }
  function niceTicks(lo, hi, want) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const step = niceStep(span, want), out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    return out;
  }
  const fmtTick = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2));
  /** Index of the first element of the ascending array a that is not below x. */
  const lowerBound = (a, x) => { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; };
  /** Index of the first element of the ascending array a that is above x. */
  const upperBound = (a, x) => { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= x) lo = m + 1; else hi = m; } return lo; };
  const missing = (v) => v === null || v === undefined || !Number.isFinite(v);

  // Size the canvas to its CSS box at the device pixel ratio, clear it, and return the drawing
  // context with the theme colours (null while the canvas has no size, e.g. hidden).
  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return null;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    ctx.font = "11px system-ui, sans-serif";
    return { ctx, W, H, lineColor: css.getPropertyValue("--line").trim() || "#262b36", muted: css.getPropertyValue("--muted").trim() || "#8b93a7" };
  }

  /**
   * One strip of time series. `axis` is { tEnd, W, t, absolute }: the time at the right edge, the
   * window (s), the times of the samples on screen (every series' `values` runs parallel to `t`)
   * and whether to label the clock itself (otherwise seconds before the right edge). `series` is
   * [{ values, color }]; a null or NaN value, or a step in `t` above `opts.gap` (default 0.25 s),
   * lifts the pen. A series may bring its own times (`t`, parallel to its values, plus `tOffset`
   * added to them: another recording overlaid on this axis) and its own look (`dash`, `alpha`,
   * `width`). `opts`: symmetric (centre the scale on zero), minSpan (least value range), bare (no
   * axes or grid, for an overview), cursor (a time: draw a vertical marker there).
   */
  function drawStrip(canvas, axis, series, opts) {
    const c = setupCanvas(canvas);
    if (!c) return;
    opts = opts || {};
    const { ctx, W: Wpx, H, lineColor, muted } = c;
    const { tEnd, W, t } = axis, gap = opts.gap === undefined ? GAP : opts.gap;
    const x0 = opts.bare ? 0 : PLOT_LEFT, x1 = Wpx - (opts.bare ? 0 : PLOT_RIGHT), y0 = opts.bare ? 2 : 6, y1 = H - (opts.bare ? 2 : 16);
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.values) if (!missing(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
    const minSpan = opts.minSpan || 1e-6;
    if (opts.symmetric) { const mx = Math.max(Math.abs(lo), Math.abs(hi), minSpan / 2); lo = -mx; hi = mx; }
    if (hi - lo < minSpan) { const c = (hi + lo) / 2; lo = c - minSpan / 2; hi = c + minSpan / 2; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const X = (tt) => x1 - (x1 - x0) * (tEnd - tt) / W;
    const Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
    if (!opts.bare) {
      ctx.lineWidth = 1; ctx.strokeStyle = lineColor; ctx.fillStyle = muted;
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (const v of niceTicks(lo, hi, 4)) {
        const y = Y(v);
        ctx.globalAlpha = v === 0 ? 1 : 0.6; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.fillText(fmtTick(v), x0 - 5, y);
      }
      // Time grid: live, seconds before the right edge; otherwise the clock (the t of a recording).
      const step = niceStep(W, 6);
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.globalAlpha = 0.6;
      const vline = (x) => { ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); };
      if (axis.absolute) {
        const dec = Math.max(0, -Math.floor(Math.log10(step)));
        for (const v of niceTicks(tEnd - W, tEnd, 6)) { const x = X(v); vline(x); if (x < x1 - 24) ctx.fillText(`${v.toFixed(dec)} s`, x, y1 + 3); }
      } else {
        for (let k = 0; k * step <= W + 1e-9; k++) { const sBack = k * step, x = X(tEnd - sBack); vline(x); if (k > 0) ctx.fillText(`−${+sBack.toFixed(3)} s`, x, y1 + 3); }
      }
      ctx.globalAlpha = 1;
    }
    // Nothing a series does may leave the plot area (the axes and labels live outside it).
    ctx.save();
    ctx.beginPath(); ctx.rect(x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2); ctx.clip();
    ctx.lineJoin = "round";
    const colOf = (tt) => Math.floor(X(tt) - x0);
    for (const s of series) {
      const ts = s.t || t, off = s.tOffset || 0, m = Math.min(ts.length, s.values.length);
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || (opts.bare ? 1 : 1.5);
      ctx.setLineDash(s.dash || []); ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
      ctx.beginPath();
      let pen = false;
      if (m <= 2 * (x1 - x0)) {
        for (let i = 0; i < m; i++) {
          const v = s.values[i];
          if (missing(v)) { pen = false; continue; }
          if (pen && ts[i] - ts[i - 1] > gap) pen = false;                 // missing samples: show the hole
          const x = X(ts[i] + off), y = Y(v);
          if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
        }
      } else {
        // More samples than pixels: one min–max bar per pixel column keeps every spike visible.
        let col = null, clo = 0, chi = 0;
        for (let i = 0; i <= m; i++) {
          const v = i < m ? s.values[i] : null, c = i < m ? colOf(ts[i] + off) : null;
          const isGap = i > 0 && i < m && ts[i] - ts[i - 1] > gap;
          if (col !== null && (c !== col || missing(v) || isGap)) {
            const x = x0 + col + 0.5;
            if (pen) ctx.lineTo(x, Y(chi)); else ctx.moveTo(x, Y(chi));
            ctx.lineTo(x, Y(clo)); pen = true; col = null;
          }
          if (missing(v)) { pen = false; continue; }
          if (isGap) pen = false;
          if (col === null) { col = c; clo = chi = v; } else { if (v < clo) clo = v; if (v > chi) chi = v; }
        }
      }
      ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    if (opts.cursor !== undefined && opts.cursor !== null && Number.isFinite(opts.cursor)) {
      const x = Math.round(X(opts.cursor)) + 0.5;
      ctx.lineWidth = 1; ctx.strokeStyle = muted; ctx.globalAlpha = 0.9;
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  const dB = (a) => 20 * Math.log10(Math.max(a, 1e-6));
  /** The highest bin of `spec` at or above `fmin` Hz (default 1) and below `fmax`: { k, f, amp, dB }, or null. */
  function spectrumPeak(spec, fmin, fmax) {
    if (!spec) return null;
    fmin = fmin === undefined ? 1 : fmin;
    fmax = fmax === undefined ? Infinity : fmax;
    let peak = -1;
    for (let k = 1; k < spec.f.length && spec.f[k] <= fmax; k++) if (spec.f[k] >= fmin && (peak < 0 || spec.amp[k] > spec.amp[peak])) peak = k;
    return peak > 0 ? { k: peak, f: spec.f[peak], amp: spec.amp[peak], dB: dB(spec.amp[peak]) } : null;
  }
  /** The readout line for a spectrum: its peak, bin width and the segments it averages. */
  function spectrumReadout(spec, unit, fmax) {
    const pk = spectrumPeak(spec, 1, fmax);
    return (pk ? `peak ${pk.f.toFixed(2)} Hz: ${pk.amp.toPrecision(3)} ${unit} (${pk.dB.toFixed(1)} dB)   ` : "")
      + `Δf ${spec.df.toFixed(2)} Hz   ${spec.segments} × ${spec.n} samples at ${spec.fs.toFixed(0)} Hz`;
  }

  /**
   * The amplitude spectrum `spec` (from scopeDSP.welch) in dB, up to `opts.fmax` Hz (clipped to
   * half the sample rate), with the highest peak above 1 Hz marked. `opts`: unit (for the readout),
   * accent (the trace colour), empty (the text shown when spec is null), others (more spectra
   * drawn under it on the same axes, [{ spec, color, dash, alpha }], sharing the dB scale).
   * Returns the readout line of `spec`.
   */
  function drawSpectrumStrip(canvas, spec, opts) {
    const c = setupCanvas(canvas);
    if (!c) return "";
    const { ctx, W: Wpx, H, lineColor, muted } = c;
    const accent = opts.accent || "#6ea8fe", unit = opts.unit || "";
    const x0 = PLOT_LEFT, x1 = Wpx - PLOT_RIGHT, y0 = 6, y1 = H - 16;
    ctx.fillStyle = muted;
    if (!spec) {
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(opts.empty || "waiting for 64 gap-free samples", (x0 + x1) / 2, (y0 + y1) / 2);
      return "";
    }
    const fmax = Math.min(opts.fmax || spec.fs / 2, spec.fs / 2);
    const bins = (sp) => Math.min(sp.f.length, Math.floor(fmax / sp.df) + 1);
    const others = (opts.others || []).filter((o) => o && o.spec);
    let lo = Infinity, hi = -Infinity;
    for (const sp of others.map((o) => o.spec).concat([spec])) {
      const nb = bins(sp);
      for (let k = 1; k < nb; k++) {                             // the DC bin is left out: detrended, it only says how well
        const v = dB(sp.amp[k]);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const nb = bins(spec), pk = spectrumPeak(spec, 1, fmax), peak = pk ? pk.k : -1;
    if (!Number.isFinite(lo)) { lo = -60; hi = 0; }
    lo = Math.floor((lo - 3) / 10) * 10; hi = Math.ceil((hi + 3) / 10) * 10;
    if (hi - lo < 30) hi = lo + 30;
    const X = (f) => x0 + (x1 - x0) * f / fmax, Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
    ctx.lineWidth = 1; ctx.strokeStyle = lineColor;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of niceTicks(lo, hi, 4)) {
      const y = Y(v);
      ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(v.toFixed(0), x0 - 5, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.globalAlpha = 0.6;
    for (const f of niceTicks(0, fmax, 6)) {
      const x = X(f); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
      if (x < x1 - 24) ctx.fillText(`${+f.toFixed(2)} Hz`, x, y1 + 3);
    }
    ctx.globalAlpha = 1;
    ctx.save();
    ctx.beginPath(); ctx.rect(x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2); ctx.clip();
    ctx.lineWidth = 1.5; ctx.lineJoin = "round";
    for (const o of others) {
      const sp = o.spec, n = bins(sp);
      ctx.strokeStyle = o.color || accent; ctx.setLineDash(o.dash || []); ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha; ctx.beginPath();
      for (let k = 1; k < n; k++) { const x = X(sp.f[k]), y = Y(dB(sp.amp[k])); if (k === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.strokeStyle = accent; ctx.beginPath();
    for (let k = 1; k < nb; k++) { const x = X(spec.f[k]), y = Y(dB(spec.amp[k])); if (k === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke();
    if (peak > 0) { ctx.fillStyle = accent; ctx.beginPath(); ctx.arc(X(spec.f[peak]), Y(dB(spec.amp[peak])), 3.5, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
    if (peak > 0) {
      const x = X(spec.f[peak]), y = Math.max(y0 + 12, Y(dB(spec.amp[peak]))), right = x > (x0 + x1) / 2;
      ctx.fillStyle = muted; ctx.textAlign = right ? "right" : "left"; ctx.textBaseline = "bottom";
      ctx.fillText(`${spec.f[peak].toFixed(2)} Hz`, x + (right ? -6 : 6), y - 2);
    }
    return spectrumReadout(spec, unit, fmax);
  }

  return { PLOT_LEFT, PLOT_RIGHT, GAP, niceStep, niceTicks, fmtTick, lowerBound, upperBound, setupCanvas, drawStrip, spectrumPeak, spectrumReadout, drawSpectrumStrip };
});
