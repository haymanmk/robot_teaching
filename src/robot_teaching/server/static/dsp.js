/* Spectrum estimation for the scope: a radix-2 FFT and a Welch amplitude spectrum from Hann-windowed,
   detrended segments. Plain functions without DOM access, so Node can run tests/dsp_check.js on
   them; in the browser they live on window.scopeDSP. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api; else root.scopeDSP = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const tables = new Map();                       // N → twiddles and the bit-reversal permutation, reused

  function table(n) {
    let tb = tables.get(n);
    if (tb) return tb;
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2), rev = new Uint32Array(n);
    for (let k = 0; k < n / 2; k++) { cos[k] = Math.cos(2 * Math.PI * k / n); sin[k] = Math.sin(2 * Math.PI * k / n); }
    const bits = Math.round(Math.log2(n));
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
    tb = { cos, sin, rev };
    tables.set(n, tb);
    return tb;
  }

  /** In-place forward FFT of the complex sequence (re, im); the length must be a power of two. */
  function fft(re, im) {
    const n = re.length;
    if (n < 2 || (n & (n - 1)) !== 0 || im.length !== n) throw new Error("fft: length must be a power of two");
    const { cos, sin, rev } = table(n);
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let x = re[i]; re[i] = re[j]; re[j] = x; x = im[i]; im[i] = im[j]; im[j] = x; }
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2, step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0, tw = 0; k < half; k++, tw += step) {
          const i = start + k, j = i + half;
          const wr = cos[tw], wi = -sin[tw];                       // e^(−2πi·k/size)
          const xr = re[j] * wr - im[j] * wi, xi = re[j] * wi + im[j] * wr;
          re[j] = re[i] - xr; im[j] = im[i] - xi;
          re[i] += xr; im[i] += xi;
        }
      }
    }
  }

  function hann(n) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / n));
    return w;
  }

  /** Remove the mean and the linear trend of y, in place. */
  function detrend(y) {
    const n = y.length, xm = (n - 1) / 2;
    let ym = 0;
    for (let i = 0; i < n; i++) ym += y[i];
    ym /= n;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { const dx = i - xm; sxy += dx * (y[i] - ym); sxx += dx * dx; }
    const slope = sxx > 0 ? sxy / sxx : 0;
    for (let i = 0; i < n; i++) y[i] -= ym + slope * (i - xm);
  }

  const pow2floor = (x) => Math.pow(2, Math.floor(Math.log2(x)));
  const pow2round = (x) => Math.pow(2, Math.round(Math.log2(x)));

  /**
   * Welch amplitude spectrum of `values` sampled at the ascending times `t` (s).
   *
   * Segments of about `segmentSeconds` (the nearest power of two of samples, at least `minSamples`) are
   * Hann-windowed after removing their mean and linear trend, and their power spectra averaged.
   * They overlap by half, or less when more than `maxSegments` would be needed to cover the data;
   * the newest sample is always in the last one. A segment that holds a null or a time step above
   * `gap` seconds is left out. The sample rate is the mean step below `gap`.
   *
   * `amp[k]` is the amplitude of a sinusoid at `f[k]` (a tone of amplitude A reads A at its bin,
   * within the window's scalloping); `df` is the bin width. Returns null when too few samples
   * remain or no segment qualifies.
   */
  function welch(values, t, opts) {
    const o = Object.assign({ segmentSeconds: 2.0, maxSegments: 32, gap: 0.25, minSamples: 64 }, opts || {});
    const m = Math.min(values.length, t.length);
    if (m < o.minSamples) return null;
    let dsum = 0, dcount = 0;
    for (let i = 1; i < m; i++) { const d = t[i] - t[i - 1]; if (d > 0 && d <= o.gap) { dsum += d; dcount++; } }
    if (!dcount) return null;
    const fs = dcount / dsum;
    const n = Math.max(o.minSamples, Math.min(pow2round(o.segmentSeconds * fs), pow2floor(m)));
    if (n > m) return null;
    let hop = n / 2;
    if (1 + Math.floor((m - n) / hop) > o.maxSegments) hop = Math.ceil((m - n) / (o.maxSegments - 1));
    const w = hann(n);
    let s1 = 0;
    for (let i = 0; i < n; i++) s1 += w[i];
    const re = new Float64Array(n), im = new Float64Array(n), acc = new Float64Array(n / 2 + 1);
    let segments = 0;
    for (let start = m - n; start >= 0; start -= hop) {
      let ok = true;
      for (let i = 0; i < n && ok; i++) {
        const v = values[start + i];
        if (v === null || v === undefined || !Number.isFinite(v) || (i > 0 && t[start + i] - t[start + i - 1] > o.gap)) ok = false;
        else re[i] = v;
      }
      if (!ok) continue;
      detrend(re);
      for (let i = 0; i < n; i++) { re[i] *= w[i]; im[i] = 0; }
      fft(re, im);
      for (let k = 0; k <= n / 2; k++) acc[k] += re[k] * re[k] + im[k] * im[k];
      segments++;
    }
    if (!segments) return null;
    const f = new Float64Array(n / 2 + 1), amp = new Float64Array(n / 2 + 1);
    for (let k = 0; k <= n / 2; k++) {
      f[k] = k * fs / n;
      amp[k] = Math.sqrt(acc[k] / segments) * (k === 0 || k === n / 2 ? 1 : 2) / s1;
    }
    return { f, amp, n, segments, fs, df: fs / n };
  }

  return { fft, hann, detrend, welch };
});
