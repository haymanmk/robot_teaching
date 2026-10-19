// Checks for src/robot_teaching/server/static/dsp.js. Run: node tests/dsp_check.js
// (tests/test_dsp.py runs it under pytest when node is installed.)
const path = require("path");
const dsp = require(path.join(__dirname, "..", "src", "robot_teaching", "server", "static", "dsp.js"));
let failures = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures++; };

// 1. The FFT agrees with a direct DFT.
{
  const n = 64, re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { re[i] = Math.sin(i * 0.37) + 0.3 * Math.cos(i * 1.91); im[i] = 0.1 * i; }
  const xr = Array.from(re), xi = Array.from(im);
  dsp.fft(re, im);
  let worst = 0;
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let j = 0; j < n; j++) { const a = -2 * Math.PI * k * j / n; sr += xr[j] * Math.cos(a) - xi[j] * Math.sin(a); si += xr[j] * Math.sin(a) + xi[j] * Math.cos(a); }
    worst = Math.max(worst, Math.abs(sr - re[k]), Math.abs(si - im[k]));
  }
  check(worst < 1e-9, `fft matches a direct DFT (largest error ${worst.toExponential(1)})`);
}

// Two tones on a linear drift with a little noise, sampled at fs for some seconds.
function synth(fs, seconds, f1, a1, f2, a2) {
  const t = [], v = [];
  for (let i = 0; i < Math.round(fs * seconds); i++) {
    const tt = i / fs;
    t.push(tt);
    v.push(a1 * Math.sin(2 * Math.PI * f1 * tt) + a2 * Math.sin(2 * Math.PI * f2 * tt + 1) + 0.5 * tt + 0.001 * Math.sin(i));
  }
  return { t, v };
}
const peakNear = (s, f0, half) => { let best = -1; for (let k = 1; k < s.f.length; k++) if (Math.abs(s.f[k] - f0) <= half && (best < 0 || s.amp[k] > s.amp[best])) best = k; return best; };

// 2. The peaks sit at the tones with their amplitudes; the drift is detrended away.
{
  const { t, v } = synth(500, 8, 7.3, 0.2, 23.1, 0.05);
  const s = dsp.welch(v, t, { segmentSeconds: 2 });
  check(s && s.n === 1024 && Math.abs(s.fs - 500) < 1e-6 && s.segments === 6, `8 s at 500 Hz: 6 half-overlapping segments of 1024 (got ${s && s.segments} × ${s && s.n}, fs ${s && s.fs.toFixed(1)})`);
  const k1 = peakNear(s, 7.3, 1), k2 = peakNear(s, 23.1, 1);
  check(Math.abs(s.f[k1] - 7.3) <= s.df && Math.abs(s.amp[k1] - 0.2) < 0.03, `7.3 Hz tone: peak at ${s.f[k1].toFixed(2)} Hz, amplitude ${s.amp[k1].toFixed(3)} (0.200)`);
  check(Math.abs(s.f[k2] - 23.1) <= s.df && Math.abs(s.amp[k2] - 0.05) < 0.01, `23.1 Hz tone: peak at ${s.f[k2].toFixed(2)} Hz, amplitude ${s.amp[k2].toFixed(3)} (0.050)`);
  let top = 1;
  for (let k = 2; k < s.f.length; k++) if (s.amp[k] > s.amp[top]) top = k;
  check(top === k1, `the largest bin above DC is the 7.3 Hz tone (${s.f[top].toFixed(2)} Hz)`);
  check(s.amp[0] < 0.01 && s.amp[1] < 0.02, `the drift is gone (DC ${s.amp[0].toExponential(1)}, first bin ${s.amp[1].toFixed(4)}, tones 0.2 and 0.05)`);
}

// 3. A hole in the data: the segments across it are left out, the rest still finds the tone.
{
  const { t, v } = synth(500, 8, 7.3, 0.2, 23.1, 0.05);
  const cut0 = Math.round(500 * 3.0), cut1 = Math.round(500 * 3.5);
  t.splice(cut0, cut1 - cut0); v.splice(cut0, cut1 - cut0);
  const s = dsp.welch(v, t, { segmentSeconds: 2 });
  const full = 1 + Math.floor((v.length - 1024) / 512);
  check(s && s.segments >= 2 && s.segments < full, `a 0.5 s hole leaves ${s && s.segments} gap-free segments of ${full}`);
  check(Math.abs(s.fs - 500) < 1e-6, `the hole does not bias the sample rate (${s.fs.toFixed(2)} Hz)`);
  const k1 = peakNear(s, 7.3, 1);
  check(Math.abs(s.amp[k1] - 0.2) < 0.03, `the tone still reads ${s.amp[k1].toFixed(3)}`);
}

// 4. A null (a missing motor torque) drops the segments that hold it; too little data gives null.
{
  const { t, v } = synth(500, 4, 7.3, 0.2, 23.1, 0.05);
  v[1600] = null;
  const s = dsp.welch(v, t, { segmentSeconds: 2 });
  check(s && s.segments === 1, `one null drops the segment that holds it (${s && s.segments} left of 2)`);
  check(dsp.welch(v.slice(0, 40), t.slice(0, 40)) === null, "fewer than 64 samples: null");
  check(dsp.welch(v.map(() => null), t) === null, "no gap-free segment: null");
}

// 5. A long window is covered by at most maxSegments segments, newest sample included.
{
  const { t, v } = synth(500, 60, 7.3, 0.2, 23.1, 0.05);
  const s = dsp.welch(v, t, { segmentSeconds: 2, maxSegments: 32 });
  check(s && s.segments <= 32 && s.segments >= 28, `60 s: ${s && s.segments} segments (≤ 32)`);
  const k1 = peakNear(s, 7.3, 1);
  check(Math.abs(s.amp[k1] - 0.2) < 0.03, `the tone reads ${s.amp[k1].toFixed(3)} over 60 s`);
}

// 6. A short window falls back to shorter segments instead of giving up.
{
  const { t, v } = synth(500, 0.6, 7.3, 0.2, 23.1, 0.05);
  const s = dsp.welch(v, t, { segmentSeconds: 2 });
  check(s && s.n === 256 && s.df > 1.9, `0.6 s of data: ${s && s.n}-point segments, Δf ${s && s.df.toFixed(2)} Hz`);
}

console.log(failures ? `${failures} FAILED` : "all dsp checks passed");
process.exit(failures ? 1 : 0);
