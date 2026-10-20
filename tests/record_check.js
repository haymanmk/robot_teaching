// Checks for src/robot_teaching/server/static/record.js (the scope recording reader) and the axis
// helpers of scope.js. Run: node tests/record_check.js (tests/test_record.py runs it under pytest).
const path = require("path");
const dir = path.join(__dirname, "..", "src", "robot_teaching", "server", "static");
const rec = require(path.join(dir, "record.js"));
const draw = require(path.join(dir, "scope.js"));
let failures = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures++; };
const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

// A recording the way the pendant writes it: 500 Hz, one joint, optional empty cells, a run column.
const HEADER = "t_s,joint,q_cmd_rad,q_rad,err_rad,tau_cmd_Nm,tau_meas_Nm,tau_g_Nm,tau_f_Nm,tau_i_Nm,run";
const OLD_HEADER = "t_s,joint,q_cmd_rad,q_rad,err_rad,tau_cmd_Nm,tau_meas_Nm,run";
// The feed-forward terms: gravity 1.0·cos t, friction 0.2, inertia 0.05·sin t, so their sum is the
// commanded torque minus a PD part of 0.3·cos t.
function row(t, qc, q, tc, tm, run, old) {
  const e = qc === null || q === null ? "" : (q - qc).toFixed(6);
  const n = (v, d) => (v === null ? "" : v.toFixed(d));
  const ff = old ? "" : `${(1.0 * Math.cos(t)).toFixed(4)},${(0.2).toFixed(4)},${(0.05 * Math.sin(t)).toFixed(4)},`;
  return `${t.toFixed(4)},J2,${n(qc, 6)},${n(q, 6)},${e},${n(tc, 4)},${n(tm, 4)},${ff}${run}`;
}
function synth(seconds, run, opts) {
  const o = Object.assign({ fs: 500, t0: 0, hole: null, nullAt: null, nullTorqueAt: null, old: false }, opts || {});
  const rows = [];
  for (let i = 0; i < Math.round(o.fs * seconds); i++) {
    const t = o.t0 + i / o.fs;
    if (o.hole && t >= o.hole[0] && t < o.hole[1]) continue;
    const qc = 0.5 * Math.sin(t), q = qc + 0.01 * Math.sin(2 * Math.PI * 3 * t);
    rows.push(row(t, qc, i === o.nullAt ? null : q, 1.5 * Math.cos(t), i === o.nullTorqueAt ? null : 1.4 * Math.cos(t), run, o.old));
  }
  return rows;
}

// 1. A plain file: columns, units, the joint, one run, a trailing newline.
{
  const r = rec.parse(HEADER + "\n" + synth(2, 1).join("\n") + "\n");
  check(r.joint === "J2" && r.rows === 1000 && r.runs.length === 1 && r.problems.length === 0, `2 s at 500 Hz: ${r.rows} rows, ${r.runs.length} run, joint ${r.joint}`);
  const run = r.runs[0];
  check(run.t instanceof Float64Array && run.t.length === 1000 && close(run.t[1] - run.t[0], 0.002, 1e-9), "typed columns at the sample step");
  check(close(run.q_cmd_deg[100], run.q_cmd[100] * 180 / Math.PI) && close(run.err_deg[100], (run.q[100] - run.q_cmd[100]) * 180 / Math.PI), "degrees derived from the radians");
  const s = rec.summarize(run);
  check(s.samples === 1000 && close(s.rate, 500, 0.01) && s.gaps === 0 && s.missing === 0, `summary: ${s.samples} samples at ${s.rate.toFixed(1)} Hz, ${s.gaps} holes`);
  check(close(s.errMax, 0.01 * 180 / Math.PI, 1e-3) && close(s.errRms, 0.01 * 180 / Math.PI / Math.SQRT2, 2e-3), `max |error| ${s.errMax.toFixed(4)}° (0.5730), rms ${s.errRms.toFixed(4)}° (0.4052)`);
  check(close(s.tauMax, 1.5, 1e-3), `peak |torque| ${s.tauMax.toFixed(3)} N·m`);
  check(close(run.tau_g[0], 1.0, 1e-4) && close(run.tau_f[0], 0.2, 1e-4) && close(run.tau_i[250], 0.05 * Math.sin(0.5), 1e-4) && close(run.tau_ff[250], Math.cos(0.5) + 0.2 + 0.05 * Math.sin(0.5), 1e-4), `the feed-forward terms and their sum (${run.tau_ff[250].toFixed(4)} N·m at 0.5 s)`);
  // The torque error: reported 1.4·cos t against commanded 1.5·cos t, so −0.1·cos t; over 2 s its
  // mean square is 0.01·(1/2 + sin 4 / 8).
  const rmsExpected = 0.1 * Math.sqrt(0.5 + Math.sin(4) / 8);
  check(close(run.tau_err[0], -0.1, 1e-3) && close(s.tauErrMax, 0.1, 1e-3) && close(s.tauErrAt, 0) && close(s.tauErrRms, rmsExpected, 1e-3), `torque error: max ${s.tauErrMax.toFixed(3)} N·m at ${s.tauErrAt} s, rms ${s.tauErrRms.toFixed(4)} (${rmsExpected.toFixed(4)})`);
}

// 2. Empty cells read as NaN (and are counted), CRLF line ends and a file without a trailing newline.
{
  const text = HEADER + "\r\n" + synth(1, 1, { nullAt: 10, nullTorqueAt: 20 }).join("\r\n");
  const r = rec.parse(text);
  const run = r.runs[0];
  check(r.rows === 500 && Number.isNaN(run.q[10]) && Number.isNaN(run.err_deg[10]) && Number.isFinite(run.q_cmd[10]), "an empty measurement is NaN; the command beside it is kept");
  check(Number.isNaN(run.tau_meas[20]) && Number.isFinite(run.tau_cmd[20]) && Number.isNaN(run.tau_err[20]), "an empty motor torque is NaN, and so is the torque error there");
  check(rec.summarize(run, 15, 25).samples === 10 && Number.isFinite(rec.summarize(run, 15, 25).tauErrRms), "the torque error statistics skip the row without a reported torque");
  check(close(run.t[499], 0.998) && Number.isFinite(run.q[499]), "CRLF and no final newline: the last row is read");
  check(rec.summarize(run).missing === 1, "the row without a measurement is counted");
}

// 3. A hole in the data is a gap in the summary and not a sample-rate bias.
{
  const r = rec.parse(HEADER + "\n" + synth(4, 1, { hole: [1.0, 1.5] }).join("\n") + "\n");
  const s = rec.summarize(r.runs[0]);
  check(r.runs.length === 1 && s.gaps === 1 && close(s.rate, 500, 0.01), `a 0.5 s hole: ${s.gaps} hole, rate ${s.rate.toFixed(1)} Hz, still one run`);
  const i = draw.lowerBound(r.runs[0].t, 1.2);
  check(close(r.runs[0].t[i], 1.5), `the first sample after the hole is at ${r.runs[0].t[i].toFixed(3)} s`);
  const part = rec.summarize(r.runs[0], 0, i);
  check(part.samples === 500 && part.gaps === 0 && close(part.t1, 0.998), "a summary of a range stops before the hole");
}

// 4. A server restart: the run column changes and the clock starts again; two runs.
{
  const rows = synth(1, 1, { t0: 10 }).concat(synth(0.5, 2));
  const r = rec.parse(HEADER + "\n" + rows.join("\n") + "\n");
  check(r.runs.length === 2 && r.runs[0].run === 1 && r.runs[1].run === 2, `${r.runs.length} runs (1 and 2)`);
  check(r.runs[0].t.length === 500 && close(r.runs[0].t[0], 10) && r.runs[1].t.length === 250 && close(r.runs[1].t[0], 0), "each run holds its own rows and clock");
  // The clock stepping backwards within one run value also starts a new run.
  const rows2 = synth(1, 1, { t0: 10 }).concat(synth(0.5, 1));
  check(rec.parse(HEADER + "\n" + rows2.join("\n") + "\n").runs.length === 2, "a clock that steps backwards starts a new run");
}

// 5. Columns are found by name; a file without the torques still loads; a wrong file is refused.
{
  const text = "run,q_rad,t_s,q_cmd_rad\n1,0.1,0.000,0.0\n1,0.2,0.002,0.1\n";
  const r = rec.parse(text);
  check(r.rows === 2 && close(r.runs[0].q[1], 0.2) && close(r.runs[0].q_cmd[1], 0.1) && close(r.runs[0].t[1], 0.002), "columns in another order are read by name");
  check(r.joint === "" && Number.isNaN(r.runs[0].tau_cmd[0]) && r.problems.length === 3, `missing torque and feed-forward columns are reported (${r.problems.join("; ")})`);
  const old = rec.parse(OLD_HEADER + "\n" + synth(0.2, 1, { old: true }).join("\n") + "\n");
  check(old.rows === 100 && Number.isNaN(old.runs[0].tau_ff[0]) && Number.isFinite(old.runs[0].tau_cmd[0]) && old.problems.length === 1 && old.problems[0].includes("feed-forward"), `an older recording loads without the feed-forward (${old.problems.join("; ")})`);
  let err = null;
  try { rec.parse("t_s,q_rad,qd_rad\n0,1,2\n"); } catch (e) { err = e.message; }
  check(err && err.includes("q_cmd_rad"), `a file without q_cmd_rad is refused: ${err}`);
  err = null;
  try { rec.parse(HEADER + "\n"); } catch (e) { err = e.message; }
  check(err && err.includes("no samples"), `a header alone is refused: ${err}`);
  const r2 = rec.parse(HEADER + "\n" + "abc,J2,1,2,3,4,5,1\n" + synth(0.2, 1).join("\n") + "\n");
  check(r2.rows === 100 && r2.problems.some((p) => p.includes("skipped")), `a row without a time is skipped and reported (${r2.problems.join("; ")})`);
}

// 6. The axis helpers shared with the live scope.
{
  check(draw.niceStep(5, 6) === 1 && draw.niceStep(0.5, 6) === 0.1 && draw.niceStep(60, 6) === 10, "nice steps for 5 s, 0.5 s and 60 s windows");
  const ticks = draw.niceTicks(-0.3, 10.2, 6);
  check(ticks.length === 6 && ticks[0] === 0 && ticks[5] === 10, `ticks on [−0.3, 10.2]: ${ticks.join(", ")}`);
  const a = [0, 1, 1, 2, 3];
  check(draw.lowerBound(a, 1) === 1 && draw.upperBound(a, 1) === 3 && draw.lowerBound(a, 5) === 5 && draw.upperBound(a, -1) === 0, "lowerBound / upperBound");
  check(draw.fmtTick(123.4) === "123" && draw.fmtTick(12.34) === "12.3" && draw.fmtTick(1.234) === "1.23" && draw.fmtTick(0.0123) === "0.012", "tick labels");
}

// 7. The spectrum peak helper shared by the two pages: the highest bin at or above 1 Hz, within fmax.
{
  const spec = { f: new Float64Array([0, 0.5, 1, 1.5, 2, 2.5]), amp: new Float64Array([5, 2, 0.2, 0.9, 0.3, 1.2]), df: 0.5, fs: 5, n: 10, segments: 1 };
  const pk = draw.spectrumPeak(spec);
  check(pk && pk.k === 5 && close(pk.f, 2.5) && close(pk.dB, 20 * Math.log10(1.2)), `peak at ${pk && pk.f} Hz (${pk && pk.dB.toFixed(2)} dB), the DC and 0.5 Hz bins left out`);
  const pk2 = draw.spectrumPeak(spec, 1, 2);
  check(pk2 && close(pk2.f, 1.5), `peak below 2 Hz at ${pk2 && pk2.f} Hz`);
  check(draw.spectrumPeak(null) === null, "no spectrum: null");
  const line = draw.spectrumReadout(spec, "°", 2);
  check(line.startsWith("peak 1.50 Hz: 0.900 °") && line.includes("Δf 0.50 Hz") && line.includes("1 × 10 samples at 5 Hz"), `readout: ${line}`);
}

console.log(failures ? `${failures} FAILED` : "all record checks passed");
process.exit(failures ? 1 : 0);
