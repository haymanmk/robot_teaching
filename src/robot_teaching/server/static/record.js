/* Reading a scope recording: the CSV the pendant's Record button writes (columns t_s, joint,
   q_cmd_rad, q_rad, err_rad, tau_cmd_Nm, tau_meas_Nm, run; one joint per file). Plain functions
   without DOM access, so Node can run tests/record_check.js on them; in the browser they live on
   window.scopeRecord. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api; else root.scopeRecord = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const GAP = 0.25;                                               // s without samples: a hole (as in the scope)
  const FIELDS = ["t", "q_cmd", "q", "tau_cmd", "tau_meas"];
  const COLUMNS = { t: "t_s", q_cmd: "q_cmd_rad", q: "q_rad", tau_cmd: "tau_cmd_Nm", tau_meas: "tau_meas_Nm" };
  const DEG = 180 / Math.PI;

  /**
   * Parse the text of a recording. Columns are found by name in the header line, so their order
   * does not matter; `t_s`, `q_cmd_rad` and `q_rad` are required, the torques and `run` optional.
   * An empty cell (a sample without a measurement) reads as NaN. The rows are split into runs:
   * a new run starts where the `run` column changes or where `t_s` steps backwards (a server
   * restart; its clock starts again from zero). Returns { joint, rows, runs, problems }; each run
   * is { run, t, q_cmd, q, tau_cmd, tau_meas, q_cmd_deg, q_deg, err_deg } with Float64Array
   * columns, the last three in degrees for the display.
   */
  function parse(text) {
    const problems = [];
    let pos = 0, end = text.indexOf("\n");
    if (end < 0) end = text.length;
    const header = text.slice(0, end).replace(/\r$/, "").split(",").map((s) => s.trim());
    const col = {};
    for (const f of FIELDS) col[f] = header.indexOf(COLUMNS[f]);
    const absent = FIELDS.filter((f) => col[f] < 0 && f !== "tau_cmd" && f !== "tau_meas").map((f) => COLUMNS[f]);
    if (absent.length) throw new Error(`not a scope recording: no ${absent.join(", ")} column (header: ${header.slice(0, 8).join(", ")})`);
    const jointCol = header.indexOf("joint"), runCol = header.indexOf("run");
    for (const f of ["tau_cmd", "tau_meas"]) if (col[f] < 0) problems.push(`no ${COLUMNS[f]} column`);
    // Count the lines once so the columns can be typed arrays.
    let lines = 0;
    for (let i = text.indexOf("\n", end + 1); i >= 0; i = text.indexOf("\n", i + 1)) lines++;
    lines += 2;
    const data = {};
    for (const f of FIELDS) data[f] = new Float64Array(lines);
    const runOf = new Float64Array(lines);
    let n = 0, joint = "", skipped = 0, jointChanged = false;
    const cells = [];
    pos = end + 1;
    while (pos < text.length) {
      end = text.indexOf("\n", pos);
      if (end < 0) end = text.length;
      let stop = end;
      if (stop > pos && text.charCodeAt(stop - 1) === 13) stop--;             // CRLF
      if (stop > pos) {
        cells.length = 0;
        for (let a = pos; a <= stop;) {
          let b = text.indexOf(",", a);
          if (b < 0 || b > stop) b = stop;
          cells.push(text.slice(a, b));
          a = b + 1;
        }
        const tv = cells[col.t] === undefined || cells[col.t] === "" ? NaN : Number(cells[col.t]);
        if (!Number.isFinite(tv)) {
          skipped++;
        } else {
          for (const f of FIELDS) {
            const s = col[f] < 0 ? undefined : cells[col[f]];
            data[f][n] = s === undefined || s === "" ? NaN : Number(s);
          }
          runOf[n] = runCol < 0 || cells[runCol] === undefined || cells[runCol] === "" ? 1 : Number(cells[runCol]);
          if (jointCol >= 0) {
            const j = cells[jointCol] === undefined ? "" : cells[jointCol].trim();
            if (!joint) joint = j; else if (j && j !== joint) jointChanged = true;
          }
          n++;
        }
      }
      pos = end + 1;
    }
    if (skipped) problems.push(`${skipped} row${skipped === 1 ? "" : "s"} without a time skipped`);
    if (jointChanged) problems.push("the joint column changes within the file; the first value is used");
    if (!n) throw new Error("the file has no samples");
    // Split into runs.
    const runs = [];
    let start = 0;
    for (let i = 1; i <= n; i++) {
      if (i === n || runOf[i] !== runOf[i - 1] || data.t[i] < data.t[i - 1]) {
        const r = { run: runOf[start] };
        for (const f of FIELDS) r[f] = data[f].subarray(start, i);
        r.q_cmd_deg = new Float64Array(i - start);
        r.q_deg = new Float64Array(i - start);
        r.err_deg = new Float64Array(i - start);
        for (let k = 0; k < i - start; k++) {
          r.q_cmd_deg[k] = r.q_cmd[k] * DEG;
          r.q_deg[k] = r.q[k] * DEG;
          r.err_deg[k] = (r.q[k] - r.q_cmd[k]) * DEG;                       // NaN when either is missing
        }
        runs.push(r);
        start = i;
      }
    }
    return { joint, rows: n, runs, problems };
  }

  /**
   * Statistics of the samples [i0, i1) of a run (default: all of it): count, time span, mean
   * sample rate (over the steps below `gap`), holes (steps above `gap`), rows without a
   * measurement, the largest |error| in degrees and where it is, the RMS error and the largest
   * |commanded torque|.
   */
  function summarize(run, i0, i1, gap) {
    i0 = i0 === undefined ? 0 : Math.max(0, i0);
    i1 = i1 === undefined ? run.t.length : Math.min(run.t.length, i1);
    gap = gap === undefined ? GAP : gap;
    const out = { samples: Math.max(0, i1 - i0), t0: NaN, t1: NaN, duration: 0, rate: NaN, gaps: 0, missing: 0, errMax: NaN, errMaxAt: NaN, errRms: NaN, tauMax: NaN };
    if (i1 <= i0) return out;
    out.t0 = run.t[i0]; out.t1 = run.t[i1 - 1]; out.duration = out.t1 - out.t0;
    let dsum = 0, dcount = 0, esum = 0, ecount = 0, emax = -1, eat = NaN, tmax = -1;
    for (let i = i0; i < i1; i++) {
      if (i > i0) { const d = run.t[i] - run.t[i - 1]; if (d > gap) out.gaps++; else if (d > 0) { dsum += d; dcount++; } }
      const e = run.err_deg[i];
      if (Number.isFinite(e)) { esum += e * e; ecount++; if (Math.abs(e) > emax) { emax = Math.abs(e); eat = run.t[i]; } } else out.missing++;
      const tau = run.tau_cmd[i];
      if (Number.isFinite(tau) && Math.abs(tau) > tmax) tmax = Math.abs(tau);
    }
    if (dcount) out.rate = dcount / dsum;
    if (ecount) { out.errMax = emax; out.errMaxAt = eat; out.errRms = Math.sqrt(esum / ecount); }
    if (tmax >= 0) out.tauMax = tmax;
    return out;
  }

  return { GAP, FIELDS, COLUMNS, parse, summarize };
});
