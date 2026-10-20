/* Offline plotter (plot.html): the pendant's scope strips for a CSV recording, read in the browser.
   Drawing is shared with the live scope (scope.js, dsp.js); the file is parsed by record.js. */
(() => {
  const $ = (id) => document.getElementById(id);
  const ZOOMS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 60, 120, 300, 600, 1800, 3600];   // s, the zoom steps (Fit goes beyond)
  const FFT_FMAX = [5, 10, 25, 50, 100, 250];                     // Hz, the spectrum's frequency-range steps (clipped to fs/2)
  const FFT_UNITS = { err: "°", tau_err: "N·m", q: "°", q_cmd: "°", tau_cmd: "N·m", tau_meas: "N·m", tau_ff: "N·m", tau_g: "N·m", tau_f: "N·m", tau_i: "N·m" };
  const FFT_SEGMENT = 2.0;                                        // s per Welch segment, as in the live scope
  const GAP = scopeDraw.GAP;
  const DASH = [4, 3], B_ALPHA = 0.7;                            // the comparison recording: dashed, a little lighter, the same hues
  const fmt = (v, d = 1) => (v === null || v === undefined || !Number.isFinite(v)) ? "—" : Number(v).toFixed(d);
  const fmtN = (n) => n.toLocaleString("en-US");

  const entries = [];                      // every run of every loaded file
  let cur = null;                          // the entry on screen
  const cmp = { id: null, offset: 0 };     // the comparison recording (B) and its shift (s) after aligning the starts
  const view = { W: 5, tEnd: 0 };          // the window (s) and the time at its right edge, in the run's clock
  const fft = { on: false, signal: "err", fmax: 50, spec: null, key: "", specB: null, keyB: "" };
  const ov = { key: "", cache: null };     // the overview trace, cached per run and canvas size
  const STRIPS = { pos: "plot-pos", err: "plot-err", tau: "plot-tau" };
  const yz = { pos: null, err: null, tau: null };       // a strip's value range [lo, hi], or null: fit the samples on screen
  const yBase = { pos: null, err: null, tau: null };    // the fitted span a strip had when its value zoom began
  const STRIP_HEIGHT = 130;                             // px, a strip's height while its value axis is automatic
  const frames = { pos: null, err: null, tau: null };   // the mapping each strip was last drawn with (scopeDraw.drawStrip)
  let errSig = "pos";                      // the tracking-error strip: "pos" (° measured − commanded) or "tau" (N·m reported − commanded)
  let cursorT = null;                      // the time under the pointer, or null
  const hidden = new Set();                // traces unticked in a legend (not kept across reloads)
  let drawPending = false;

  let toastTimer = null;
  function toast(msg, isErr = false) {
    const t = $("toast");
    t.textContent = msg; t.className = "toast" + (isErr ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.add("hidden"), isErr ? 6000 : 2500);
  }

  // ── loading ──────────────────────────────────────────────────────────────
  async function openFiles(list) {
    let last = null;
    for (const file of Array.from(list || [])) {
      let rec;
      try { rec = scopeRecord.parse(await file.text()); }
      catch (e) { toast(`${file.name}: ${e.message}`, true); continue; }
      for (const run of rec.runs) {
        const e = { id: entries.length, name: file.name, joint: rec.joint, run, nRuns: rec.runs.length, summary: scopeRecord.summarize(run), problems: rec.problems };
        e.label = `${file.name}${rec.runs.length > 1 ? ` · run ${run.run}` : ""}`;
        entries.push(e);
        last = e;
      }
      toast(`${file.name}: ${fmtN(rec.rows)} samples${rec.runs.length > 1 ? ` in ${rec.runs.length} runs` : ""}`);
    }
    if (last) select(last.id); else rebuildSelects();
  }
  const cmpEntry = () => (cmp.id === null || !cur || cmp.id === cur.id ? null : entries[cmp.id] || null);
  const shiftOf = (B) => cur.summary.t0 - B.summary.t0 + cmp.offset;   // B's clock + shift = A's clock
  function rebuildSelects() {
    const sel = $("plot-select"), cs = $("plot-compare");
    const option = (parent, value, text) => { const o = document.createElement("option"); o.value = value; o.textContent = text; parent.appendChild(o); };
    sel.innerHTML = ""; cs.innerHTML = "";
    if (!entries.length) option(sel, "", "none loaded");
    for (const e of entries) option(sel, String(e.id), e.label);
    option(cs, "", "none");
    for (const e of entries) if (!cur || e.id !== cur.id) option(cs, String(e.id), e.label);
    sel.value = cur ? String(cur.id) : "";
    cs.value = cmpEntry() ? String(cmp.id) : "";
    cs.disabled = entries.length < 2;
  }
  function select(id) {
    cur = entries[id] || null;
    if (cur && cmp.id === cur.id) cmp.id = null;
    fft.key = fft.keyB = ""; ov.key = ""; cursorT = null;
    for (const key of Object.keys(yz)) yz[key] = yBase[key] = null;
    rebuildSelects();
    showSummary();
    fit();
  }
  function compare(id) {
    cmp.id = id === null || id === undefined || !entries[id] ? null : id;
    cmp.offset = 0; $("plot-offset").value = "0";
    fft.keyB = ""; cursorT = null;
    rebuildSelects();
    showSummary();
    requestDraw();
  }
  function setOffset(v) {
    cmp.offset = Number.isFinite(v) ? v : 0;
    if (document.activeElement !== $("plot-offset")) $("plot-offset").value = cmp.offset.toFixed(3);
    showSummary();
    requestDraw();
  }
  function showSummary() {
    const s = $("plot-summary"), B = cmpEntry();
    const line = (e, tag) => {
      const m = e.summary, parts = [];
      if (tag) parts.push(tag);
      if (e.joint) parts.push(e.joint);
      if (e.nRuns > 1) parts.push(`run ${e.run.run} of ${e.nRuns}`);
      parts.push(`${fmt(m.duration, 1)} s`, `${fmtN(m.samples)} samples at ${fmt(m.rate, 0)} Hz`);
      if (m.gaps) parts.push(`${m.gaps} hole${m.gaps === 1 ? "" : "s"}`);
      if (m.missing) parts.push(`${fmtN(m.missing)} without a measurement`);
      parts.push(`max |error| ${fmt(m.errMax, 3)}° at ${fmt(m.errMaxAt, 2)} s`);
      return parts.concat(e.problems).join(" · ");
    };
    if (!cur) s.textContent = "";
    else if (!B) s.textContent = line(cur, "");
    else {
      const shift = cmp.offset ? `, shifted ${cmp.offset > 0 ? "+" : ""}${cmp.offset.toFixed(3)} s` : "";
      const other = B.joint && cur.joint && B.joint !== cur.joint ? " · a different joint" : "";
      s.textContent = `${line(cur, "A")}\n${line(B, "B")} · starts aligned${shift}${other}`;
    }
    $("plot-offset-wrap").classList.toggle("hidden", !B);
    document.querySelectorAll(".lg-b").forEach((el) => el.classList.toggle("hidden", !B));
  }

  // ── the window ───────────────────────────────────────────────────────────
  const runSpan = () => Math.max(0.1, cur.summary.duration);
  function fit() {
    if (cur) { view.W = runSpan(); view.tEnd = cur.summary.t1; }
    requestDraw();
  }
  function setEnd(tEnd) {
    const s = cur.summary;
    view.tEnd = Math.min(s.t1, Math.max(Math.min(s.t0 + view.W, s.t1), tEnd));
    requestDraw();
  }
  // Change the window, keeping the time `about` (default: the centre) where it is on screen.
  function setWindow(W, about) {
    const old = view.W;
    if (!Number.isFinite(about)) about = view.tEnd - old / 2;
    view.W = W;
    setEnd(about + (view.tEnd - about) / old * W);
  }
  function zoom(dir, about) {
    if (!cur) return;
    let W;
    if (dir < 0) W = ZOOMS.slice().reverse().find((z) => z < view.W * 0.999);
    else { W = ZOOMS.find((z) => z > view.W * 1.001); if (W === undefined || W > runSpan()) W = runSpan(); }
    if (W === undefined || W === view.W) return;
    setWindow(W, about);
  }
  // ── the value axes ──
  // Zoom a strip's value range by `factor` (above 1: in) about the value at `frac` of the plot
  // height from the bottom (default: the middle); from the fitted range when it was automatic.
  function yZoom(key, factor, frac) {
    const f = frames[key];
    if (!f) return;
    const [lo, hi] = yz[key] || [f.lo, f.hi];
    const v = lo + (hi - lo) * (Number.isFinite(frac) ? Math.min(1, Math.max(0, frac)) : 0.5);
    const nlo = v - (v - lo) / factor, nhi = v + (hi - v) / factor;
    if (!(nhi - nlo > 1e-9) || nhi - nlo > 1e9) return;
    if (!yz[key]) yBase[key] = hi - lo;
    yz[key] = [nlo, nhi];
    requestDraw();
  }
  function yAuto(key) { yz[key] = yBase[key] = null; requestDraw(); }
  // A zoomed strip grows with its magnification: 1.5× the height per halving of the fitted span,
  // up to three times, so zooming in buys pixels and not only a narrower range.
  function stripHeight(key) {
    const r = yz[key], base = yBase[key];
    if (!r || !base) return STRIP_HEIGHT;
    const ratio = base / Math.max(1e-12, r[1] - r[0]);
    return Math.round(STRIP_HEIGHT * Math.min(3, Math.max(1, 1 + 0.5 * Math.log2(ratio))));
  }
  function applyHeights() {
    for (const key of Object.keys(STRIPS)) {
      const cv = $(STRIPS[key]), h = `${stripHeight(key)}px`;
      if (cv.style.height !== h) cv.style.height = h;
    }
  }
  const yUnit = (key) => (key === "tau" || (key === "err" && errSig === "tau") ? " N·m" : "°");
  function showYSpans() {
    for (const key of Object.keys(STRIPS)) {
      const r = yz[key];
      $(`${STRIPS[key]}-yspan`).textContent = r ? `${scopeDraw.fmtTick(r[1] - r[0])}${yUnit(key)} span` : "";
      $(`${STRIPS[key]}-yauto`).disabled = !r;
    }
  }
  function setErrSignal(v) {
    errSig = v === "tau" ? "tau" : "pos";
    $("plot-err-signal").value = errSig;
    $("plot-err-units").textContent = errSig === "tau" ? "N·m reported − commanded" : "° measured − commanded";
    yz.err = yBase.err = null;
    try { localStorage.setItem("plotErrSignal", errSig); } catch (_) { /* storage blocked */ }
    requestDraw();
  }
  // Time under the pointer on a strip (the plot area spans PLOT_LEFT … width − PLOT_RIGHT).
  function timeAt(canvas, x) {
    const x0 = scopeDraw.PLOT_LEFT, x1 = canvas.clientWidth - scopeDraw.PLOT_RIGHT;
    return view.tEnd - view.W * (x1 - x) / Math.max(1, x1 - x0);
  }
  function wireStrip(canvas, key) {
    let drag = null;
    canvas.addEventListener("pointerdown", (ev) => {
      if (!cur) return;
      // A plain drag moves the window (and, on a zoomed value axis, the range); with Shift (and a
      // comparison on screen) it slides B instead.
      drag = { x: ev.clientX, y: ev.clientY, end: view.tEnd, offset: cmp.offset, shift: !!(ev.shiftKey && cmpEntry()), range: yz[key] ? yz[key].slice() : null,
        px: Math.max(1, canvas.clientWidth - scopeDraw.PLOT_LEFT - scopeDraw.PLOT_RIGHT), moved: false };
      canvas.setPointerCapture(ev.pointerId);
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!cur) return;
      if (drag) {
        if (Math.abs(ev.clientX - drag.x) > 2 || Math.abs(ev.clientY - drag.y) > 2) drag.moved = true;
        if (drag.moved) {
          cursorT = null;
          const dt = (ev.clientX - drag.x) / drag.px * view.W;
          if (drag.shift) { setOffset(drag.offset + dt); return; }
          const f = frames[key];
          if (drag.range && f) {                                  // the trace follows the pointer up and down too
            const dv = (ev.clientY - drag.y) / Math.max(1, f.y1 - f.y0) * (drag.range[1] - drag.range[0]);
            yz[key] = [drag.range[0] + dv, drag.range[1] + dv];
          }
          setEnd(drag.end - dt);
          return;
        }
      }
      cursorT = timeAt(canvas, ev.offsetX);
      requestDraw();
    });
    const release = () => { drag = null; };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointerleave", () => { if (cursorT !== null) { cursorT = null; requestDraw(); } });
    canvas.addEventListener("wheel", (ev) => {
      const d = ev.deltaY || ev.deltaX;                           // with Shift some browsers report the wheel as horizontal
      if (ev.ctrlKey || ev.metaKey) { ev.preventDefault(); zoom(d > 0 ? +1 : -1, timeAt(canvas, ev.offsetX)); }
      else if (ev.shiftKey) {
        ev.preventDefault();
        const f = frames[key];
        yZoom(key, d > 0 ? 1 / 1.5 : 1.5, f ? (f.y1 - ev.offsetY) / Math.max(1, f.y1 - f.y0) : 0.5);
      }                                                           // a plain wheel scrolls the page
    }, { passive: false });
  }
  function wireOverview(canvas) {
    let down = false;
    const jump = (ev) => {
      const s = cur.summary, t = s.t0 + runSpan() * ev.offsetX / Math.max(1, canvas.clientWidth);
      setEnd(t + view.W / 2);
    };
    canvas.addEventListener("pointerdown", (ev) => { if (!cur) return; down = true; canvas.setPointerCapture(ev.pointerId); jump(ev); });
    canvas.addEventListener("pointermove", (ev) => { if (down && cur) jump(ev); });
    const release = () => { down = false; };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
  }

  // ── spectrum strip ──
  function fftShow(on) {
    fft.on = on;
    $("plot-fft-strip").classList.toggle("hidden", !on);
    $("plot-fft-toggle").classList.toggle("on", on);
    try { localStorage.setItem("plotFft", on ? "1" : "0"); } catch (_) { /* storage blocked */ }
    requestDraw();
  }
  function fftZoom(dir) {
    const i = Math.min(FFT_FMAX.length - 1, Math.max(0, FFT_FMAX.indexOf(fft.fmax) + dir));
    fft.fmax = FFT_FMAX[i];
    try { localStorage.setItem("plotFftFmax", String(fft.fmax)); } catch (_) { /* storage blocked */ }
    $("plot-fft-zoom-in").disabled = i === 0;
    $("plot-fft-zoom-out").disabled = i === FFT_FMAX.length - 1;
    requestDraw();
  }
  function drawSpectrum(run, i0, i1, t, accent, bs) {
    const unit = FFT_UNITS[fft.signal], welchOpts = { segmentSeconds: FFT_SEGMENT, maxSegments: 32, gap: GAP };
    const signal = (r) => (fft.signal === "err" ? r.err_deg : fft.signal === "q" ? r.q_deg : fft.signal === "q_cmd" ? r.q_cmd_deg : r[fft.signal]);
    $("plot-fft-units").textContent = `amplitude, dB re 1 ${unit}`;
    const key = `${cur.id}|${fft.signal}|${i0}|${i1}`;
    if (key !== fft.key) { fft.spec = scopeDSP.welch(signal(run).subarray(i0, i1), t, welchOpts); fft.key = key; }
    const keyB = bs ? `${cmp.id}|${fft.signal}|${bs.j0}|${bs.j1}` : "";
    if (keyB !== fft.keyB) { fft.specB = bs ? scopeDSP.welch(signal(bs.rb).subarray(bs.j0, bs.j1), bs.t, welchOpts) : null; fft.keyB = keyB; }
    const empty = "fewer than 64 gap-free samples on screen";
    let line = scopeDraw.drawSpectrumStrip($("plot-fft"), fft.spec, { unit, fmax: fft.fmax, accent, empty, others: bs ? [{ spec: fft.specB, color: accent, dash: DASH, alpha: B_ALPHA }] : [] });
    if (bs) line = `A  ${line || empty}     B  ${fft.specB ? scopeDraw.spectrumReadout(fft.specB, unit, Math.min(fft.fmax, fft.specB.fs / 2)) : empty}`;
    $("plot-fft-readout").textContent = line;
  }

  // ── drawing ──────────────────────────────────────────────────────────────
  function requestDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => { drawPending = false; draw(); });
  }
  function drawEmpty(canvas, msg) {
    const c = scopeDraw.setupCanvas(canvas);
    if (!c) return;
    c.ctx.fillStyle = c.muted; c.ctx.textAlign = "center"; c.ctx.textBaseline = "middle";
    c.ctx.fillText(msg, c.W / 2, c.H / 2);
  }
  const fmtWindow = (W) => (W < 1 ? `${Math.round(W * 1000)} ms` : W < 120 ? `${+W.toFixed(2)} s` : `${+(W / 60).toFixed(1)} min`);
  function draw() {
    const css = getComputedStyle($("scope-card"));
    const blue = css.getPropertyValue("--scope-cmd").trim() || "#3b82f6", amber = css.getPropertyValue("--scope-meas").trim() || "#d95926", accent = css.getPropertyValue("--scope-err").trim() || "#6ea8fe";
    const green = css.getPropertyValue("--scope-ff").trim() || "#199e70";
    applyHeights();
    $("plot-zoom-in").disabled = !cur || view.W <= ZOOMS[0] + 1e-9;
    $("plot-zoom-out").disabled = !cur || view.W >= runSpan() - 1e-9;
    $("plot-fit").disabled = !cur;
    if (!cur) {
      $("plot-window").textContent = "no file";
      ["plot-overview", "plot-pos", "plot-err", "plot-tau"].forEach((id) => drawEmpty($(id), "open a recording"));
      for (const key of Object.keys(frames)) frames[key] = null;
      showYSpans();
      if (fft.on) $("plot-fft-readout").textContent = scopeDraw.drawSpectrumStrip($("plot-fft"), null, { empty: "open a recording" });
      $("plot-readout").textContent = "";
      return;
    }
    const run = cur.run, W = view.W, tEnd = view.tEnd;
    $("plot-window").textContent = `${fmtWindow(W)} window`;
    // On screen: from one sample before the left edge to the last one at or before the right edge.
    const i1 = scopeDraw.upperBound(run.t, tEnd);
    const i0 = Math.max(0, scopeDraw.lowerBound(run.t, tEnd - W) - 1);
    const t = run.t.subarray(i0, i1);
    const axis = { tEnd, W, t, absolute: true };
    // The comparison, B: its samples that fall in the window once shifted onto A's clock, drawn
    // first (dashed, lighter) so A stays on top where the two coincide.
    const B = cmpEntry();
    let bs = null;
    if (B) {
      const rb = B.run, sh = shiftOf(B);
      const j1 = scopeDraw.upperBound(rb.t, tEnd - sh), j0 = Math.max(0, scopeDraw.lowerBound(rb.t, tEnd - W - sh) - 1);
      bs = { rb, j0, j1, sh, t: rb.t.subarray(j0, j1) };
    }
    const bSeries = (col, color, key) => (bs ? [{ key, values: bs.rb[col].subarray(bs.j0, bs.j1), color, t: bs.t, tOffset: bs.sh, dash: DASH, alpha: B_ALPHA }] : []);
    const aSeries = (col, color, key) => ({ key, values: run[col].subarray(i0, i1), color });
    const shown = (list) => list.filter((s) => !hidden.has(s.key));
    const errCol = errSig === "tau" ? "tau_err" : "err_deg";
    frames.pos = scopeDraw.drawStrip($("plot-pos"), axis, shown(bSeries("q_cmd_deg", blue, "b_q_cmd").concat(bSeries("q_deg", amber, "b_q"), [aSeries("q_cmd_deg", blue, "q_cmd"), aSeries("q_deg", amber, "q")])), { minSpan: 0.2, cursor: cursorT, range: yz.pos });
    // Without a comparison the error strip has one trace and no legend, so it is always drawn.
    const errSeries = bs ? shown(bSeries(errCol, accent, "b_err").concat([aSeries(errCol, accent, "err")])) : [aSeries(errCol, accent, "err")];
    frames.err = scopeDraw.drawStrip($("plot-err"), axis, errSeries, { symmetric: true, minSpan: errSig === "tau" ? 0.05 : 0.1, cursor: cursorT, range: yz.err });
    frames.tau = scopeDraw.drawStrip($("plot-tau"), axis, shown(bSeries("tau_ff", green, "b_tau_ff").concat(bSeries("tau_cmd", blue, "b_tau_cmd"), bSeries("tau_meas", amber, "b_tau_meas"), [aSeries("tau_ff", green, "tau_ff"), aSeries("tau_cmd", blue, "tau_cmd"), aSeries("tau_meas", amber, "tau_meas")])), { symmetric: true, minSpan: 0.5, cursor: cursorT, range: yz.tau });
    showYSpans();
    drawOverview(accent);
    if (fft.on) drawSpectrum(run, i0, i1, t, accent, bs);
    drawReadout(run, i0, i1, bs);
  }
  // The whole run's tracking error, drawn once per run and canvas size, with the window outlined.
  function drawOverview(accent) {
    const cv = $("plot-overview"), dpr = window.devicePixelRatio || 1;
    const run = cur.run, s = cur.summary, span = runSpan();
    const key = `${cur.id}|${cv.clientWidth}|${cv.clientHeight}|${dpr}`;
    if (ov.key !== key) {
      scopeDraw.drawStrip(cv, { tEnd: s.t0 + span, W: span, t: run.t, absolute: true }, [{ values: run.err_deg, color: accent }], { bare: true, symmetric: true, minSpan: 0.1 });
      if (!cv.width) return;                                      // hidden
      ov.cache = document.createElement("canvas"); ov.cache.width = cv.width; ov.cache.height = cv.height;
      ov.cache.getContext("2d").drawImage(cv, 0, 0);
      ov.key = key;
    } else {
      const c = scopeDraw.setupCanvas(cv);
      if (!c) return;
      c.ctx.setTransform(1, 0, 0, 1, 0, 0); c.ctx.drawImage(ov.cache, 0, 0);
    }
    const ctx = cv.getContext("2d"), Wpx = cv.clientWidth, H = cv.clientHeight;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = (tt) => Wpx * (tt - s.t0) / span;
    const xa = Math.max(0, X(view.tEnd - view.W)), xb = Math.min(Wpx, X(view.tEnd));
    ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
    ctx.fillRect(0, 0, xa, H); ctx.fillRect(xb, 0, Wpx - xb, H);
    ctx.strokeStyle = accent; ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(xa) + 0.5, 0.5, Math.max(1, Math.round(xb - xa)) - 1, H - 1);
  }
  function drawReadout(run, i0, i1, bs) {
    const ro = $("plot-readout");
    // The sample of r nearest to tc (its own clock), or none when the nearest is more than a hole away.
    const sampleLine = (r, tc, tag) => {
      let i = scopeDraw.lowerBound(r.t, tc);
      if (i > 0 && (i >= r.t.length || r.t[i] - tc > tc - r.t[i - 1])) i--;
      if (!r.t.length || Math.abs(r.t[i] - tc) > GAP) return `${tag}no sample here`;
      const tm = r.tau_meas[i];
      return `${tag}t ${r.t[i].toFixed(4)} s   commanded ${fmt(r.q_cmd_deg[i], 3)}°   measured ${fmt(r.q_deg[i], 3)}°   error ${fmt(r.err_deg[i], 3)}°   torque ${fmt(r.tau_cmd[i], 3)} N·m`
        + (Number.isFinite(tm) ? ` (motor ${fmt(tm, 3)}, error ${fmt(r.tau_err[i], 3)})` : "")
        + (Number.isFinite(r.tau_ff[i]) ? `   feed-forward ${fmt(r.tau_ff[i], 3)} (g ${fmt(r.tau_g[i], 3)}, f ${fmt(r.tau_f[i], 3)}, i ${fmt(r.tau_i[i], 3)})` : "");
    };
    const statsLine = (r, a, b, tag) => {
      const s = scopeRecord.summarize(r, a, b);
      return `${tag}${fmtN(s.samples)} sample${s.samples === 1 ? "" : "s"} on screen   max |error| ${fmt(s.errMax, 3)}° at ${fmt(s.errMaxAt, 2)} s   rms ${fmt(s.errRms, 3)}°   peak |torque| ${fmt(s.tauMax, 2)} N·m`
        + (errSig === "tau" ? `   max |torque error| ${fmt(s.tauErrMax, 3)} N·m at ${fmt(s.tauErrAt, 2)} s   rms ${fmt(s.tauErrRms, 3)} N·m` : "")
        + (s.gaps ? `   ${s.gaps} hole${s.gaps === 1 ? "" : "s"}` : "") + (s.missing ? `   ${fmtN(s.missing)} without a measurement` : "");
    };
    const tagA = bs ? "A  " : "";
    if (cursorT !== null && run.t.length) {
      ro.textContent = sampleLine(run, cursorT, tagA) + (bs ? "\n" + sampleLine(bs.rb, cursorT - bs.sh, "B  ") : "");
      return;
    }
    ro.textContent = statsLine(run, i0, i1, tagA) + (bs ? "\n" + statsLine(bs.rb, bs.j0, bs.j1, "B  ") : "");
  }

  // ── wiring ───────────────────────────────────────────────────────────────
  function init() {
    $("plot-file").onchange = (ev) => { openFiles(ev.target.files); ev.target.value = ""; };
    $("plot-select").onchange = (ev) => { if (ev.target.value !== "") select(Number(ev.target.value)); };
    $("plot-compare").onchange = (ev) => compare(ev.target.value === "" ? null : Number(ev.target.value));
    $("plot-offset").oninput = (ev) => setOffset(Number(ev.target.value));
    $("plot-offset").onchange = (ev) => { ev.target.value = cmp.offset.toFixed(3); };
    $("plot-align").onclick = () => setOffset(0);
    $("plot-zoom-in").onclick = () => zoom(-1);
    $("plot-zoom-out").onclick = () => zoom(+1);
    $("plot-fit").onclick = fit;
    $("plot-fft-toggle").onclick = () => fftShow(!fft.on);
    $("plot-fft-zoom-in").onclick = () => fftZoom(-1);
    $("plot-fft-zoom-out").onclick = () => fftZoom(+1);
    $("plot-fft").addEventListener("wheel", (ev) => { if (!(ev.ctrlKey || ev.metaKey)) return; ev.preventDefault(); fftZoom(ev.deltaY > 0 ? +1 : -1); }, { passive: false });
    for (const key of Object.keys(STRIPS)) {
      wireStrip($(STRIPS[key]), key);
      $(`${STRIPS[key]}-yin`).onclick = () => yZoom(key, 2);
      $(`${STRIPS[key]}-yout`).onclick = () => yZoom(key, 0.5);
      $(`${STRIPS[key]}-yauto`).onclick = () => yAuto(key);
    }
    $("plot-err-signal").onchange = (ev) => setErrSignal(ev.target.value);
    document.querySelectorAll("#scope-card input[data-trace]").forEach((cb) => {
      cb.onchange = () => { if (cb.checked) hidden.delete(cb.dataset.trace); else hidden.add(cb.dataset.trace); cb.closest(".li").classList.toggle("off", !cb.checked); requestDraw(); };
    });
    try { setErrSignal(localStorage.getItem("plotErrSignal") || "pos"); } catch (_) { setErrSignal("pos"); }
    wireOverview($("plot-overview"));
    try {
      fft.on = localStorage.getItem("plotFft") === "1";
      const sig = localStorage.getItem("plotFftSignal"), fm = Number(localStorage.getItem("plotFftFmax"));
      if (FFT_UNITS[sig]) fft.signal = sig;
      if (FFT_FMAX.includes(fm)) fft.fmax = fm;
    } catch (_) { /* storage blocked */ }
    $("plot-fft-signal").value = fft.signal;
    $("plot-fft-signal").onchange = (ev) => { fft.signal = ev.target.value; try { localStorage.setItem("plotFftSignal", fft.signal); } catch (_) { /* storage blocked */ } requestDraw(); };
    fftZoom(0);
    fftShow(fft.on);
    // Drop a file anywhere on the page.
    const drop = $("plot-drop");
    document.addEventListener("dragover", (ev) => { ev.preventDefault(); drop.classList.add("over"); });
    document.addEventListener("dragleave", (ev) => { if (!ev.relatedTarget) drop.classList.remove("over"); });
    document.addEventListener("drop", (ev) => { ev.preventDefault(); drop.classList.remove("over"); openFiles(ev.dataTransfer.files); });
    window.addEventListener("keydown", (ev) => {
      const tag = ev.target && ev.target.tagName;
      if (!cur || tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (ev.key === "ArrowLeft") setEnd(view.tEnd - view.W / 2);
      else if (ev.key === "ArrowRight") setEnd(view.tEnd + view.W / 2);
      else if (ev.key === "-" || ev.key === "_") zoom(+1);
      else if (ev.key === "+" || ev.key === "=") zoom(-1);
      else return;
      ev.preventDefault();
    });
    window.addEventListener("resize", requestDraw);
    requestDraw();
  }
  init();
})();
