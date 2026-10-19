/* Offline plotter (plot.html): the pendant's scope strips for a CSV recording, read in the browser.
   Drawing is shared with the live scope (scope.js, dsp.js); the file is parsed by record.js. */
(() => {
  const $ = (id) => document.getElementById(id);
  const ZOOMS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 60, 120, 300, 600, 1800, 3600];   // s, the zoom steps (Fit goes beyond)
  const FFT_FMAX = [5, 10, 25, 50, 100, 250];                     // Hz, the spectrum's frequency-range steps (clipped to fs/2)
  const FFT_UNITS = { err: "°", q: "°", q_cmd: "°", tau_cmd: "N·m", tau_meas: "N·m" };
  const FFT_SEGMENT = 2.0;                                        // s per Welch segment, as in the live scope
  const GAP = scopeDraw.GAP;
  const fmt = (v, d = 1) => (v === null || v === undefined || !Number.isFinite(v)) ? "—" : Number(v).toFixed(d);
  const fmtN = (n) => n.toLocaleString("en-US");

  const entries = [];                      // every run of every loaded file
  let cur = null;                          // the entry on screen
  const view = { W: 5, tEnd: 0 };          // the window (s) and the time at its right edge, in the run's clock
  const fft = { on: false, signal: "err", fmax: 50, spec: null, key: "" };
  const ov = { key: "", cache: null };     // the overview trace, cached per run and canvas size
  let cursorT = null;                      // the time under the pointer, or null
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
    const sel = $("plot-select");
    sel.innerHTML = "";
    if (!entries.length) { const o = document.createElement("option"); o.value = ""; o.textContent = "none loaded"; sel.appendChild(o); }
    for (const e of entries) { const o = document.createElement("option"); o.value = String(e.id); o.textContent = e.label; sel.appendChild(o); }
    if (last) select(last.id);
  }
  function select(id) {
    cur = entries[id] || null;
    $("plot-select").value = cur ? String(cur.id) : "";
    fft.key = ""; ov.key = ""; cursorT = null;
    const s = $("plot-summary");
    if (!cur) { s.textContent = ""; fit(); return; }
    const m = cur.summary, parts = [];
    if (cur.joint) parts.push(cur.joint);
    if (cur.nRuns > 1) parts.push(`run ${cur.run.run} of ${cur.nRuns}`);
    parts.push(`${fmt(m.duration, 1)} s`, `${fmtN(m.samples)} samples at ${fmt(m.rate, 0)} Hz`);
    if (m.gaps) parts.push(`${m.gaps} hole${m.gaps === 1 ? "" : "s"}`);
    if (m.missing) parts.push(`${fmtN(m.missing)} without a measurement`);
    parts.push(`max |error| ${fmt(m.errMax, 3)}° at ${fmt(m.errMaxAt, 2)} s`);
    s.textContent = parts.concat(cur.problems).join(" · ");
    fit();
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
  // Time under the pointer on a strip (the plot area spans PLOT_LEFT … width − PLOT_RIGHT).
  function timeAt(canvas, x) {
    const x0 = scopeDraw.PLOT_LEFT, x1 = canvas.clientWidth - scopeDraw.PLOT_RIGHT;
    return view.tEnd - view.W * (x1 - x) / Math.max(1, x1 - x0);
  }
  function wireStrip(canvas) {
    let drag = null;
    canvas.addEventListener("pointerdown", (ev) => {
      if (!cur) return;
      drag = { x: ev.clientX, end: view.tEnd, px: Math.max(1, canvas.clientWidth - scopeDraw.PLOT_LEFT - scopeDraw.PLOT_RIGHT), moved: false };
      canvas.setPointerCapture(ev.pointerId);
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!cur) return;
      if (drag) {
        if (Math.abs(ev.clientX - drag.x) > 2) drag.moved = true;
        if (drag.moved) { cursorT = null; setEnd(drag.end - (ev.clientX - drag.x) / drag.px * view.W); return; }
      }
      cursorT = timeAt(canvas, ev.offsetX);
      requestDraw();
    });
    const release = () => { drag = null; };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointerleave", () => { if (cursorT !== null) { cursorT = null; requestDraw(); } });
    canvas.addEventListener("wheel", (ev) => {
      if (!(ev.ctrlKey || ev.metaKey)) return;                    // a plain wheel scrolls the page
      ev.preventDefault();
      zoom(ev.deltaY > 0 ? +1 : -1, timeAt(canvas, ev.offsetX));
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
  function drawSpectrum(run, i0, i1, t, accent) {
    const unit = FFT_UNITS[fft.signal];
    $("plot-fft-units").textContent = `amplitude, dB re 1 ${unit}`;
    const key = `${cur.id}|${fft.signal}|${i0}|${i1}`;
    if (key !== fft.key) {
      const src = fft.signal === "err" ? run.err_deg : fft.signal === "q" ? run.q_deg : fft.signal === "q_cmd" ? run.q_cmd_deg : run[fft.signal];
      fft.spec = scopeDSP.welch(src.subarray(i0, i1), t, { segmentSeconds: FFT_SEGMENT, maxSegments: 32, gap: GAP });
      fft.key = key;
    }
    $("plot-fft-readout").textContent = scopeDraw.drawSpectrumStrip($("plot-fft"), fft.spec, { unit, fmax: fft.fmax, accent, empty: "fewer than 64 gap-free samples on screen" });
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
    $("plot-zoom-in").disabled = !cur || view.W <= ZOOMS[0] + 1e-9;
    $("plot-zoom-out").disabled = !cur || view.W >= runSpan() - 1e-9;
    $("plot-fit").disabled = !cur;
    if (!cur) {
      $("plot-window").textContent = "no file";
      ["plot-overview", "plot-pos", "plot-err", "plot-tau"].forEach((id) => drawEmpty($(id), "open a recording"));
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
    scopeDraw.drawStrip($("plot-pos"), axis, [{ values: run.q_cmd_deg.subarray(i0, i1), color: blue }, { values: run.q_deg.subarray(i0, i1), color: amber }], { minSpan: 0.2, cursor: cursorT });
    scopeDraw.drawStrip($("plot-err"), axis, [{ values: run.err_deg.subarray(i0, i1), color: accent }], { symmetric: true, minSpan: 0.1, cursor: cursorT });
    scopeDraw.drawStrip($("plot-tau"), axis, [{ values: run.tau_cmd.subarray(i0, i1), color: blue }, { values: run.tau_meas.subarray(i0, i1), color: amber }], { symmetric: true, minSpan: 0.5, cursor: cursorT });
    drawOverview(accent);
    if (fft.on) drawSpectrum(run, i0, i1, t, accent);
    drawReadout(run, i0, i1);
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
  function drawReadout(run, i0, i1) {
    const ro = $("plot-readout");
    if (cursorT !== null && run.t.length) {
      let i = scopeDraw.lowerBound(run.t, cursorT);
      if (i > 0 && (i >= run.t.length || run.t[i] - cursorT > cursorT - run.t[i - 1])) i--;
      const tm = run.tau_meas[i];
      ro.textContent = `t ${run.t[i].toFixed(4)} s   commanded ${fmt(run.q_cmd_deg[i], 3)}°   measured ${fmt(run.q_deg[i], 3)}°   error ${fmt(run.err_deg[i], 3)}°   torque ${fmt(run.tau_cmd[i], 3)} N·m`
        + (Number.isFinite(tm) ? ` (motor ${fmt(tm, 3)})` : "");
      return;
    }
    const s = scopeRecord.summarize(run, i0, i1);
    ro.textContent = `${fmtN(s.samples)} samples on screen   max |error| ${fmt(s.errMax, 3)}° at ${fmt(s.errMaxAt, 2)} s   rms ${fmt(s.errRms, 3)}°   peak |torque| ${fmt(s.tauMax, 2)} N·m`
      + (s.gaps ? `   ${s.gaps} hole${s.gaps === 1 ? "" : "s"}` : "") + (s.missing ? `   ${fmtN(s.missing)} without a measurement` : "");
  }

  // ── wiring ───────────────────────────────────────────────────────────────
  function init() {
    $("plot-file").onchange = (ev) => { openFiles(ev.target.files); ev.target.value = ""; };
    $("plot-select").onchange = (ev) => { if (ev.target.value !== "") select(Number(ev.target.value)); };
    $("plot-zoom-in").onclick = () => zoom(-1);
    $("plot-zoom-out").onclick = () => zoom(+1);
    $("plot-fit").onclick = fit;
    $("plot-fft-toggle").onclick = () => fftShow(!fft.on);
    $("plot-fft-zoom-in").onclick = () => fftZoom(-1);
    $("plot-fft-zoom-out").onclick = () => fftZoom(+1);
    $("plot-fft").addEventListener("wheel", (ev) => { if (!(ev.ctrlKey || ev.metaKey)) return; ev.preventDefault(); fftZoom(ev.deltaY > 0 ? +1 : -1); }, { passive: false });
    ["plot-pos", "plot-err", "plot-tau"].forEach((id) => wireStrip($(id)));
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
