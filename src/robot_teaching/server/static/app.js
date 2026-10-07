/* reBot Arm teach pendant — vanilla JS client for the robot_teaching API. */
(() => {
  const $ = (id) => document.getElementById(id);
  const RAD = Math.PI / 180;
  const deg = (r) => (r / RAD);
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);

  let config = null;
  let program = null;
  let state = null;
  let programRev = -1;
  let jointStep = 1.0, cartStepLin = 0.005, cartStepAng = 2.0, cartFrame = "base";
  let sliderBusy = false;
  let lastProgError = "";
  let poseFrame = "home";                       // "home": relative to the home position; "base": raw base frame
  try { poseFrame = localStorage.getItem("poseFrame") || "home"; } catch (_) { /* storage blocked */ }
  let wsRef = null;                             // the open state websocket (the scope sends its joint selection on it)

  // ── API ────────────────────────────────────────────────────────────────
  async function api(method, path, body) {
    const res = await fetch(path, {
      method, headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let data = null;
    try { data = await res.json(); } catch (_) { /* no body */ }
    if (!res.ok) {
      const msg = (data && (data.detail || data.message)) || `${res.status} ${res.statusText}`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }
  async function act(method, path, body, okMsg) {
    try {
      const r = await api(method, path, body);
      if (okMsg !== false) toast(okMsg || (r && r.message) || "ok");
      return r;
    } catch (e) { toast(e.message, true); return null; }
  }

  let toastTimer = null;
  function toast(msg, isErr = false) {
    const t = $("toast");
    t.textContent = msg; t.className = "toast" + (isErr ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.add("hidden"), isErr ? 5000 : 2200);
  }

  // ── build static widgets ───────────────────────────────────────────────
  function segButtons(el, values, label, current, onPick) {
    el.innerHTML = "";
    values.forEach((v) => {
      const b = document.createElement("button");
      b.textContent = label(v);
      if (Math.abs(v - current) < 1e-12) b.classList.add("on");
      b.onclick = () => { onPick(v); [...el.children].forEach((c) => c.classList.toggle("on", c === b)); };
      el.appendChild(b);
    });
  }

  function buildJog() {
    const jt = $("joint-table").querySelector("tbody");
    jt.innerHTML = "";
    config.joint_names.forEach((name, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="name">${name}</td><td id="jq${i}">—</td><td id="jt${i}">—</td><td id="jv${i}">—</td><td id="jg${i}">—</td>`;
      jt.appendChild(tr);
    });
    segButtons($("joint-step"), config.jog.joint_steps_deg, (v) => `${v}°`, jointStep, (v) => (jointStep = v));
    const jj = $("joint-jog"); jj.innerHTML = "";
    config.joint_names.forEach((name, i) => {
      const d = document.createElement("div"); d.className = "jog";
      d.innerHTML = `<span class="lbl">J${i + 1}</span><button class="btn">−</button><button class="btn">+</button>`;
      const [minus, plus] = d.querySelectorAll("button");
      minus.onclick = () => act("POST", "/api/jog/joint", { joint: i, delta: -jointStep * RAD }, false);
      plus.onclick = () => act("POST", "/api/jog/joint", { joint: i, delta: jointStep * RAD }, false);
      jj.appendChild(d);
    });
    segButtons($("cart-step-lin"), config.jog.cartesian_steps_m, (v) => `${(v * 1000).toFixed(v < 0.001 ? 1 : 0)} mm`, cartStepLin, (v) => (cartStepLin = v));
    segButtons($("cart-step-ang"), config.jog.cartesian_steps_deg, (v) => `${v}°`, cartStepAng, (v) => (cartStepAng = v));
    [...$("cart-frame").children].forEach((b) => (b.onclick = () => {
      cartFrame = b.dataset.frame; [...$("cart-frame").children].forEach((c) => c.classList.toggle("on", c === b));
    }));
    const cj = $("cart-jog"); cj.innerHTML = "";
    [["x", "X"], ["y", "Y"], ["z", "Z"], ["roll", "Roll"], ["pitch", "Pitch"], ["yaw", "Yaw"]].forEach(([axis, label]) => {
      const d = document.createElement("div"); d.className = "jog";
      d.innerHTML = `<span class="lbl">${label}</span><button class="btn">−</button><button class="btn">+</button>`;
      const [minus, plus] = d.querySelectorAll("button");
      const step = () => (["x", "y", "z"].includes(axis) ? cartStepLin : cartStepAng * RAD);
      minus.onclick = () => act("POST", "/api/jog/cartesian", { axis, delta: -step(), frame: cartFrame }, false);
      plus.onclick = () => act("POST", "/api/jog/cartesian", { axis, delta: step(), frame: cartFrame }, false);
      cj.appendChild(d);
    });
    // The slider is "percent open": left end = closed position, right end = open position,
    // whatever the motor angles are, so it always moves toward the button that was pressed.
    const s = $("grip-slider");
    s.oninput = () => { sliderBusy = true; $("grip-slider-val").textContent = gripLabel(s.value / 100); };
    s.onchange = () => { act("POST", "/api/gripper", { position: gripPosition(s.value / 100) }, false); sliderBusy = false; };
    $("btn-grip-open").onclick = () => act("POST", "/api/gripper", { action: "open" }, "gripper opening");
    $("btn-grip-close").onclick = () => act("POST", "/api/gripper", { action: "close" }, "gripper closing");
    $("btn-grip-hand").onclick = () => act("POST", "/api/gripper", { action: "hand" }, "gripper free: move it by hand");
    if (!config.gripper.has_gripper) document.querySelector(".gripper-row").classList.add("hidden");
  }

  // ── gripper helpers (0 = closed, 1 = open) ─────────────────────────────
  const gripPosition = (frac) => config.gripper.closed_position + frac * (config.gripper.open_position - config.gripper.closed_position);
  function gripFraction(pos) {
    const span = config.gripper.open_position - config.gripper.closed_position;
    if (Math.abs(span) < 1e-9) return 0;
    return Math.min(1, Math.max(0, (pos - config.gripper.closed_position) / span));
  }
  const gripLabel = (frac) => `${Math.round(frac * 100)} % open (${fmt(gripPosition(frac), 2)} rad)`;

  // ── state rendering ────────────────────────────────────────────────────
  function renderState() {
    if (!state || !config) return;
    const badge = $("mode-badge");
    badge.textContent = state.mode.replace("_", " ");
    badge.className = "badge " + state.mode;
    const fd = $("fd-status");
    fd.textContent = state.mode === "free_drive" ? state.free_drive_status : (state.jogging ? "jogging" : (state.gains_settled ? "" : "settling"));
    fd.classList.toggle("hidden", !fd.textContent);

    state.q.forEach((q, i) => {
      $(`jq${i}`).textContent = fmt(deg(q), 2);
      $(`jt${i}`).textContent = fmt(deg(state.q_target[i]), 2);
      $(`jv${i}`).textContent = fmt(deg(state.qd[i]), 1);
      $(`jg${i}`).textContent = fmt(state.tau_g[i], 2);
    });
    const shown = poseFrame === "home" ? state.pose_home : state.pose;
    if (shown) {
      const [x, y, z] = shown.xyz, [r, p, w] = shown.rpy;
      $("pose-x").textContent = fmt(x * 1000); $("pose-y").textContent = fmt(y * 1000); $("pose-z").textContent = fmt(z * 1000);
      $("pose-r").textContent = fmt(deg(r)); $("pose-p").textContent = fmt(deg(p)); $("pose-w").textContent = fmt(deg(w));
    }
    $("ee-speed").textContent = `${fmt(state.ee_speed[0] * 1000, 0)} mm/s, ${fmt(deg(state.ee_speed[1]), 0)} °/s`;
    $("gripper-pos").textContent = fmt(state.gripper, 2);
    $("gripper-target").textContent = fmt(state.gripper_target, 2);
    $("tick-dt").textContent = `${fmt(state.tick_dt * 1000, 2)} ms`;
    if (!sliderBusy && state.gripper_target !== null) {
      const frac = gripFraction(state.gripper_target);
      $("grip-slider").value = Math.round(frac * 100);
      $("grip-slider-val").textContent = gripLabel(frac);
    }

    const disc = !state.connected;
    const hold = state.mode === "hold", free = state.mode === "free_drive", play = state.mode === "playback", off = state.mode === "disabled";
    $("btn-connect").classList.toggle("hidden", !disc);
    $("btn-disconnect").classList.toggle("hidden", disc);
    $("btn-disconnect").disabled = play;
    $("jog-controls").classList.toggle("off", disc || !hold || !state.gains_settled);
    document.querySelector(".gripper-row").classList.toggle("off", disc || off || play);
    $("btn-grip-hand").classList.toggle("on", !!state.gripper_hand);
    $("btn-grip-hand").textContent = state.gripper_hand ? "Hand ✓" : "Hand";
    $("btn-free").disabled = disc || free || play || off;
    $("btn-hold").disabled = disc || hold || off;
    $("btn-stop").disabled = disc || off;
    $("btn-enable").classList.toggle("hidden", disc || !off);
    $("btn-estop").classList.toggle("hidden", !disc && off);
    $("btn-estop").disabled = disc;
    $("btn-play").disabled = disc || !hold || !state.gains_settled || state.jogging;
    $("btn-plan").disabled = disc || !hold;
    $("btn-home").disabled = disc || !hold || !state.gains_settled || state.jogging;
    $("btn-pb-stop").disabled = !play;
    $("btn-record").disabled = disc || off || play;

    const pb = state.playback;
    $("pb-bar").style.width = pb ? `${(pb.progress * 100).toFixed(1)}%` : "0%";
    $("pb-elapsed").textContent = fmt(pb ? pb.elapsed : 0);
    $("pb-duration").textContent = fmt(pb ? pb.duration : 0);
    $("pb-point").textContent = pb && pb.point_index >= 0 ? pb.point_index + 1 : "—";
    $("pb-laps").textContent = pb ? pb.laps : 0;
    document.querySelectorAll("#points-table tbody tr").forEach((tr, i) => tr.classList.toggle("current", !!pb && pb.point_index === i));
    if (pb && !pb.stopping && Math.abs(pb.speed_target * 100 - parseFloat($("pb-speed").value)) > 1) {
      $("pb-speed").value = Math.round(pb.speed_target * 100); $("pb-speed-val").textContent = $("pb-speed").value;
    }

    const banner = $("error-banner");
    banner.classList.toggle("hidden", !state.error);
    if (state.error) banner.textContent = "Controller error (holding position): " + state.error;
  }

  // ── program rendering ──────────────────────────────────────────────────
  function renderProgram() {
    if (!program) return;
    if (document.activeElement !== $("prog-name")) $("prog-name").value = program.name;
    $("prog-dirty").textContent = program.dirty ? "• unsaved" : "";
    const tb = $("points-table").querySelector("tbody");
    tb.innerHTML = "";
    $("pose-col-head").textContent = `Pose (mm, ${poseFrame})`;
    program.points.forEach((p, i) => {
      const tr = document.createElement("tr");
      const pp = poseFrame === "home" ? p.pose_home : p.pose;
      const pose = pp ? pp.xyz.map((v) => (v * 1000).toFixed(0)).join(", ") : "—";
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td><input class="pname" value="${escapeHtml(p.name)}"></td>
        <td><select><option value="joint"${p.motion === "joint" ? " selected" : ""}>Joint</option><option value="linear"${p.motion === "linear" ? " selected" : ""}>Linear</option></select></td>
        <td><input type="number" min="5" max="100" step="5" value="${Math.round(p.speed * 100)}"></td>
        <td><input type="checkbox"${p.blend ? " checked" : ""} title="pass through without stopping"></td>
        <td><input type="number" min="0" step="0.1" value="${p.dwell}"></td>
        <td><input type="number" step="0.1" value="${p.gripper.toFixed(2)}"></td>
        <td class="small muted">${pose}</td>
        <td><div class="ops">
          <button class="btn mini" title="move up">▲</button>
          <button class="btn mini" title="move down">▼</button>
          <button class="btn mini blue" title="move the arm to this point (hold mode)">Go</button>
          <button class="btn mini" title="overwrite this point with the current arm pose and gripper">Update</button>
          <button class="btn mini red" title="delete">✕</button>
        </div></td>`;
      const [nameIn, motionSel, speedIn, blendIn, dwellIn, gripIn] = tr.querySelectorAll("input, select");
      const patch = (body) => act("PATCH", `/api/program/points/${p.id}`, body, false).then(() => loadProgram());
      nameIn.onchange = () => patch({ name: nameIn.value });
      motionSel.onchange = () => patch({ motion: motionSel.value });
      speedIn.onchange = () => patch({ speed: Math.min(1, Math.max(0.05, speedIn.value / 100)) });
      blendIn.onchange = () => patch({ blend: blendIn.checked });
      dwellIn.onchange = () => patch({ dwell: Math.max(0, parseFloat(dwellIn.value) || 0) });
      gripIn.onchange = () => patch({ gripper: parseFloat(gripIn.value) });
      const [up, down, go, upd, del] = tr.querySelectorAll(".ops button");
      up.disabled = i === 0; down.disabled = i === program.points.length - 1;
      const reorder = (j) => {
        const ids = program.points.map((x) => x.id); const [id] = ids.splice(i, 1); ids.splice(j, 0, id);
        act("POST", "/api/program/reorder", { ids }, false).then(() => loadProgram());
      };
      up.onclick = () => reorder(i - 1);
      down.onclick = () => reorder(i + 1);
      go.onclick = () => act("POST", `/api/program/points/${p.id}/move_to`, {}, `moving to ${p.name}`);
      upd.onclick = () => patch({ update_from_robot: true });
      del.onclick = () => { if (confirm(`Delete point ${p.name}?`)) act("DELETE", `/api/program/points/${p.id}`, undefined, false).then(() => loadProgram()); };
      tb.appendChild(tr);
    });
    const problems = $("prog-problems");
    problems.classList.toggle("hidden", !program.problems || program.problems.length === 0);
    problems.textContent = (program.problems || []).join("\n");
    const start = $("pb-start"); const prev = start.value;
    start.innerHTML = program.points.map((p, i) => `<option value="${i}">${i + 1}: ${escapeHtml(p.name)}</option>`).join("");
    if (prev && prev < program.points.length) start.value = prev;
    renderState();
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  async function loadProgram() {
    try { program = await api("GET", "/api/program"); programRev = program.rev; renderProgram(); }
    catch (e) { if (e.message !== lastProgError) toast(e.message, true); lastProgError = e.message; }
  }
  async function loadProgramList() {
    try {
      const r = await api("GET", "/api/programs");
      const sel = $("prog-list");
      sel.innerHTML = `<option value="">Load…</option>` + r.programs.map((p) => `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)} (${p.points})</option>`).join("");
    } catch (e) { toast(e.message, true); }
  }

  // ── wiring ─────────────────────────────────────────────────────────────
  function wire() {
    [...$("pose-frame").children].forEach((b) => {
      b.classList.toggle("on", b.dataset.frame === poseFrame);
      b.onclick = () => {
        poseFrame = b.dataset.frame;
        try { localStorage.setItem("poseFrame", poseFrame); } catch (_) { /* storage blocked */ }
        [...$("pose-frame").children].forEach((c) => c.classList.toggle("on", c === b));
        renderProgram();
      };
    });
    $("btn-connect").onclick = async () => {
      $("btn-connect").disabled = true; toast("connecting…");
      try { await act("POST", "/api/connect", undefined, "connected, holding the current pose"); }
      finally { $("btn-connect").disabled = false; }
    };
    $("btn-disconnect").onclick = async () => {
      if (!confirm("Disconnect moves the arm home, waits for the move, then switches the motors off.\nKeep clear of the arm. Continue?")) return;
      toast("moving home, then disconnecting…");
      $("btn-disconnect").disabled = true;
      try { await act("POST", "/api/disconnect", { home: true }); }
      finally { $("btn-disconnect").disabled = false; }
    };
    $("btn-home").onclick = async () => {
      const r = await act("POST", "/api/home", {}, false);
      if (r) toast(`moving home: ${r.duration.toFixed(1)} s`);
    };
    $("btn-free").onclick = () => act("POST", "/api/mode", { mode: "free_drive" }, "free drive: gains fading in");
    $("btn-hold").onclick = () => act("POST", "/api/mode", { mode: "hold" }, "holding");
    $("btn-stop").onclick = () => act("POST", "/api/stop", undefined, "stopping");
    $("btn-estop").onclick = () => { if (confirm("E-STOP disables all motors. A loaded arm will fall. Continue?")) act("POST", "/api/estop", undefined, "motors disabled"); };
    $("btn-enable").onclick = () => act("POST", "/api/enable", undefined, "motors enabled");

    $("btn-record").onclick = async () => {
      const r = await act("POST", "/api/program/points", {
        motion: $("rec-motion").value, speed: parseInt($("rec-speed").value, 10) / 100,
        blend: $("rec-blend").checked, dwell: parseFloat($("rec-dwell").value) || 0,
      }, false);
      if (r) { toast(`recorded ${r.point.name}`); loadProgram(); }
    };
    $("prog-name").onchange = () => { program.name = $("prog-name").value; };
    $("btn-new").onclick = async () => {
      if (program && program.dirty && !confirm("Discard unsaved changes?")) return;
      const name = prompt("Program name", "untitled"); if (!name) return;
      if (await act("POST", "/api/program/new", { name }, "new program")) loadProgram();
    };
    $("btn-save").onclick = async () => {
      const name = $("prog-name").value.trim() || "untitled";
      if (await act("POST", `/api/programs/${encodeURIComponent(name)}/save`, undefined, `saved ${name}`)) { loadProgram(); loadProgramList(); }
    };
    $("prog-list").onchange = async () => {
      const name = $("prog-list").value; if (!name) return;
      if (program && program.dirty && !confirm("Discard unsaved changes?")) { $("prog-list").value = ""; return; }
      if (await act("POST", `/api/programs/${encodeURIComponent(name)}/load`, undefined, `loaded ${name}`)) loadProgram();
      $("prog-list").value = "";
    };
    $("btn-delete-file").onclick = async () => {
      const name = prompt("Delete which saved program? (name)"); if (!name) return;
      if (await act("DELETE", `/api/programs/${encodeURIComponent(name)}`, undefined, `deleted ${name}`)) loadProgramList();
    };

    $("pb-speed").oninput = () => { $("pb-speed-val").textContent = $("pb-speed").value; };
    $("pb-speed").onchange = () => { if (state && state.mode === "playback") act("POST", "/api/playback/speed", { speed: $("pb-speed").value / 100 }, false); };
    const pbReq = () => ({ speed: $("pb-speed").value / 100, loop: $("pb-loop").checked, start_index: parseInt($("pb-start").value || "0", 10) });
    $("btn-plan").onclick = async () => {
      const r = await act("POST", "/api/playback/plan", pbReq(), false);
      if (r) { $("plan-summary").classList.remove("hidden"); $("plan-summary").textContent = `duration ${r.duration.toFixed(2)} s, ${r.samples} samples\narrivals: ${r.point_times.map((t) => t.toFixed(2)).join(", ")} s\nmax joint velocity (rad/s): ${r.max_joint_velocity.map((v) => v.toFixed(2)).join(", ")}`; }
    };
    $("btn-play").onclick = async () => {
      const r = await act("POST", "/api/playback/start", pbReq(), false);
      if (r) toast(`playing: ${r.plan.duration.toFixed(1)} s planned`);
    };
    $("btn-pb-stop").onclick = () => act("POST", "/api/playback/stop", undefined, "stopping");
  }

  // ── keyboard stop: Space (outside text fields) or Escape (anywhere) ────
  // Buttons keep focus after a click and a browser activates a focused button on
  // Space, so the handler runs in the capture phase, cancels the default action on
  // both keydown and keyup, and drops focus before sending the stop.
  const isTyping = (el) => !!el && (el.tagName === "TEXTAREA" || el.isContentEditable ||
    (el.tagName === "INPUT" && !["number", "range", "checkbox", "radio", "button", "submit"].includes(el.type)));
  const isStopKey = (ev) => ev.key === "Escape" || (ev.code === "Space" && !isTyping(ev.target));
  function keyStop(ev) {
    ev.preventDefault(); ev.stopPropagation();
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    if (ev.repeat) return;                        // key held down: one stop is enough
    if (state && !state.connected) return;        // nothing to stop
    const btn = $("btn-stop");
    btn.classList.add("flash"); setTimeout(() => btn.classList.remove("flash"), 400);
    act("POST", "/api/stop", undefined, `stop (${ev.key === "Escape" ? "Esc" : "Space"})`);
  }
  window.addEventListener("keydown", (ev) => { if (isStopKey(ev)) keyStop(ev); }, true);
  window.addEventListener("keyup", (ev) => { if (isStopKey(ev)) { ev.preventDefault(); ev.stopPropagation(); } }, true);
  // A clicked button must not stay focused, or Enter/Space would repeat it (e.g. a jog step).
  document.addEventListener("click", (ev) => { const b = ev.target.closest && ev.target.closest("button"); if (b) b.blur(); });

  // ── scope: one joint at the control rate ───────────────────────────────
  // The server appends, to every state message, the samples of the selected joint recorded
  // since the previous message; the client keeps the last SCOPE_WINDOW seconds and draws them.
  const SCOPE_WINDOW = 5.0;                                       // s
  const SCOPE_KEYS = ["t", "q_cmd", "q", "tau_cmd", "tau_meas"];
  const scope = { joint: null, t: [], q_cmd: [], q: [], tau_cmd: [], tau_meas: [] };
  let scopeDrawPending = false;

  function buildScope() {
    const sel = $("scope-joint");
    config.joint_names.forEach((name, i) => {
      const o = document.createElement("option"); o.value = String(i); o.textContent = `J${i + 1} ${name}`; sel.appendChild(o);
    });
    let saved = null;
    try { saved = localStorage.getItem("scopeJoint"); } catch (_) { /* storage blocked */ }
    if (saved !== null && saved !== "" && Number(saved) < config.joint_names.length) sel.value = saved;
    sel.onchange = () => scopeSelect(sel.value === "" ? null : Number(sel.value));
    scopeSelect(sel.value === "" ? null : Number(sel.value));
    window.addEventListener("resize", requestScopeDraw);
  }
  function scopeSelect(j) {
    scope.joint = j;
    SCOPE_KEYS.forEach((k) => (scope[k] = []));
    try { localStorage.setItem("scopeJoint", j === null ? "" : String(j)); } catch (_) { /* storage blocked */ }
    scopeSendSelection();
    requestScopeDraw();
  }
  function scopeSendSelection() {
    if (wsRef && wsRef.readyState === WebSocket.OPEN) wsRef.send(JSON.stringify({ trace_joint: scope.joint }));
  }
  function scopeAppend(chunk) {
    if (!chunk || chunk.joint !== scope.joint || !chunk.t.length) return;
    SCOPE_KEYS.forEach((k) => { for (const v of chunk[k]) scope[k].push(v); });
    const tEnd = scope.t[scope.t.length - 1];
    let cut = 0;
    while (cut < scope.t.length && scope.t[cut] < tEnd - SCOPE_WINDOW) cut++;
    if (cut) SCOPE_KEYS.forEach((k) => scope[k].splice(0, cut));
    requestScopeDraw();
  }
  function requestScopeDraw() {
    if (scopeDrawPending) return;
    scopeDrawPending = true;
    requestAnimationFrame(() => { scopeDrawPending = false; drawScope(); });
  }
  function niceTicks(lo, hi, want) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / want, p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * p).find((s) => span / s <= want) || 10 * p;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    return out;
  }
  const fmtTick = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2));
  function drawStrip(canvas, tEnd, series, opts) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const lineColor = css.getPropertyValue("--line").trim() || "#262b36", muted = css.getPropertyValue("--muted").trim() || "#8b93a7";
    const x0 = 52, x1 = W - 8, y0 = 6, y1 = H - 16;
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.values) if (v !== null && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
    const minSpan = opts.minSpan || 1e-6;
    if (opts.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi), minSpan / 2); lo = -m; hi = m; }
    if (hi - lo < minSpan) { const c = (hi + lo) / 2; lo = c - minSpan / 2; hi = c + minSpan / 2; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const X = (t) => x1 - (x1 - x0) * (tEnd - t) / SCOPE_WINDOW;
    const Y = (v) => y1 - (y1 - y0) * (v - lo) / (hi - lo);
    ctx.lineWidth = 1; ctx.strokeStyle = lineColor; ctx.fillStyle = muted; ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of niceTicks(lo, hi, 4)) {
      const y = Y(v);
      ctx.globalAlpha = v === 0 ? 1 : 0.6; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(fmtTick(v), x0 - 5, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.globalAlpha = 0.6;
    for (let s = 0; s <= SCOPE_WINDOW; s += 1) {
      const x = X(tEnd - s); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
      if (s > 0) ctx.fillText(`−${s} s`, x, y1 + 3);
    }
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1.5; ctx.lineJoin = "round";
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.beginPath();
      let pen = false;
      for (let i = 0; i < scope.t.length; i++) {
        const v = s.values[i];
        if (v === null || !Number.isFinite(v)) { pen = false; continue; }
        const x = X(scope.t[i]), y = Y(v);
        if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
      }
      ctx.stroke();
    }
  }
  function drawScope() {
    const css = getComputedStyle($("scope-card"));
    const blue = css.getPropertyValue("--scope-cmd").trim() || "#3b82f6", amber = css.getPropertyValue("--scope-meas").trim() || "#d95926", accent = css.getPropertyValue("--scope-err").trim() || "#6ea8fe";
    const n = scope.t.length;
    const tEnd = n ? scope.t[n - 1] : 0;
    const qCmd = scope.q_cmd.map((v) => (v === null ? null : deg(v)));
    const q = scope.q.map((v) => (v === null ? null : deg(v)));
    const err = scope.q.map((v, i) => (v === null || scope.q_cmd[i] === null ? null : deg(v - scope.q_cmd[i])));
    drawStrip($("scope-pos"), tEnd, [{ values: qCmd, color: blue }, { values: q, color: amber }], { minSpan: 0.2 });
    drawStrip($("scope-err"), tEnd, [{ values: err, color: accent }], { symmetric: true, minSpan: 0.1 });
    drawStrip($("scope-tau"), tEnd, [{ values: scope.tau_cmd, color: blue }, { values: scope.tau_meas, color: amber }], { symmetric: true, minSpan: 0.5 });
    const ro = $("scope-readout");
    if (!n) { ro.textContent = scope.joint === null ? "" : "waiting for samples…"; return; }
    const last = n - 1, tm = scope.tau_meas[last];
    ro.textContent = `error ${fmt(err[last], 3)}°   torque ${fmt(scope.tau_cmd[last], 2)} N·m` + (tm === null ? "" : ` (motor ${fmt(tm, 2)})`) + `   ${n} samples`;
  }

  // ── websocket ──────────────────────────────────────────────────────────
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/state`);
    wsRef = ws;
    ws.onopen = () => { $("ws-dot").classList.add("on"); scopeSendSelection(); };
    ws.onclose = () => { $("ws-dot").classList.remove("on"); if (wsRef === ws) wsRef = null; setTimeout(connect, 1000); };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      state = msg.state;
      if (msg.program_rev !== programRev) loadProgram();
      renderState();
      if (msg.trace) scopeAppend(msg.trace);
    };
  }

  async function init() {
    try { config = await api("GET", "/api/config"); } catch (e) { toast("cannot load config: " + e.message, true); return; }
    $("pb-speed").value = Math.round(config.playback.default_speed * 100); $("pb-speed-val").textContent = $("pb-speed").value;
    $("rec-speed").value = Math.round(config.playback.default_speed * 100);
    buildJog(); buildScope(); wire(); await loadProgram(); await loadProgramList(); connect();
  }
  init();
})();
