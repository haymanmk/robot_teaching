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
  const FF_TERMS = ["gravity", "friction", "inertia"];
  let ffTouched = 0;                            // when a feed-forward checkbox was last clicked (the state catches up after the POST)

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
    if (state.feedforward && Date.now() - ffTouched > 600) FF_TERMS.forEach((k) => { $(`ff-${k}`).checked = !!state.feedforward[k]; });
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
    $("btn-reload-gains").disabled = play;

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
    $("btn-reload-gains").onclick = async () => {
      const r = await act("POST", "/api/gains/reload", undefined, false);
      if (r) { toast(r.message); console.log("gains", r.gains); }
    };
    FF_TERMS.forEach((k) => {
      $(`ff-${k}`).onchange = async (ev) => {
        ffTouched = Date.now();
        const r = await act("POST", "/api/feedforward", { [k]: ev.target.checked }, false);
        if (r) toast(r.message); else ev.target.checked = !ev.target.checked;
      };
    });

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
  // since the previous message. The client keeps the last SCOPE_KEEP seconds of them (as long
  // as the server's own history) and draws the last scope.window seconds, or, while paused, a
  // frozen copy that can be dragged back in time.
  const SCOPE_KEEP = 60.0;                                        // s of samples kept on the client
  const SCOPE_WINDOWS = [0.5, 1, 2, 5, 10, 20, 60];               // s, the zoom steps
  const SCOPE_GAP = scopeDraw.GAP;                                // s without samples: lift the pen instead of bridging it
  const SCOPE_KEYS = ["t", "q_cmd", "q", "tau_cmd", "tau_meas", "tau_g", "tau_f", "tau_i"];
  const FFT_FMAX = [5, 10, 25, 50, 100, 250];                     // Hz, the spectrum's frequency-range steps (clipped to fs/2)
  const FFT_UNITS = { err: "°", tau_err: "N·m", q: "°", q_cmd: "°", tau_cmd: "N·m", tau_meas: "N·m", tau_ff: "N·m" };
  const FFT_SEGMENT = 2.0;                                        // s per Welch segment (rounded to a power of two of samples)
  // errMax is the largest |measured − commanded| (°) over every sample received since the joint
  // was picked or "Reset max" was pressed, not only over the samples on screen; it survives a
  // reconnect and a server restart. epoch identifies the server run the samples came from.
  const scope = { joint: null, window: 5.0, epoch: null, errMax: null, shown: 0, t: [], q_cmd: [], q: [], tau_cmd: [], tau_meas: [] };
  const view = { snap: null, end: 0 };                            // paused: the frozen samples and the time at the right edge
  const rec = { on: false, rows: [], t: [], epoch: null, run: 1, started: 0, name: "", limit: 0 };   // recording: CSV rows plus each row's t
  const fft = { on: false, signal: "err", fmax: 50, spec: null, key: "", at: 0 };   // spectrum strip: settings and the last estimate
  const hidden = new Set();                                       // traces unticked in a legend (not kept across reloads)
  let errSig = "pos";                                             // the error strip: "pos" (° measured − commanded) or "tau" (N·m reported − commanded)
  let scopeDrawPending = false;

  function buildScope() {
    const sel = $("scope-joint");
    config.joint_names.forEach((name, i) => {
      const o = document.createElement("option"); o.value = String(i); o.textContent = `J${i + 1} ${name}`; sel.appendChild(o);
    });
    let saved = null, savedWindow = NaN;
    try { saved = localStorage.getItem("scopeJoint"); savedWindow = Number(localStorage.getItem("scopeWindow")); } catch (_) { /* storage blocked */ }
    if (saved !== null && saved !== "" && Number(saved) < config.joint_names.length) sel.value = saved;
    if (SCOPE_WINDOWS.includes(savedWindow)) scope.window = savedWindow;
    sel.onchange = () => scopeSelect(sel.value === "" ? null : Number(sel.value));
    $("scope-zoom-in").onclick = () => scopeZoom(-1);
    $("scope-zoom-out").onclick = () => scopeZoom(+1);
    $("scope-pause").onclick = () => scopePause(!view.snap);
    $("scope-record").onclick = () => (rec.on ? scopeRecordStop() : scopeRecordStart());
    $("scope-reset-max").onclick = () => { scope.errMax = null; requestScopeDraw(); };
    ["scope-pos", "scope-err", "scope-tau"].forEach((id) => scopeWireCanvas($(id)));
    document.querySelectorAll("#scope-card input[data-trace]").forEach((cb) => {
      cb.onchange = () => { if (cb.checked) hidden.delete(cb.dataset.trace); else hidden.add(cb.dataset.trace); cb.closest(".li").classList.toggle("off", !cb.checked); requestScopeDraw(); };
    });
    try { errSig = localStorage.getItem("scopeErrSignal") === "tau" ? "tau" : "pos"; } catch (_) { /* storage blocked */ }
    $("scope-err-signal").value = errSig;
    $("scope-err-signal").onchange = (ev) => { errSig = ev.target.value === "tau" ? "tau" : "pos"; try { localStorage.setItem("scopeErrSignal", errSig); } catch (_) { /* storage blocked */ } requestScopeDraw(); };
    try {
      fft.on = localStorage.getItem("scopeFft") === "1";
      const sig = localStorage.getItem("scopeFftSignal"), fm = Number(localStorage.getItem("scopeFftFmax"));
      if (FFT_UNITS[sig]) fft.signal = sig;
      if (FFT_FMAX.includes(fm)) fft.fmax = fm;
    } catch (_) { /* storage blocked */ }
    $("fft-signal").value = fft.signal;
    $("fft-signal").onchange = () => { fft.signal = $("fft-signal").value; try { localStorage.setItem("scopeFftSignal", fft.signal); } catch (_) { /* storage blocked */ } requestScopeDraw(); };
    $("scope-fft-toggle").onclick = () => fftShow(!fft.on);
    $("fft-zoom-in").onclick = () => fftZoom(-1);
    $("fft-zoom-out").onclick = () => fftZoom(+1);
    $("scope-fft").addEventListener("wheel", (ev) => { if (!(ev.ctrlKey || ev.metaKey)) return; ev.preventDefault(); fftZoom(ev.deltaY > 0 ? +1 : -1); }, { passive: false });
    fftZoom(0);
    fftShow(fft.on);
    window.addEventListener("beforeunload", (ev) => { if (rec.on) { ev.preventDefault(); ev.returnValue = ""; } });
    scopeZoom(0);
    scopeSelect(sel.value === "" ? null : Number(sel.value));
    window.addEventListener("resize", requestScopeDraw);
  }
  function scopeSelect(j) {
    scope.joint = j;
    SCOPE_KEYS.forEach((k) => (scope[k] = []));
    scope.errMax = null;
    if (view.snap) scopePause(false);                             // the frozen copy belongs to the previous joint
    try { localStorage.setItem("scopeJoint", j === null ? "" : String(j)); } catch (_) { /* storage blocked */ }
    scopeSendSelection();
    requestScopeDraw();
  }
  function scopeSendSelection() {
    if (wsRef && wsRef.readyState === WebSocket.OPEN) wsRef.send(JSON.stringify({ trace_joint: scope.joint }));
  }
  function scopeZoom(dir) {
    const i = Math.min(SCOPE_WINDOWS.length - 1, Math.max(0, SCOPE_WINDOWS.indexOf(scope.window) + dir));
    scope.window = SCOPE_WINDOWS[i];
    try { localStorage.setItem("scopeWindow", String(scope.window)); } catch (_) { /* storage blocked */ }
    $("scope-zoom-in").disabled = i === 0;
    $("scope-zoom-out").disabled = i === SCOPE_WINDOWS.length - 1;
    requestScopeDraw();
  }
  function scopePause(on) {
    if (on && !view.snap) {
      view.snap = {}; SCOPE_KEYS.forEach((k) => (view.snap[k] = scope[k].slice()));
      view.end = scope.t.length ? scope.t[scope.t.length - 1] : 0;
    } else if (!on) {
      view.snap = null;
    }
    $("scope-pause").textContent = view.snap ? "▶ Resume" : "⏸ Pause";
    $("scope-pause").classList.toggle("on", !!view.snap);
    $("scope-card").classList.toggle("paused", !!view.snap);
    requestScopeDraw();
  }
  function scopeWireCanvas(canvas) {
    // Paused: drag to look back in time. Any time: Ctrl/⌘ + wheel zooms (a plain wheel scrolls the page).
    let drag = null;
    canvas.addEventListener("pointerdown", (ev) => {
      if (!view.snap || !view.snap.t.length) return;
      drag = { x: ev.clientX, end: view.end, px: Math.max(1, canvas.clientWidth - scopeDraw.PLOT_LEFT - scopeDraw.PLOT_RIGHT) };
      canvas.setPointerCapture(ev.pointerId);
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!drag || !view.snap) return;
      const t = view.snap.t;
      view.end = Math.min(t[t.length - 1], Math.max(t[0], drag.end - (ev.clientX - drag.x) / drag.px * scope.window));
      requestScopeDraw();
    });
    const release = () => { drag = null; };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("wheel", (ev) => {
      if (!(ev.ctrlKey || ev.metaKey)) return;
      ev.preventDefault();
      scopeZoom(ev.deltaY > 0 ? +1 : -1);
    }, { passive: false });
  }
  function scopeAppend(chunk) {
    if (!chunk || chunk.joint !== scope.joint || !chunk.t.length) return;
    const t0 = chunk.t[0], epoch = chunk.epoch === undefined ? null : chunk.epoch;
    if (scope.epoch !== null && epoch !== scope.epoch) {
      // A restarted server: its clock starts again from zero, so nothing kept compares with the
      // new samples. (Left in place they would count as newer than the window's end, never be
      // trimmed, and be drawn past the right edge beside the live profile.) A recording keeps
      // its rows; the run column marks the restart.
      SCOPE_KEYS.forEach((k) => (scope[k] = []));
      if (rec.on) rec.run++;
    } else {
      // On every (re)connection the server restarts the stream from its newest samples, so a
      // chunk may begin at or before the newest sample we hold: the re-sent rows replace what
      // we kept, on screen and in the recording.
      let keep = scope.t.length;
      while (keep > 0 && scope.t[keep - 1] >= t0) keep--;
      if (keep < scope.t.length) SCOPE_KEYS.forEach((k) => scope[k].splice(keep));
      if (rec.on && rec.epoch === epoch) {
        let n = rec.t.length;
        while (n > 0 && rec.t[n - 1] >= t0) n--;
        rec.rows.length = rec.t.length = n;
      }
    }
    scope.epoch = epoch;
    for (let i = 0; i < chunk.t.length; i++) {
      const qc = chunk.q_cmd[i], qm = chunk.q[i];
      if (qc !== null && qm !== null) {
        const e = Math.abs(deg(qm - qc));
        if (Number.isFinite(e) && (scope.errMax === null || e > scope.errMax)) scope.errMax = e;
      }
      if (rec.on) scopeRecordRow(chunk, i, epoch);
    }
    SCOPE_KEYS.forEach((k) => { for (const v of chunk[k]) scope[k].push(v); });
    const tEnd = scope.t[scope.t.length - 1];
    let cut = 0;
    while (cut < scope.t.length && scope.t[cut] < tEnd - SCOPE_KEEP) cut++;
    if (cut) SCOPE_KEYS.forEach((k) => scope[k].splice(0, cut));
    if (view.snap) drawReadout(); else requestScopeDraw();        // paused: the traces stay, the readout is live
  }

  // Recording: every sample of the selected joint, as received, into CSV rows; Stop downloads them.
  const num = (v, d) => (v === null || v === undefined ? "" : Number(v).toFixed(d));
  const stamp = (ms) => { const d = new Date(ms), p = (x) => String(x).padStart(2, "0"); return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`; };
  function scopeRecordStart() {
    if (scope.joint === null) { toast("pick a joint to record", true); return; }
    rec.on = true; rec.rows = []; rec.t = []; rec.epoch = scope.epoch; rec.run = 1; rec.started = Date.now();
    rec.name = `scope_J${scope.joint + 1}_${stamp(rec.started)}.csv`;
    rec.limit = Math.round(3600 * (config.control_rate || 500));    // an hour at the control rate
    $("scope-record").classList.add("on");
    requestScopeDraw();
  }
  function scopeRecordRow(chunk, i, epoch) {
    const qc = chunk.q_cmd[i], qm = chunk.q[i];
    rec.rows.push(`${num(chunk.t[i], 4)},J${chunk.joint + 1},${num(qc, 6)},${num(qm, 6)},${qc === null || qm === null ? "" : (qm - qc).toFixed(6)},${num(chunk.tau_cmd[i], 4)},${num(chunk.tau_meas[i], 4)},${num(chunk.tau_g[i], 4)},${num(chunk.tau_f[i], 4)},${num(chunk.tau_i[i], 4)},${rec.run}`);
    rec.t.push(chunk.t[i]); rec.epoch = epoch;
    if (rec.rows.length >= rec.limit) { scopeRecordStop(); toast(`recording stopped after ${rec.limit} rows and saved`, true); }
  }
  function scopeRecordStop() {
    if (!rec.on) return;
    rec.on = false;
    const n = rec.rows.length, name = rec.name;
    const csv = "t_s,joint,q_cmd_rad,q_rad,err_rad,tau_cmd_Nm,tau_meas_Nm,tau_g_Nm,tau_f_Nm,tau_i_Nm,run\n" + rec.rows.join("\n") + "\n";
    rec.rows = []; rec.t = [];
    $("scope-record").classList.remove("on");
    requestScopeDraw();
    if (!n) { toast("nothing recorded"); return; }
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    toast(`saved ${name}: ${n} rows`);
  }

  function requestScopeDraw() {
    if (scopeDrawPending) return;
    scopeDrawPending = true;
    requestAnimationFrame(() => { scopeDrawPending = false; drawScope(); });
  }
  // ── spectrum strip ──
  function fftShow(on) {
    fft.on = on;
    $("fft-strip").classList.toggle("hidden", !on);
    $("scope-fft-toggle").classList.toggle("on", on);
    try { localStorage.setItem("scopeFft", on ? "1" : "0"); } catch (_) { /* storage blocked */ }
    requestScopeDraw();
  }
  function fftZoom(dir) {
    const i = Math.min(FFT_FMAX.length - 1, Math.max(0, FFT_FMAX.indexOf(fft.fmax) + dir));
    fft.fmax = FFT_FMAX[i];
    try { localStorage.setItem("scopeFftFmax", String(fft.fmax)); } catch (_) { /* storage blocked */ }
    $("fft-zoom-in").disabled = i === 0;
    $("fft-zoom-out").disabled = i === FFT_FMAX.length - 1;
    requestScopeDraw();
  }
  // The spectrum of the chosen signal over the samples on screen (src[i0:i1], times t; series holds
  // the derived position and error series already in degrees). Live, it is re-estimated at most
  // every 150 ms; paused, whenever the view moves.
  function drawSpectrum(src, i0, i1, t, series) {
    const unit = FFT_UNITS[fft.signal];
    $("fft-units").textContent = `amplitude, dB re 1 ${unit}`;
    const key = `${scope.joint}|${fft.signal}|${scope.window}|${view.snap ? view.end : "live"}`;
    const now = performance.now();
    if (key !== fft.key || (!view.snap && now - fft.at >= 150)) {
      const values = fft.signal === "err" ? series.err : fft.signal === "tau_err" ? series.tauErr : fft.signal === "q" ? series.q : fft.signal === "q_cmd" ? series.qCmd : fft.signal === "tau_ff" ? series.ff : src[fft.signal].slice(i0, i1);
      fft.spec = window.scopeDSP ? scopeDSP.welch(values, t, { segmentSeconds: FFT_SEGMENT, maxSegments: 32, gap: SCOPE_GAP }) : null;
      fft.key = key; fft.at = now;
    }
    const accent = getComputedStyle($("scope-card")).getPropertyValue("--scope-err").trim() || "#6ea8fe";
    $("fft-readout").textContent = scopeDraw.drawSpectrumStrip($("scope-fft"), fft.spec, { unit, fmax: fft.fmax, accent, empty: scope.joint === null ? "pick a joint" : "waiting for 64 gap-free samples" });
  }
  function drawScope() {
    const css = getComputedStyle($("scope-card"));
    const blue = css.getPropertyValue("--scope-cmd").trim() || "#3b82f6", amber = css.getPropertyValue("--scope-meas").trim() || "#d95926", accent = css.getPropertyValue("--scope-err").trim() || "#6ea8fe";
    const green = css.getPropertyValue("--scope-ff").trim() || "#199e70";
    const src = view.snap || scope, W = scope.window, n = src.t.length;
    const tEnd = view.snap ? view.end : (n ? src.t[n - 1] : 0);
    // On screen: from one sample before the left edge to the last one at or before the right edge.
    const i1 = view.snap ? scopeDraw.upperBound(src.t, tEnd) : n;
    const i0 = Math.max(0, scopeDraw.lowerBound(src.t, tEnd - W) - 1);
    const t = src.t.slice(i0, i1);
    const qCmd = src.q_cmd.slice(i0, i1).map((v) => (v === null ? null : deg(v)));
    const q = src.q.slice(i0, i1).map((v) => (v === null ? null : deg(v)));
    const err = q.map((v, i) => (v === null || qCmd[i] === null ? null : v - qCmd[i]));
    const tauErr = src.tau_meas.slice(i0, i1).map((v, i) => { const c = src.tau_cmd[i0 + i]; return v === null || c === null ? null : v - c; });
    // The feed-forward the commanded torque contains: the gravity, friction and inertia terms as applied.
    const ff = src.tau_g.slice(i0, i1).map((g, i) => { const f = src.tau_f[i0 + i], d = src.tau_i[i0 + i]; return g === null || f === null || d === null ? null : g + f + d; });
    const axis = { tEnd, W, t, absolute: !!view.snap };
    const shown = (list) => list.filter((s) => !hidden.has(s.key));
    scopeDraw.drawStrip($("scope-pos"), axis, shown([{ key: "q_cmd", values: qCmd, color: blue }, { key: "q", values: q, color: amber }]), { minSpan: 0.2 });
    $("scope-err-units").textContent = errSig === "tau" ? "N·m reported − commanded" : "° measured − commanded";
    scopeDraw.drawStrip($("scope-err"), axis, [{ values: errSig === "tau" ? tauErr : err, color: accent }], { symmetric: true, minSpan: errSig === "tau" ? 0.05 : 0.1 });
    scopeDraw.drawStrip($("scope-tau"), axis, shown([{ key: "tau_ff", values: ff, color: green }, { key: "tau_cmd", values: src.tau_cmd.slice(i0, i1), color: blue }, { key: "tau_meas", values: src.tau_meas.slice(i0, i1), color: amber }]), { symmetric: true, minSpan: 0.5 });
    scope.shown = t.length;
    if (fft.on) drawSpectrum(src, i0, i1, t, { qCmd, q, err, tauErr, ff });
    drawReadout();
  }
  function drawReadout() {
    const W = scope.window;
    $("scope-window").textContent = (W < 1 ? `${W * 1000} ms` : `${W} s`) + (view.snap ? " (paused)" : "");
    $("scope-record").textContent = rec.on ? `■ Stop ${Math.round((Date.now() - rec.started) / 1000)} s` : "● Record";
    const ro = $("scope-readout"), n = scope.t.length;
    if (!n) { ro.textContent = scope.joint === null ? "" : "waiting for samples…"; return; }
    const last = n - 1, tm = scope.tau_meas[last], qc = scope.q_cmd[last], qm = scope.q[last];
    const e = qc === null || qm === null ? null : deg(qm - qc);
    ro.textContent = `error ${fmt(e, 3)}°  max |error| ${fmt(scope.errMax, 3)}°   torque ${fmt(scope.tau_cmd[last], 2)} N·m` + (tm === null ? "" : ` (motor ${fmt(tm, 2)})`) + `   ${scope.shown} samples on screen`;
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
