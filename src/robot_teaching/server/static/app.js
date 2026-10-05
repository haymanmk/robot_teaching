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
    if (state.pose) {
      const [x, y, z] = state.pose.xyz, [r, p, w] = state.pose.rpy;
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

    const hold = state.mode === "hold", free = state.mode === "free_drive", play = state.mode === "playback", off = state.mode === "disabled";
    $("jog-card").classList.toggle("off", !hold || !state.gains_settled);
    $("btn-free").disabled = free || play || off;
    $("btn-hold").disabled = hold || off;
    $("btn-stop").disabled = off;
    $("btn-enable").classList.toggle("hidden", !off);
    $("btn-estop").classList.toggle("hidden", off);
    $("btn-play").disabled = !hold || !state.gains_settled || state.jogging;
    $("btn-plan").disabled = !hold;
    $("btn-pb-stop").disabled = !play;
    $("btn-record").disabled = off || play;

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
    program.points.forEach((p, i) => {
      const tr = document.createElement("tr");
      const pose = p.pose ? p.pose.xyz.map((v) => (v * 1000).toFixed(0)).join(", ") : "—";
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td><input class="pname" value="${escapeHtml(p.name)}"></td>
        <td><select><option value="joint"${p.motion === "joint" ? " selected" : ""}>Joint</option><option value="linear"${p.motion === "linear" ? " selected" : ""}>Linear</option></select></td>
        <td><input type="number" min="5" max="100" step="5" value="${Math.round(p.speed * 100)}"></td>
        <td><input type="checkbox"${p.blend ? " checked" : ""}${p.motion === "linear" ? " disabled" : ""}></td>
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
    $("rec-motion").onchange = () => { if ($("rec-motion").value === "linear") $("rec-blend").checked = false; };
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

  // ── websocket ──────────────────────────────────────────────────────────
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/state`);
    ws.onopen = () => $("ws-dot").classList.add("on");
    ws.onclose = () => { $("ws-dot").classList.remove("on"); setTimeout(connect, 1000); };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      state = msg.state;
      if (msg.program_rev !== programRev) loadProgram();
      renderState();
    };
  }

  async function init() {
    try { config = await api("GET", "/api/config"); } catch (e) { toast("cannot load config: " + e.message, true); return; }
    $("pb-speed").value = Math.round(config.playback.default_speed * 100); $("pb-speed-val").textContent = $("pb-speed").value;
    $("rec-speed").value = Math.round(config.playback.default_speed * 100);
    buildJog(); wire(); await loadProgram(); await loadProgramList(); connect();
  }
  init();
})();
