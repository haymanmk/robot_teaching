"""The ripple feed-forward: the table, learning it from recordings, and the term in the controller."""

import dataclasses

import numpy as np
import pytest

from robot_teaching.backend.sim import SimBackend
from robot_teaching.cli import main
from robot_teaching.controller import Mode, TeachController
from robot_teaching.planning import plan_program
from robot_teaching.program import Program, TaughtPoint
from robot_teaching.ripple import Recording, RippleMap, RippleSine, RippleTable, fit_table, read_recording

from conftest import L_POSE

CYCLES = 19.8                     # cycles per rad, the simulator's ripple (as measured on J3)
J3 = 2


def _with_ripple(cfg, path, on=True, **ripple):
    return dataclasses.replace(cfg, feedforward=dataclasses.replace(cfg.feedforward, ripple=on),
                               ripple=dataclasses.replace(cfg.ripple, file=str(path), **ripple))


def settle(sim, ctl, n):
    sim.step(n)
    return ctl.snapshot()


# ── the table ──────────────────────────────────────────────────────────────────────────────

def test_table_interpolates_blends_the_directions_and_is_zero_outside():
    t = RippleTable(1.0, 0.1, forward=[0.0, 1.0, 0.0], reverse=[0.0, -0.5, 0.0])
    assert t.stop == pytest.approx(1.2)
    assert t.torque(1.1, 1.0) == pytest.approx(1.0) and t.torque(1.1, -1.0) == pytest.approx(-0.5)
    assert t.torque(1.05, 1.0) == pytest.approx(0.5), "between two values: a straight line"
    assert t.torque(1.1, 0.0) == pytest.approx(0.25), "at rest: the mean of the two directions"
    assert t.torque(1.1, 0.5) == pytest.approx(0.75 * 1.0 + 0.25 * -0.5)
    assert t.torque(0.99, 1.0) == 0.0 and t.torque(1.21, -1.0) == 0.0
    for bad in (dict(step=0.0), dict(forward=[0.0, 1.0]), dict(forward=[0.0, np.nan, 0.0])):
        with pytest.raises(ValueError):
            RippleTable(**{**dict(start=1.0, step=0.1, forward=[0.0, 1.0, 0.0], reverse=[0.0, 0.5, 0.0]), **bad})


def test_table_with_a_sinusoid_continues_as_it_outside_its_range():
    sine = RippleSine(2.0, forward=[0.5, 0.0], reverse=[0.0, 0.3])       # one cycle per 0.5 rad
    assert sine.per_revolution == pytest.approx(4.0 * np.pi)
    assert sine.amplitude("forward") == pytest.approx(0.5) and sine.amplitude("reverse") == pytest.approx(0.3)
    t = RippleTable(1.0, 0.1, forward=[0.0, 1.0, 0.0], reverse=[0.3, -0.5, 0.3], sine=sine)
    assert t.torque(1.1, 1.0) == pytest.approx(1.0) and t.torque(1.1, -1.0) == pytest.approx(-0.5), "inside: the table"
    assert t.torque(1.0, 1.0) == pytest.approx(0.0) and t.torque(1.2, -1.0) == pytest.approx(0.3), "its ends included"
    assert t.torque(0.625, 1.0) == pytest.approx(0.5) and t.torque(0.625, -1.0) == pytest.approx(0.0, abs=1e-12)
    assert t.torque(2.0, -1.0) == pytest.approx(0.3) and t.torque(2.0, 0.0) == pytest.approx(0.15), "blended alike"
    assert t.torque(-2.875, 1.0) == pytest.approx(0.5), "however far"
    for bad in (dict(cycles=0.0), dict(forward=[0.5]), dict(reverse=[0.0, np.inf])):
        with pytest.raises(ValueError):
            RippleSine(**{**dict(cycles=2.0, forward=[0.5, 0.0], reverse=[0.0, 0.3]), **bad})


def test_map_gives_a_torque_only_to_the_joints_with_a_table_and_survives_a_file(tmp_path):
    table = RippleTable(0.9, 0.05, forward=[0.0, 0.4, 0.4, 0.0], reverse=[0.0, 0.2, 0.2, 0.0])
    m = RippleMap(6, {J3: table})
    q = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
    assert np.allclose(m.torque(q, np.ones(6)), [0, 0, 0.4, 0, 0, 0])
    assert np.allclose(m.torque(q, -np.ones(6)), [0, 0, 0.2, 0, 0, 0])
    assert m and not RippleMap(6) and m.describe() == "J3 0.900..1.050 rad" and RippleMap(6).describe() == "no table"
    path = tmp_path / "ripple.yaml"
    m.save(path)
    back = RippleMap.load(path, 6)
    assert list(back.tables) == [J3] and np.allclose(back.tables[J3].forward, table.forward)
    assert back.tables[J3].start == pytest.approx(0.9) and back.tables[J3].step == pytest.approx(0.05)
    assert not RippleMap.load(tmp_path / "missing.yaml", 6) and not RippleMap.load(None, 6), "no file: no tables"
    with pytest.raises(ValueError):
        RippleMap.load(path, 2)                         # J3 on an arm of two joints
    path.write_text("J3:\n  start: 0.9\n  step: 0.05\n  forward: [0, 1, 0]\n")
    with pytest.raises(ValueError, match="reverse"):
        RippleMap.load(path, 6)
    path.write_text("elbow:\n  start: 0.9\n")
    with pytest.raises(ValueError, match="joint label"):
        RippleMap.load(path, 6)
    # A table with a sinusoid: it is stored and read back with it.
    table.sine = RippleSine(20.053523, [0.31, -0.2], [0.25, 0.1])
    assert RippleMap(6, {J3: table}).describe() == "J3 0.900..1.050 rad (beyond: sinusoid, 20.05 cycles/rad)"
    RippleMap(6, {J3: table, 0: RippleTable(0.0, 0.1, [0.0, 0.1], [0.0, 0.1])}).save(path)
    back = RippleMap.load(path, 6)
    assert back.tables[0].sine is None and back.tables[J3].sine.cycles == pytest.approx(20.053523)
    assert np.allclose(back.tables[J3].sine.forward, [0.31, -0.2]) and np.allclose(back.tables[J3].sine.reverse, [0.25, 0.1])
    assert back.tables[J3].torque(3.0, 1.0) == pytest.approx(table.torque(3.0, 1.0)) != 0.0
    path.write_text(path.read_text().replace("cycles_per_rad", "cycles"))
    with pytest.raises(ValueError, match="cycles_per_rad"):
        RippleMap.load(path, 6)


# ── learning it ────────────────────────────────────────────────────────────────────────────

def _strokes(lo=0.6, hi=1.4, speed=0.15, rate=500.0, strokes=4, joint=J3, noise=0.05, seed=0):
    """A joint going back and forth at constant speed; its motor torque is a slow trend
    (gravity, friction with the direction) plus a ripple that differs between the directions."""
    rng = np.random.default_rng(seed)
    n = int((hi - lo) / speed * rate)
    up = np.linspace(lo, hi, n)
    q_cmd = np.concatenate([up if k % 2 == 0 else up[::-1] for k in range(strokes)])
    t = np.arange(len(q_cmd)) / rate
    direction = np.concatenate([np.full(n, 1.0 if k % 2 == 0 else -1.0) for k in range(strokes)])
    q = q_cmd - 0.002 * direction
    tau = 6.0 * np.cos(q) + 0.4 * direction + _truth(q, direction) + noise * rng.standard_normal(len(q))
    return Recording(joint, t, q_cmd, q, tau)


def _truth(q, direction):
    phase = 2.0 * np.pi * CYCLES * np.asarray(q)
    return 0.5 * np.sin(phase) + 0.2 * direction * np.cos(phase) + 0.1 * np.sin(2.0 * phase + 1.0)


def test_fit_recovers_the_ripple_of_each_direction_and_leaves_the_trend_out():
    fit = fit_table([_strokes(seed=0), _strokes(seed=1)])
    t = fit.table
    assert fit.joint == J3 and fit.speed == pytest.approx(0.15, rel=0.05)
    assert t.step == pytest.approx(0.002) and t.start == pytest.approx(0.6, abs=0.01) and t.stop == pytest.approx(1.4, abs=0.01)
    inner = np.arange(0.7, 1.3, 0.001)                  # away from the ends, where the trend is one-sided
    for direction, name in ((1.0, "forward"), (-1.0, "reverse")):
        got = np.array([t.torque(x, direction) for x in inner])
        want = _truth(inner, direction)
        assert np.std(got - want) < 0.08 * np.std(want), f"{name}: the ripple, not the gravity or the friction"
        assert abs(np.mean(got)) < 0.02
        assert fit.rms[name] == pytest.approx(np.std(want), rel=0.1) and fit.explained[name] > 0.95
        assert fit.samples[name] > 4000
    # The ripple's fundamental is found as a sinusoid, and the ends come down to it.
    s = t.sine
    assert s is fit.sine and s.cycles == pytest.approx(CYCLES, abs=0.02) and 0.9 < fit.sine_share < 1.0
    assert np.allclose(s.forward, [0.5, 0.2], atol=0.03) and np.allclose(s.reverse, [0.5, -0.2], atol=0.03)
    assert t.forward[0] == pytest.approx(s.curves(t.start)[0]) and t.reverse[-1] == pytest.approx(s.curves(t.stop)[1])
    assert abs(t.forward[1] - s.curves(t.start + t.step)[0]) < 0.05, "smoothly"

    # Without it they come down to zero.
    plain = fit_table([_strokes(seed=0), _strokes(seed=1)], sine=False)
    assert plain.table.sine is None and plain.sine is None and plain.sine_share == 0.0
    assert plain.table.forward[0] == 0.0 and plain.table.reverse[-1] == 0.0
    assert abs(plain.table.forward[1]) < 0.2 * np.max(np.abs(plain.table.forward))
    assert np.allclose(plain.table.forward[20:-20], t.forward[20:-20]), "away from the ends the table is the same"


def _fundamental(q, direction):
    phase = 2.0 * np.pi * CYCLES * np.asarray(q)
    return 0.5 * np.sin(phase) + 0.2 * direction * np.cos(phase)


def test_fit_gives_angles_it_has_not_seen_the_sinusoid():
    low, high = _strokes(0.6, 0.9), _strokes(1.1, 1.4, seed=1)
    t = fit_table([low, high]).table
    assert t.start == pytest.approx(0.6, abs=0.01) and t.stop == pytest.approx(1.4, abs=0.01)
    assert abs(t.torque(0.75, 1.0) - _truth(0.75, 1.0)) < 0.1 and abs(t.torque(1.25, -1.0) - _truth(1.25, -1.0)) < 0.1
    # Between the two strokes and beyond them nothing was recorded: the ripple's fundamental.
    for x in (np.arange(0.93, 1.07, 0.003), np.arange(0.2, 0.58, 0.003), np.arange(1.42, 2.0, 0.003)):
        for direction in (1.0, -1.0):
            got = np.array([t.torque(v, direction) for v in x])
            assert np.max(np.abs(got - _fundamental(x, direction))) < 0.06
    every = np.arange(0.2, 2.0, 0.0005)
    assert np.max(np.abs(np.diff([t.torque(v, 1.0) for v in every]))) < 0.06, "no step where the table ends"

    # The frequency can be given instead of searched for.
    given = fit_table([low, high], sine_cycles=CYCLES).table
    assert given.sine.cycles == CYCLES and abs(given.torque(1.8, 1.0) - _fundamental(1.8, 1.0)) < 0.05

    # Without the sinusoid those angles get nothing.
    t = fit_table([low, high], sine=False).table
    assert t.torque(1.0, 1.0) == 0.0 and t.torque(1.0, -1.0) == 0.0 and t.torque(1.6, 1.0) == 0.0
    assert abs(t.torque(0.75, 1.0) - _truth(0.75, 1.0)) < 0.1


def test_fit_keeps_a_sinusoid_only_where_the_ripple_is_one():
    noise = _strokes()
    noise.tau = noise.tau - _truth(noise.q, np.sign(np.gradient(noise.q_cmd)))    # the trend and the noise alone
    fit = fit_table([noise])
    assert fit.sine is not None and fit.sine_share < 0.3 and fit.table.sine is None
    assert fit.table.forward[0] == 0.0 and fit.table.torque(2.0, 1.0) == 0.0, "zero outside, as without one"
    short = fit_table([_strokes(1.0, 1.1)])                                       # two cycles: too few to tell
    assert short.sine is None and short.table.sine is None
    with pytest.raises(ValueError, match="sine_cycles"):
        fit_table([_strokes()], sine_cycles=0.0)


def test_fit_refuses_what_it_cannot_use():
    with pytest.raises(ValueError, match="no recordings"):
        fit_table([])
    with pytest.raises(ValueError, match="different joints"):
        fit_table([_strokes(joint=1), _strokes(joint=2)])
    still = _strokes()
    still.q_cmd[:] = 1.0
    with pytest.raises(ValueError, match="does not move"):
        fit_table([still])
    with pytest.raises(ValueError):
        fit_table([_strokes()], step=0.0)


def _write_csv(path, rec, header="t_s,joint,q_cmd_rad,q_rad,err_rad,tau_cmd_Nm,tau_meas_Nm,run", meas=True):
    rows = [header]
    for t, qc, q, tau in zip(rec.t, rec.q_cmd, rec.q, rec.tau):
        rows.append(f"{t:.4f},J{rec.joint + 1},{qc:.6f},{q:.6f},{q - qc:.6f},{tau:.4f},{f'{tau + 0.01:.4f}' if meas else ''},1")
    path.write_text("\n".join(rows) + "\n")


def test_recording_reader_takes_the_motor_torque_or_the_command(tmp_path):
    rec = _strokes(strokes=1)
    _write_csv(tmp_path / "a.csv", rec)
    a = read_recording(tmp_path / "a.csv")
    assert a.joint == J3 and a.name == "a.csv" and len(a.t) == len(rec.t)
    assert np.allclose(a.q, rec.q, atol=1e-6) and np.allclose(a.tau, rec.tau + 0.01, atol=1e-4), "the motor's report"
    _write_csv(tmp_path / "b.csv", rec, meas=False)
    assert np.allclose(read_recording(tmp_path / "b.csv").tau, rec.tau, atol=1e-4), "the command where the motor reported none"
    (tmp_path / "c.csv").write_text("t_s,joint,q_cmd_rad,q_rad\n")
    with pytest.raises(ValueError, match="no samples"):
        read_recording(tmp_path / "c.csv")


def test_ripple_fit_command_writes_the_joint_and_keeps_the_others(tmp_path, capsys):
    out = tmp_path / "ripple.yaml"
    RippleMap(6, {0: RippleTable(0.0, 0.1, [0.0, 0.1, 0.0], [0.0, 0.1, 0.0])}).save(out)
    _write_csv(tmp_path / "j3.csv", _strokes())
    assert main(["ripple-fit", str(tmp_path / "j3.csv"), "--out", str(out), "--dry-run"]) == 0
    assert list(RippleMap.load(out, 6).tables) == [0], "a dry run writes nothing"
    assert main(["ripple-fit", str(tmp_path / "j3.csv"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "J3:" in text and "added the table of J3" in text and "Reload gains" in text
    m = RippleMap.load(out, 6)
    assert sorted(m.tables) == [0, J3] and abs(m.tables[J3].torque(1.0, 1.0) - _truth(1.0, 1.0)) < 0.1
    assert main(["ripple-fit", str(tmp_path / "j3.csv"), "--out", str(out)]) == 0
    assert "replaced the table of J3" in capsys.readouterr().out
    assert main(["ripple-fit", str(tmp_path / "nothing.csv"), "--out", str(out)]) == 1
    # The sinusoid: found, stored and used beyond the recorded 0.6 to 1.4 rad, unless it is refused.
    assert "sinusoid 19.8" in text and "continues as it outside its range" in text
    assert abs(m.tables[J3].torque(2.0, 1.0) - _fundamental(2.0, 1.0)) < 0.06
    capsys.readouterr()
    assert main(["ripple-fit", str(tmp_path / "j3.csv"), "--out", str(out), "--sine-per-rev", "124"]) == 0
    assert "(124.0 per revolution)" in capsys.readouterr().out
    assert RippleMap.load(out, 6).tables[J3].sine.per_revolution == pytest.approx(124.0)
    assert main(["ripple-fit", str(tmp_path / "j3.csv"), "--out", str(out), "--no-sine"]) == 0
    assert "sinusoid" not in capsys.readouterr().out
    m = RippleMap.load(out, 6)
    assert m.tables[J3].sine is None and m.tables[J3].torque(2.0, 1.0) == 0.0
    assert main(["ripple-fit", str(tmp_path / "j3.csv"), "--out", str(out), "--sine-per-rev", "0"]) == 1


# ── the term in the controller ─────────────────────────────────────────────────────────────

def _flat_table(path, forward=0.4, reverse=0.2):
    """A table around the L pose's J3 with one value per direction (and one on J5)."""
    RippleMap(6, {J3: RippleTable(0.5, 0.1, [0.0] + [forward] * 10 + [0.0], [0.0] + [reverse] * 10 + [0.0]),
                  4: RippleTable(-0.5, 0.1, [0.0] + [0.1] * 9 + [0.0], [0.0] + [0.1] * 9 + [0.0])}).save(path)


def _rig(cfg, model, **sim_kw):
    sim = SimBackend(realtime=False, q0=L_POSE, model=model, **sim_kw)
    ctl = TeachController(sim, cfg)
    ctl.start()
    return sim, ctl


def test_ripple_term_is_off_by_default_and_ramps_in_at_the_measured_angle(cfg, model, tmp_path):
    _flat_table(tmp_path / "r.yaml")
    sim, ctl = _rig(_with_ripple(cfg, tmp_path / "r.yaml", on=False), model)
    s = settle(sim, ctl, 300)
    assert np.all(s.tau_r == 0.0) and ctl.feedforward()["ripple"] is False
    assert ctl.gains()["ripple"]["joints"] == ["J3", "J5"]
    ok, msg = ctl.set_feedforward(ripple=True)
    assert ok and msg.endswith("ripple on")
    s = settle(sim, ctl, 20)
    assert 0.0 < s.tau_r[J3] < 0.25, "the term ramps in, it does not step"
    s = settle(sim, ctl, int(cfg.free_drive.transition_duration * 500) + 20)
    assert np.allclose(s.tau_r, [0, 0, 0.3, 0, 0.1, 0]), "standing still: the mean of the two directions"
    assert np.allclose(sim._cmd[4], s.tau_g + s.tau_o + s.tau_r), "and it is in the torque sent"
    assert ctl.trace_since(J3, None, 5)["tau_r"][-1] == pytest.approx(0.3)
    # Jogging the joint: the curve of the direction it is told to move in.
    assert ctl.jog_to(L_POSE + np.array([0, 0, 0.2, 0, 0, 0]))[0]
    assert settle(sim, ctl, 100).tau_r[J3] == pytest.approx(0.4, abs=1e-3)
    settle(sim, ctl, 600)
    assert ctl.jog_to(L_POSE)[0]
    assert settle(sim, ctl, 100).tau_r[J3] == pytest.approx(0.2, abs=1e-3)
    ctl.shutdown()


def test_ripple_term_is_clipped_and_fades_out_in_free_drive(cfg, model, tmp_path):
    _flat_table(tmp_path / "r.yaml")
    sim, ctl = _rig(_with_ripple(cfg, tmp_path / "r.yaml", max_torque=0.25), model)
    s = settle(sim, ctl, 300)
    assert np.allclose(s.tau_r, [0, 0, 0.25, 0, 0.1, 0])
    assert ctl.free_drive()[0]
    s = settle(sim, ctl, 30)
    assert 0.0 < s.tau_r[J3] < 0.25, "it fades with the gains"
    s = settle(sim, ctl, int(cfg.free_drive.transition_duration * 500) + 10)
    assert s.mode == Mode.FREE_DRIVE.value and np.all(s.tau_r == 0.0) and ctl.feedforward()["ripple"] is True
    assert ctl.hold()[0]
    s = settle(sim, ctl, int(cfg.free_drive.transition_duration * 500) + 300)
    assert s.tau_r[J3] == pytest.approx(0.25), "back in hold it returns"
    ctl.shutdown()


def test_reload_gains_reads_the_ripple_file_again(cfg, model, tmp_path):
    path = tmp_path / "r.yaml"
    on = _with_ripple(cfg, path)
    sim, ctl = _rig(on, model)
    assert np.all(settle(sim, ctl, 300).tau_r == 0.0) and ctl.gains()["ripple"]["joints"] == [], "no file yet"
    _flat_table(path)
    ok, msg = ctl.reload_gains(on)
    assert ok and "ripple table: J3 0.500..1.600 rad, J5" in msg
    assert settle(sim, ctl, 50).tau_r[J3] == pytest.approx(0.3)
    path.write_text("J3:\n  start: 0.5\n  step: 0.1\n  forward: [0, 1, 0]\n  reverse: [0, 1]\n")
    ok, msg = ctl.reload_gains(on)
    assert not ok and "gains not changed" in msg and settle(sim, ctl, 5).tau_r[J3] == pytest.approx(0.3)
    ok, msg = ctl.reload_gains(_with_ripple(cfg, path, lead_time=1.0))
    assert not ok and "lead_time" in msg
    path.unlink()
    assert ctl.reload_gains(on)[0] and np.all(settle(sim, ctl, 5).tau_r == 0.0)
    ctl.shutdown()


# ── record, fit, play again: the whole loop on a simulated arm with a ripple ───────────────

def _j3_strokes():
    a, b = L_POSE.copy(), L_POSE.copy()
    a[J3], b[J3] = 0.78, 1.10
    prog = Program()
    for q in (b, a, b, a):
        prog.add(TaughtPoint(q=q, gripper=0.0, speed=1.0))
    return a, prog


def _play(cfg, model, speed):
    """Play J3 back and forth on an arm whose J3 has a ripple; return the recording of J3 and,
    while it moves, the RMS of the ripple in its position (rad) and in the observer's estimate
    (N·m): what is left of each after a smooth curve along the angle is taken out."""
    start, prog = _j3_strokes()
    amp = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    sim = SimBackend(realtime=False, q0=start, model=model, ripple=amp, ripple_friction=0.4 * amp)
    ctl = TeachController(sim, cfg)
    ctl.start()
    s = settle(sim, ctl, 300)
    traj = plan_program(ctl.model, prog, s.q_target, cfg, gripper_start=s.gripper_target)
    assert ctl.play(traj, speed=speed)[0]
    seq = ctl.trace_since(J3, None, 1)["seq"]
    while ctl.snapshot().mode == Mode.PLAYBACK.value:
        sim.step(1)
    tr = ctl.trace_since(J3, seq, 10 ** 6)
    ctl.shutdown()
    rec = Recording(J3, *(np.array(tr[k], dtype=float) for k in ("t", "q_cmd", "q", "tau_meas")))
    v = np.gradient(rec.q_cmd, rec.t)
    ripple = {"err": [], "tau_d": []}
    for sign in (1.0, -1.0):
        m = sign * v > 0.8 * np.abs(v).max()
        x = rec.q[m] - rec.q[m].mean()
        for key, y in (("err", rec.q - rec.q_cmd), ("tau_d", np.array(tr["tau_d"], dtype=float))):
            ripple[key].append(np.std(y[m] - np.polyval(np.polyfit(x, y[m], 3), x)))
    return rec, float(np.mean(ripple["err"])), float(np.mean(ripple["tau_d"]))


def test_table_learned_from_a_slow_recording_takes_the_ripple_out(cfg, model, tmp_path):
    path = tmp_path / "ripple.yaml"
    off, on = _with_ripple(cfg, path, on=False), _with_ripple(cfg, path)
    slow, before_slow, _ = _play(off, model, 0.25)
    _, before_fast, _ = _play(off, model, 0.5)
    assert before_slow > 1.2e-3 and before_fast > 0.5e-3, "the simulated ripple shows in the position"

    fit = fit_table([slow])
    assert fit.explained["forward"] > 0.9 and fit.explained["reverse"] > 0.9
    RippleMap(6, {J3: fit.table}).save(path)
    again, first_slow, _ = _play(on, model, 0.25)
    assert first_slow < 0.5 * before_slow, f"one pass: {before_slow * 1e3:.2f} -> {first_slow * 1e3:.2f} mrad"

    # A recording made with the table in use gives the next table.
    RippleMap(6, {J3: fit_table([again]).table}).save(path)
    _, second_slow, _ = _play(on, model, 0.25)
    _, second_fast, _ = _play(on, model, 0.5)
    assert second_slow < 0.2 * before_slow, f"two passes: {before_slow * 1e3:.2f} -> {second_slow * 1e3:.2f} mrad"
    assert second_fast < 0.4 * before_fast, "and it holds at twice the speed it was learned at"

    # The observer alone answers the ripple too late to remove it; with the table it must not
    # cancel the ripple a second time.
    obs = dataclasses.replace(cfg.feedforward, observer=True)
    _, observer_only, tau_d_alone = _play(dataclasses.replace(off, feedforward=obs), model, 0.5)
    _, both, tau_d_both = _play(dataclasses.replace(on, feedforward=dataclasses.replace(obs, ripple=True)), model, 0.5)
    assert observer_only > 0.6 * before_fast, "the observer alone leaves most of it"
    assert both < 1.3 * second_fast + 2e-5, "together: no worse than the table alone"
    assert tau_d_both < 0.4 * tau_d_alone, "the observer's estimate no longer carries the ripple the table covers"
