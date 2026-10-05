import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_teaching.backend.sim import SimBackend
from robot_teaching.controller import TeachController
from robot_teaching.program import ProgramStore
from robot_teaching.server import create_app

from conftest import L_POSE


@pytest.fixture
def client(cfg, model, tmp_path):
    sim = SimBackend(realtime=False, q0=L_POSE, model=model)
    ctl = TeachController(sim, cfg)
    app = create_app(ctl, cfg, ProgramStore(tmp_path), backend_name="sim", model=model)
    with TestClient(app) as c:
        c.sim = sim
        c.ctl = ctl
        sim.step(200)
        yield c


def test_config_and_state(client):
    cfg = client.get("/api/config").json()
    assert cfg["n_joints"] == 6 and cfg["joint_names"][0] == "joint1" and cfg["backend"] == "sim"
    st = client.get("/api/state").json()
    assert st["state"]["mode"] == "hold" and len(st["state"]["q"]) == 6
    assert client.get("/").status_code == 200


def test_record_edit_and_play_flow(client):
    # Record a point in hold, jog, record another, then play.
    r = client.post("/api/program/points", json={"motion": "joint", "speed": 0.8}).json()
    assert r["point"]["name"] == "P1" and r["point"]["pose"]["xyz"]
    assert client.post("/api/jog/joint", json={"joint": 0, "delta": 0.3}).status_code == 200
    client.sim.step(700)
    r2 = client.post("/api/program/points", json={"motion": "linear", "speed": 0.5, "dwell": 0.2}).json()
    assert abs(r2["point"]["q"][0] - 0.3) < 1e-6
    pid = r2["point"]["id"]
    p = client.patch(f"/api/program/points/{pid}", json={"name": "place", "gripper": 2.0}).json()["point"]
    assert p["name"] == "place" and p["gripper"] == 2.0
    prog = client.get("/api/program").json()
    assert len(prog["points"]) == 2 and prog["dirty"] and prog["problems"] == []
    client.post("/api/program/reorder", json={"ids": [pid, prog["points"][0]["id"]]})
    assert client.get("/api/program").json()["points"][0]["id"] == pid
    plan = client.post("/api/playback/plan", json={"loop": False}).json()
    assert plan["duration"] > 0 and len(plan["point_times"]) == 2
    r = client.post("/api/playback/start", json={"speed": 1.0, "loop": False})
    assert r.status_code == 200, r.text
    ticks, gripper_seen = 0, set()
    while True:
        st = client.get("/api/state").json()["state"]
        gripper_seen.add(st["gripper_target"])
        if st["mode"] != "playback":
            break
        client.sim.step(50)
        ticks += 50
        assert ticks < 30000
    # Order after reorder: [place (gripper 2.0), P1 (gripper 0.0)] → 2.0 was applied, then 0.0 at the end.
    assert 2.0 in gripper_seen and st["gripper_target"] == 0.0
    assert np.max(np.abs(np.array(st["q_target"]) - np.array(prog["points"][0]["q"]))) < 1e-6
    # Save / list / load.
    assert client.post("/api/programs/demo/save").status_code == 200
    assert client.get("/api/programs").json()["programs"][0]["name"] == "demo"
    assert client.post("/api/program/new", json={"name": "scratch"}).json()["points"] == []
    assert len(client.post("/api/programs/demo/load").json()["points"]) == 2
    assert client.post("/api/programs/bad name!/save").status_code == 400


def test_mode_rules_and_errors(client):
    assert client.post("/api/mode", json={"mode": "free_drive"}).status_code == 200
    client.sim.step(10)
    assert client.post("/api/jog/joint", json={"joint": 0, "delta": 0.1}).status_code == 409
    assert client.post("/api/playback/start", json={}).status_code == 409
    client.post("/api/mode", json={"mode": "hold"})
    client.sim.step(400)
    assert client.post("/api/playback/start", json={}).status_code == 400          # empty program
    assert client.post("/api/jog/joint", json={"joint": 9, "delta": 0.1}).status_code == 400
    assert client.post("/api/gripper", json={"action": "open"}).status_code == 200
    assert client.post("/api/gripper", json={}).status_code == 400
    assert client.post("/api/estop").status_code == 200
    client.sim.step(5)
    assert client.get("/api/state").json()["state"]["mode"] == "disabled"
    assert client.post("/api/enable").status_code == 200


def test_cartesian_jog(client):
    before = client.get("/api/state").json()["state"]["pose"]["xyz"]
    assert client.post("/api/jog/cartesian", json={"axis": "z", "delta": 0.02, "frame": "base"}).status_code == 200
    client.sim.step(800)
    after = client.get("/api/state").json()["state"]["pose"]["xyz"]
    assert after[2] - before[2] == pytest.approx(0.02, abs=2e-3)
    assert abs(after[0] - before[0]) < 2e-3 and abs(after[1] - before[1]) < 2e-3
    assert client.post("/api/jog/cartesian", json={"axis": "x", "delta": 5.0}).status_code == 409


def test_websocket_stream(client):
    with client.websocket_connect("/ws/state") as ws:
        msg = ws.receive_json()
    assert "state" in msg and "program_rev" in msg and msg["state"]["mode"] == "hold"
