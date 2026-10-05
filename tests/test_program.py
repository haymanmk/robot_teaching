import json

import numpy as np
import pytest

from robot_teaching.program import Program, ProgramStore, TaughtPoint


def test_point_validation():
    with pytest.raises(ValueError):
        TaughtPoint(q=[0] * 6, speed=0.0)
    with pytest.raises(ValueError):
        TaughtPoint(q=[0] * 6, motion="spline")
    p = TaughtPoint(q=[0] * 6, motion="linear", blend=True)
    assert p.blend is False, "linear points always stop"


def test_program_roundtrip(tmp_path):
    prog = Program(name="demo")
    prog.add(TaughtPoint(q=[0, 0.7, 1.1, 0, 0, 0], gripper=0.0))
    prog.add(TaughtPoint(q=[0.2, 0.8, 1.2, 0, 0, 0], gripper=2.0, motion="linear", speed=0.3, dwell=0.5), index=0)
    assert [p.name for p in prog.points] == ["P2", "P1"]
    path = prog.save(tmp_path / "demo.json")
    loaded = Program.load(path)
    assert loaded.to_dict()["points"] == prog.to_dict()["points"]
    assert json.loads(path.read_text())["format_version"] == 1


def test_program_edit_ops():
    prog = Program()
    a = prog.add(TaughtPoint(q=[0] * 6))
    b = prog.add(TaughtPoint(q=[0.1] * 6))
    c = prog.add(TaughtPoint(q=[0.2] * 6))
    prog.reorder([c.id, a.id, b.id])
    assert [p.id for p in prog.points] == [c.id, a.id, b.id]
    prog.move(b.id, 0)
    assert prog.points[0].id == b.id
    prog.remove(a.id)
    assert len(prog.points) == 2
    with pytest.raises(KeyError):
        prog.find(a.id)
    with pytest.raises(ValueError):
        prog.reorder([b.id])


def test_validate_against_limits(model):
    prog = Program()
    prog.add(TaughtPoint(q=[0, 0.7, 1.1, 0, 0, 0]))
    prog.add(TaughtPoint(q=[0, -0.5, 1.1, 0, 0, 0]))          # joint2 below 0
    prog.add(TaughtPoint(q=[0, 0.7, 1.1]))                    # wrong length
    problems = prog.validate(model.lower, model.upper, 0.02)
    assert len(problems) == 2
    assert "joint 2" in problems[0]
    assert "expected 6" in problems[1]


def test_store(tmp_path):
    store = ProgramStore(tmp_path / "programs")
    prog = Program(name="x")
    prog.add(TaughtPoint(q=[0] * 6))
    store.save(prog, "pick place-1")
    assert [p["name"] for p in store.list()] == ["pick place-1"]
    assert len(store.load("pick place-1").points) == 1
    with pytest.raises(ValueError):
        store.save(prog, "../escape")
    with pytest.raises(FileNotFoundError):
        store.load("missing")
    store.delete("pick place-1")
    assert store.list() == []
