"""HTTP/WebSocket API for the teach pendant.

All command endpoints return ``{"ok": true, "message": ...}`` or raise an HTTP
error with a human-readable ``detail``. The WebSocket at ``/ws/state`` streams
the controller snapshot plus the program revision so the UI knows when to
refetch the program.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pinocchio as pin
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import TeachingConfig, resolve_vector
from ..controller import Mode, TeachController
from ..model import RobotModel
from ..planning import PlanningError, plan_program
from ..program import Program, ProgramStore, TaughtPoint

STATIC_DIR = Path(__file__).parent / "static"


# ── request models ────────────────────────────────────────────────────────────

class ModeRequest(BaseModel):
    mode: Literal["hold", "free_drive"]


class JointJog(BaseModel):
    joint: int = Field(ge=0)
    delta: float


class CartesianJog(BaseModel):
    axis: Literal["x", "y", "z", "roll", "pitch", "yaw"]
    delta: float
    frame: Literal["base", "tool"] = "base"


class GripperRequest(BaseModel):
    position: float | None = None
    action: Literal["open", "close", "hand"] | None = None


class RecordRequest(BaseModel):
    name: str | None = None
    motion: Literal["joint", "linear"] = "joint"
    speed: float | None = Field(default=None, gt=0, le=1)
    blend: bool = False
    dwell: float = Field(default=0.0, ge=0)
    index: int | None = None


class PointUpdate(BaseModel):
    name: str | None = None
    motion: Literal["joint", "linear"] | None = None
    speed: float | None = Field(default=None, gt=0, le=1)
    blend: bool | None = None
    dwell: float | None = Field(default=None, ge=0)
    gripper: float | None = None
    q: list[float] | None = None
    update_from_robot: bool = False


class ReorderRequest(BaseModel):
    ids: list[str]


class NewProgramRequest(BaseModel):
    name: str = "untitled"


class PlaybackRequest(BaseModel):
    speed: float | None = Field(default=None, gt=0, le=1)
    loop: bool = False
    start_index: int = Field(default=0, ge=0)


class SpeedRequest(BaseModel):
    speed: float = Field(gt=0, le=1)


class MoveToRequest(BaseModel):
    speed: float | None = Field(default=None, gt=0, le=1)


# ── application state ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self, controller: TeachController, cfg: TeachingConfig, store: ProgramStore,
                 model: RobotModel | None = None, backend_name: str = "") -> None:
        self.controller = controller
        self.cfg = cfg
        self.store = store
        self.model = model or RobotModel(controller.n)
        self.model_lock = threading.Lock()      # pinocchio Data is not thread safe
        self.program = Program()
        self.rev = 0
        self.dirty = False
        self.backend_name = backend_name
        self.last_plan: dict | None = None

    # -- helpers --------------------------------------------------------------

    def snap(self):
        s = self.controller.snapshot()
        if s is None:
            raise HTTPException(503, "controller not running")
        return s

    def require_connected(self) -> None:
        if not self.controller.is_connected:
            raise HTTPException(409, "not connected; press Connect first")

    def pose_dict(self, q) -> dict:
        with self.model_lock:
            return self.model.pose(np.asarray(q, float)).to_dict()

    def touch(self) -> None:
        self.rev += 1
        self.dirty = True
        self.program.touch()

    def current_q(self) -> np.ndarray:
        """Where the arm 'is' for teaching: the hold target in HOLD, the measurement otherwise."""
        s = self.snap()
        return s.q_target.copy() if s.mode == Mode.HOLD.value else s.q.copy()

    def gripper_value(self) -> float:
        s = self.snap()
        if s.gripper_target is not None:
            return float(s.gripper_target)
        if s.gripper is not None:
            return float(s.gripper)
        return float(self.cfg.gripper.closed_position)

    def gripper_bounds(self) -> tuple[float, float]:
        g = self.cfg.gripper
        return min(g.closed_position, g.open_position), max(g.closed_position, g.open_position)

    def plan(self, program: Program, loop: bool, start_index: int, cfg: TeachingConfig | None = None):
        cfg = cfg or self.cfg
        self.require_connected()
        s = self.snap()
        if s.mode != Mode.HOLD.value:
            raise HTTPException(409, "switch to hold mode before playing")
        if not s.gains_settled or s.jogging:
            raise HTTPException(409, "wait for the arm to settle")
        if start_index >= len(program.points):
            raise HTTPException(400, "start_index beyond the last point")
        try:
            with self.model_lock:
                return plan_program(self.model, program, s.q_target, cfg,
                                    gripper_start=s.gripper_target, start_index=start_index, loop=loop)
        except PlanningError as e:
            raise HTTPException(400, f"planning failed: {e}")


def _result(ok_msg: tuple[bool, str], conflict: bool = True) -> dict:
    ok, msg = ok_msg
    if not ok:
        raise HTTPException(409 if conflict else 400, msg)
    return {"ok": True, "message": msg}


# ── app factory ───────────────────────────────────────────────────────────────

def create_app(controller: TeachController, cfg: TeachingConfig, store: ProgramStore,
               backend_name: str = "", model: RobotModel | None = None, connect_on_start: bool = True) -> FastAPI:
    """Build the app. With ``connect_on_start=False`` the arm is connected from the UI (POST /api/connect)."""
    state = AppState(controller, cfg, store, model, backend_name)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if connect_on_start:
            controller.start()
        try:
            yield
        finally:
            if controller.is_connected:
                controller.shutdown()

    app = FastAPI(title="robot_teaching", version="0.1.0", lifespan=lifespan)
    app.state.teaching = state
    n = controller.n

    # ── info ──────────────────────────────────────────────────────────────
    @app.get("/api/config")
    def get_config():
        lim = cfg.limits
        return {
            "backend": state.backend_name,
            "n_joints": n,
            "joint_names": list(state.model.joint_names),
            "joint_lower": state.model.lower.tolist(),
            "joint_upper": state.model.upper.tolist(),
            "joint_margin": lim.joint_position_margin,
            "joint_velocity": resolve_vector(lim.joint_velocity, n, "v").tolist(),
            "joint_acceleration": resolve_vector(lim.joint_acceleration, n, "a").tolist(),
            "cartesian_linear_velocity": lim.cartesian_linear_velocity,
            "cartesian_angular_velocity": lim.cartesian_angular_velocity,
            "jog": {
                "joint_steps_deg": cfg.jog.joint_steps_deg,
                "cartesian_steps_m": cfg.jog.cartesian_steps_m,
                "cartesian_steps_deg": cfg.jog.cartesian_steps_deg,
            },
            "gripper": {
                "has_gripper": controller.backend.has_gripper,
                "closed_position": cfg.gripper.closed_position,
                "open_position": cfg.gripper.open_position,
            },
            "playback": {"default_speed": cfg.playback.default_speed},
            "park": {"pose": resolve_vector(cfg.park.pose, n, "park.pose").tolist(), "speed": cfg.park.speed},
            "control_rate": controller.rate,
            "connected": controller.is_connected,
        }

    @app.get("/api/state")
    def get_state():
        return {"state": state.snap().to_dict(), "program_rev": state.rev}

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        period = 1.0 / max(1.0, cfg.control.state_stream_rate)
        try:
            while True:
                s = controller.snapshot()
                if s is not None:
                    await ws.send_json({"state": s.to_dict(), "program_rev": state.rev})
                await asyncio.sleep(period)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ── connection ────────────────────────────────────────────────────────
    @app.post("/api/connect")
    def connect():
        """Open the bus, switch the motors to MIT mode, enable them and hold the current pose."""
        if controller.is_connected:
            return {"ok": True, "message": "already connected"}
        try:
            controller.start()
        except Exception as e:
            raise HTTPException(502, f"could not connect to the arm: {type(e).__name__}: {e}")
        return {"ok": True, "message": "connected, holding the current pose"}

    @app.post("/api/disconnect")
    def disconnect():
        """Stop the loop and close the bus. On the real arm this disables the motors: park first."""
        if not controller.is_connected:
            return {"ok": True, "message": "already disconnected"}
        s = state.snap()
        if s.mode == Mode.PLAYBACK.value:
            raise HTTPException(409, "stop playback before disconnecting")
        controller.shutdown()
        return {"ok": True, "message": "disconnected (motors off)"}

    @app.post("/api/park")
    def park(req: MoveToRequest):
        """Planned joint move to the configured rest pose, so the arm can be disconnected safely."""
        pose = resolve_vector(cfg.park.pose, n, "park.pose")
        single = Program(name="park", points=[TaughtPoint(
            q=pose.tolist(), gripper=state.gripper_value(), name="park", motion="joint",
            speed=req.speed if req.speed is not None else cfg.park.speed,
        )])
        # The rest pose may sit exactly on a joint limit (the URDF zero does), so plan it without the margin.
        no_margin = dataclasses.replace(cfg, limits=dataclasses.replace(cfg.limits, joint_position_margin=0.0))
        traj = state.plan(single, loop=False, start_index=0, cfg=no_margin)
        return _result(controller.play(traj, speed=1.0, loop=False)) | {"duration": traj.duration}

    # ── modes / safety ────────────────────────────────────────────────────
    @app.post("/api/mode")
    def set_mode(req: ModeRequest):
        state.require_connected()
        if req.mode == "hold":
            return _result(controller.hold())
        return _result(controller.free_drive())

    @app.post("/api/stop")
    def stop():
        state.require_connected()
        return _result(controller.stop_motion())

    @app.post("/api/estop")
    def estop():
        state.require_connected()
        return _result(controller.estop())

    @app.post("/api/enable")
    def enable():
        state.require_connected()
        return _result(controller.enable())

    # ── jogging ───────────────────────────────────────────────────────────
    @app.post("/api/jog/joint")
    def jog_joint(req: JointJog):
        if req.joint >= n:
            raise HTTPException(400, f"joint index must be < {n}")
        state.require_connected()
        s = state.snap()
        if s.mode != Mode.HOLD.value:
            raise HTTPException(409, "jogging is only available in hold mode")
        goal = s.q_target.copy()
        goal[req.joint] += req.delta
        goal = state.model.clamp(goal, cfg.limits.joint_position_margin)
        return _result(controller.jog_to(goal))

    @app.post("/api/jog/cartesian")
    def jog_cartesian(req: CartesianJog):
        state.require_connected()
        s = state.snap()
        if s.mode != Mode.HOLD.value:
            raise HTTPException(409, "jogging is only available in hold mode")
        q_seed = s.q_target.copy()
        with state.model_lock:
            T = state.model.fk(q_seed)
            R, p = T.rotation.copy(), T.translation.copy()
            if req.axis in ("x", "y", "z"):
                d = np.zeros(3)
                d["xyz".index(req.axis)] = req.delta
                p = p + (d if req.frame == "base" else R @ d)
            else:
                w = np.zeros(3)
                w[("roll", "pitch", "yaw").index(req.axis)] = req.delta
                dR = pin.exp3(w)
                R = dR @ R if req.frame == "base" else R @ dR
            res = state.model.ik(pin.SE3(R, p), q_seed)
        if not res.success:
            raise HTTPException(409, f"target pose not reachable (IK error {res.error:.2e})")
        q_goal = np.asarray(res.q, float)
        jump = float(np.max(np.abs(q_goal - q_seed)))
        if jump > cfg.jog.max_ik_joint_jump:
            raise HTTPException(409, f"IK solution jumps {jump:.2f} rad on a joint; use smaller steps")
        if not state.model.within_limits(q_goal, cfg.limits.joint_position_margin):
            raise HTTPException(409, "target pose is outside the joint limits")
        return _result(controller.jog_to(q_goal))

    # ── gripper ───────────────────────────────────────────────────────────
    @app.post("/api/gripper")
    def gripper(req: GripperRequest):
        if not controller.backend.has_gripper:
            raise HTTPException(400, "this arm has no gripper")
        state.require_connected()
        if req.action == "hand":
            return _result(controller.gripper_hand())
        if req.action == "open":
            pos = cfg.gripper.open_position
        elif req.action == "close":
            pos = cfg.gripper.closed_position
        elif req.position is not None:
            lo, hi = state.gripper_bounds()
            pos = float(np.clip(req.position, lo, hi))
        else:
            raise HTTPException(400, "give position or action")
        return _result(controller.set_gripper(pos))

    # ── program editing ───────────────────────────────────────────────────
    def program_payload() -> dict:
        d = state.program.to_dict()
        d["rev"] = state.rev
        d["dirty"] = state.dirty
        d["problems"] = state.program.validate(state.model.lower, state.model.upper, cfg.limits.joint_position_margin)
        return d

    @app.get("/api/program")
    def get_program():
        return program_payload()

    @app.post("/api/program/new")
    def new_program(req: NewProgramRequest):
        state.program = Program(name=ProgramStore.check_name(req.name))
        state.rev += 1
        state.dirty = False
        return program_payload()

    @app.post("/api/program/points")
    def record_point(req: RecordRequest):
        state.require_connected()
        q = state.current_q()
        if not state.model.within_limits(q, 0.0):
            raise HTTPException(409, "the arm is outside its joint limits")
        point = TaughtPoint(
            q=q.tolist(), gripper=state.gripper_value(), name=req.name or "", motion=req.motion,
            speed=req.speed if req.speed is not None else cfg.playback.default_speed,
            blend=req.blend, dwell=req.dwell, pose=state.pose_dict(q),
        )
        state.program.add(point, req.index)
        state.touch()
        return {"point": point.to_dict(), "rev": state.rev}

    @app.patch("/api/program/points/{point_id}")
    def update_point(point_id: str, req: PointUpdate):
        try:
            p = state.program.find(point_id)
        except KeyError:
            raise HTTPException(404, "no such point")
        if req.update_from_robot:
            state.require_connected()
            p.q = state.current_q().tolist()
            p.gripper = state.gripper_value()
            p.pose = state.pose_dict(p.q)
        if req.q is not None:
            if len(req.q) != n:
                raise HTTPException(400, f"q must have {n} values")
            p.q = [float(v) for v in req.q]
            p.pose = state.pose_dict(p.q)
        for name in ("name", "motion", "speed", "blend", "dwell", "gripper"):
            v = getattr(req, name)
            if v is not None:
                setattr(p, name, v)
        try:
            p.__post_init__()
        except ValueError as e:
            raise HTTPException(400, str(e))
        state.touch()
        return {"point": p.to_dict(), "rev": state.rev}

    @app.delete("/api/program/points/{point_id}")
    def delete_point(point_id: str):
        try:
            state.program.remove(point_id)
        except KeyError:
            raise HTTPException(404, "no such point")
        state.touch()
        return {"ok": True, "rev": state.rev}

    @app.post("/api/program/reorder")
    def reorder(req: ReorderRequest):
        try:
            state.program.reorder(req.ids)
        except ValueError as e:
            raise HTTPException(400, str(e))
        state.touch()
        return program_payload()

    @app.post("/api/program/points/{point_id}/move_to")
    def move_to_point(point_id: str, req: MoveToRequest):
        try:
            p = state.program.find(point_id)
        except KeyError:
            raise HTTPException(404, "no such point")
        single = Program(name="move_to", points=[TaughtPoint(
            q=p.q, gripper=p.gripper, name=p.name, motion="joint",
            speed=req.speed if req.speed is not None else p.speed, blend=False, dwell=0.0,
        )])
        traj = state.plan(single, loop=False, start_index=0)
        return _result(controller.play(traj, speed=1.0, loop=False)) | {"duration": traj.duration}

    # ── program files ─────────────────────────────────────────────────────
    @app.get("/api/programs")
    def list_programs():
        return {"programs": store.list(), "current": state.program.name, "dirty": state.dirty}

    @app.post("/api/programs/{name}/save")
    def save_program(name: str):
        try:
            store.save(state.program, name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        state.dirty = False
        state.rev += 1
        return {"ok": True, "name": state.program.name}

    @app.post("/api/programs/{name}/load")
    def load_program(name: str):
        try:
            prog = store.load(name)
        except FileNotFoundError:
            raise HTTPException(404, "no such program")
        except ValueError as e:
            raise HTTPException(400, str(e))
        for p in prog.points:
            if len(p.q) == n:
                p.pose = state.pose_dict(p.q)
        state.program = prog
        state.rev += 1
        state.dirty = False
        return program_payload()

    @app.delete("/api/programs/{name}")
    def delete_program(name: str):
        try:
            store.delete(name)
        except FileNotFoundError:
            raise HTTPException(404, "no such program")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    # ── playback ──────────────────────────────────────────────────────────
    @app.post("/api/playback/plan")
    def plan_only(req: PlaybackRequest):
        traj = state.plan(state.program, req.loop, req.start_index)
        state.last_plan = traj.summary()
        return state.last_plan

    @app.post("/api/playback/start")
    def playback_start(req: PlaybackRequest):
        traj = state.plan(state.program, req.loop, req.start_index)
        speed = req.speed if req.speed is not None else cfg.playback.default_speed
        state.last_plan = traj.summary()
        return _result(controller.play(traj, speed=speed, loop=req.loop)) | {"plan": state.last_plan}

    @app.post("/api/playback/speed")
    def playback_speed(req: SpeedRequest):
        return _result(controller.set_speed(req.speed))

    @app.post("/api/playback/stop")
    def playback_stop():
        return _result(controller.stop_motion())

    # ── UI ────────────────────────────────────────────────────────────────
    @app.middleware("http")
    async def revalidate_ui_files(request: Request, call_next):
        # The page, script and stylesheet change together; a browser must not run a
        # cached script against a newer page. "no-cache" forces an ETag revalidation.
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    if STATIC_DIR.exists():
        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
