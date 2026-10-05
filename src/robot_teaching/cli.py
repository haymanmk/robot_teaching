"""Command line entry point: ``robot-teaching serve`` and ``robot-teaching plan``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .backend import make_backend
    from .config import load_config
    from .controller import TeachController
    from .program import ProgramStore
    from .server import create_app

    cfg = load_config(args.config)
    if args.backend == "sim":
        q0 = np.array([float(v) for v in args.sim_q0.split(",")]) if args.sim_q0 else None
        backend = make_backend("sim", q0=q0, rate=cfg.control.rate)
    else:
        backend = make_backend("rebotarm", hw_yaml=args.hw_yaml)
    controller = TeachController(backend, cfg)
    store = ProgramStore(cfg.programs_path())
    app = create_app(controller, cfg, store, backend_name=args.backend)
    print(f"[robot-teaching] backend={args.backend} rate={controller.rate:.0f} Hz programs={store.dir}")
    print(f"[robot-teaching] UI: http://{args.host}:{args.port}/   API docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _plan(args: argparse.Namespace) -> int:
    from .config import load_config
    from .model import RobotModel
    from .planning import PlanningError, plan_program
    from .program import Program

    cfg = load_config(args.config)
    model = RobotModel()
    program = Program.load(args.program)
    q_start = np.array([float(v) for v in args.start.split(",")]) if args.start else program.points[0].q_array()
    try:
        traj = plan_program(model, program, q_start, cfg, gripper_start=None, loop=args.loop)
    except PlanningError as e:
        print(f"planning failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(traj.summary(), indent=2))
    if args.csv:
        out = Path(args.csv)
        np.savetxt(out, np.column_stack([traj.t, traj.q, traj.qd]), delimiter=",",
                   header="t," + ",".join(f"q{i+1}" for i in range(traj.n_joints)) + "," +
                          ",".join(f"qd{i+1}" for i in range(traj.n_joints)), comments="")
        print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot-teaching", description="Teach-and-playback interface for the reBot Arm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web UI and control loop")
    s.add_argument("--backend", choices=["sim", "rebotarm"], default="sim")
    s.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 to reach it from another device)")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--config", default=None, help="path to teaching.yaml")
    s.add_argument("--hw-yaml", default=None, help="upstream hardware YAML (default from rebotarm.yaml)")
    s.add_argument("--sim-q0", default="0,0.7,1.1,0,0,0", help="initial joint angles for the simulator, comma separated (rad)")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=_serve)

    p = sub.add_parser("plan", help="plan a saved program offline and print the summary")
    p.add_argument("program", help="program JSON file")
    p.add_argument("--start", default=None, help="start configuration, comma separated (rad); default: first point")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--csv", default=None, help="write the sampled trajectory to this CSV file")
    p.add_argument("--config", default=None)
    p.set_defaults(func=_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
