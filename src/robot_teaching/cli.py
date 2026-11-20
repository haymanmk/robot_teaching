"""Command line entry point: ``robot-teaching serve``, ``plan`` and ``ripple-fit``."""

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
        ripple = np.array([float(v) for v in args.sim_ripple.split(",")]) if args.sim_ripple else 0.0
        backend = make_backend("sim", q0=q0, rate=cfg.control.rate, ripple=ripple)
    else:
        hw_yaml = args.hw_yaml or cfg.hardware_yaml
        if hw_yaml and Path(hw_yaml).expanduser().exists():
            hw_yaml = str(Path(hw_yaml).expanduser().resolve())      # the vendor resolves bare names in its own dir
        backend = make_backend("rebotarm", hw_yaml=hw_yaml,
                               can_bring_up=cfg.can.bring_up and not args.no_can_up, can_bitrate=cfg.can.bitrate)
    controller = TeachController(backend, cfg)
    store = ProgramStore(cfg.programs_path())
    connect = args.connect or ("auto" if args.backend == "sim" else "manual")
    app = create_app(controller, cfg, store, backend_name=args.backend, connect_on_start=(connect == "auto"))
    print(f"[robot-teaching] backend={args.backend} rate={controller.rate:.0f} Hz programs={store.dir}")
    print(f"[robot-teaching] config: {cfg.source_path or 'defaults'}"
          + (f" + {cfg.local_path}" if cfg.local_path else " (no local override)")
          + (f"; hardware YAML: {args.hw_yaml or cfg.hardware_yaml or 'vendor default'}" if args.backend == "rebotarm" else ""))
    print(f"[robot-teaching] connection: {connect}" + ("" if connect == "auto" else " (press Connect in the UI or POST /api/connect)"))
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
        np.savetxt(out, np.column_stack([traj.t, traj.q, traj.qd, traj.qdd]), delimiter=",",
                   header="t," + ",".join(f"q{i+1}" for i in range(traj.n_joints)) + "," +
                          ",".join(f"qd{i+1}" for i in range(traj.n_joints)) + "," +
                          ",".join(f"qdd{i+1}" for i in range(traj.n_joints)), comments="")
        print(f"wrote {out}")
    return 0


def _ripple_fit(args: argparse.Namespace) -> int:
    from .config import load_config
    from .ripple import DIRECTIONS, RippleMap, fit_table, read_recording

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else cfg.ripple_path()
    if out is None:
        print("ripple.file is empty in the configuration: give --out", file=sys.stderr)
        return 1
    try:
        recordings = [read_recording(p) for p in args.recordings]
        if args.sine_per_rev is not None and args.sine_per_rev <= 0.0:
            raise ValueError("--sine-per-rev must be positive")
        fit = fit_table(recordings, step=args.step, smooth=args.smooth, sine=not args.no_sine,
                        sine_cycles=None if args.sine_per_rev is None else args.sine_per_rev / (2.0 * np.pi))
        tables = RippleMap.load(out, 64)                  # the other joints' tables are kept
    except (OSError, ValueError) as e:
        print(f"ripple-fit failed: {e}", file=sys.stderr)
        return 1
    t = fit.table
    print(f"J{fit.joint + 1}: {len(recordings)} recording(s), top commanded speed {fit.speed:.3f} rad/s")
    print(f"  table {t.start:.3f} .. {t.stop:.3f} rad in steps of {t.step * 1e3:g} mrad ({len(t.forward)} values per direction)")
    for name in DIRECTIONS:
        peak = float(np.max(np.abs(getattr(t, name))))
        print(f"  {name:8s} {fit.rms[name]:.3f} N·m RMS, peak {peak:.3f} N·m, from {fit.samples[name]} samples; "
              f"it reproduces {100.0 * fit.explained[name]:.0f} % of the motor torque's fast variation")
        if peak > float(cfg.ripple.max_torque):
            print(f"           the peak is above ripple.max_torque ({cfg.ripple.max_torque:g} N·m): the term will be clipped")
    if fit.sine is not None:
        s = fit.sine
        print(f"  sinusoid {s.cycles:.2f} cycles per rad ({s.per_revolution:.1f} per revolution), amplitude "
              f"{s.amplitude('forward'):.3f} N·m forward and {s.amplitude('reverse'):.3f} N·m reverse; "
              f"it reproduces {100.0 * fit.sine_share:.0f} % of the measured ripple")
        if t.sine is not None:
            print("           the table continues as it outside its range and where nothing was recorded")
        else:
            print("           too little to rely on: the table is zero outside its range and where nothing was recorded")
    elif not args.no_sine:
        print("  no sinusoid (too short a stretch to fit one): the table is zero outside its range")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    replaced = fit.joint in tables.tables
    tables.tables[fit.joint] = t
    out.parent.mkdir(parents=True, exist_ok=True)
    tables.save(out)
    print(f"{'replaced' if replaced else 'added'} the table of J{fit.joint + 1} in {out}")
    print("press Reload gains in the pendant (or restart the server), then tick the ripple feed-forward")
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
    s.add_argument("--no-can-up", action="store_true",
                   help="do not bring the SocketCAN interface up on connect (overrides can.bring_up in teaching.yaml)")
    s.add_argument("--sim-q0", default="0,0.7,1.1,0,0,0", help="initial joint angles for the simulator, comma separated (rad)")
    s.add_argument("--sim-ripple", default=None,
                   help="simulator only: amplitude of an angle-locked torque ripple in N·m, one value for every "
                        "joint or one per joint, comma separated (e.g. 0,0,0.5,0,0,0), to try the ripple feed-forward")
    s.add_argument("--connect", choices=["auto", "manual"], default=None,
                   help="connect to the arm at startup (auto) or from the UI's Connect button (manual); "
                        "default: auto for sim, manual for rebotarm")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=_serve)

    p = sub.add_parser("plan", help="plan a saved program offline and print the summary")
    p.add_argument("program", help="program JSON file")
    p.add_argument("--start", default=None, help="start configuration, comma separated (rad); default: first point")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--csv", default=None, help="write the sampled trajectory to this CSV file")
    p.add_argument("--config", default=None)
    p.set_defaults(func=_plan)

    r = sub.add_parser("ripple-fit", help="learn a joint's ripple feed-forward table from scope recordings",
                       description="Learn the ripple feed-forward table of one joint from recordings made with the "
                                   "scope's Record button while that joint moves back and forth slowly, and store it "
                                   "in the ripple file of the configuration. A recording made with the table in use "
                                   "gives the next, better table. The fit also finds the sinusoid of the joint "
                                   "angle closest to the ripple; the table continues as that sinusoid at the "
                                   "angles the recordings do not cover.")
    r.add_argument("recordings", nargs="+", help="scope CSV files of one joint")
    r.add_argument("--out", default=None, help="file to write (default: ripple.file of the configuration)")
    r.add_argument("--step", type=float, default=0.002, help="rad between table values (default 0.002)")
    r.add_argument("--smooth", type=float, default=0.05,
                   help="rad; variation slower than about a cycle per 4 x this is left out of the table (default 0.05)")
    r.add_argument("--no-sine", action="store_true",
                   help="no sinusoid: the table is zero at the angles the recordings do not cover")
    r.add_argument("--sine-per-rev", type=float, default=None,
                   help="the sinusoid's cycles per revolution of the joint, where it is known (e.g. 126); "
                        "default: searched for")
    r.add_argument("--dry-run", action="store_true", help="print the result without writing the file")
    r.add_argument("--config", default=None)
    r.set_defaults(func=_ripple_fit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
