"""Turn a program into one time-parameterized trajectory.

Walking the program from the arm's current configuration:

* consecutive ``joint`` points are planned together as one spline, so a point
  with ``blend`` is passed through without stopping;
* consecutive ``linear`` points are planned together as one Cartesian path:
  a point with ``blend`` is passed through with a corner blend, the others
  are reached on a straight line and stopped at;
* a point stops whenever it has a dwell, changes the gripper target, or is
  followed by a point of the other motion type;
* after arriving, the gripper target is emitted as an event and the arm holds
  still for the gripper settle time and the dwell.

With ``loop=True`` a closing joint move back to the first point is appended
and ``loop_start_time`` marks where a looping playback restarts.
"""

from __future__ import annotations

import numpy as np

from ..config import TeachingConfig, resolve_vector
from ..model import RobotModel
from ..program import Program, TaughtPoint
from .cartesian import plan_linear_block
from .joint_spline import plan_joint_path
from .trajectory import PlanningError, Trajectory, TrajectoryEvent


def _hold(q: np.ndarray, duration: float, dt: float) -> Trajectory | None:
    if duration <= 0.0:
        return None
    t = np.arange(0.0, duration, dt)
    if len(t) == 0 or duration - t[-1] > 1e-9:
        t = np.append(t, duration)
    n = len(q)
    return Trajectory(t, np.tile(q, (len(t), 1)), np.zeros((len(t), n)))


def plan_program(
    model: RobotModel,
    program: Program,
    q_start: np.ndarray,
    cfg: TeachingConfig,
    gripper_start: float | None = None,
    start_index: int = 0,
    loop: bool = False,
) -> Trajectory:
    n = model.n
    lim = cfg.limits
    vmax = resolve_vector(lim.joint_velocity, n, "limits.joint_velocity")
    amax = resolve_vector(lim.joint_acceleration, n, "limits.joint_acceleration")
    dt = float(lim.planning_dt)
    margin = float(lim.joint_position_margin)

    points: list[TaughtPoint] = list(program.points[start_index:])
    if not points:
        raise PlanningError("the program has no points to play")
    problems = program.validate(model.lower, model.upper, margin)
    if problems:
        raise PlanningError("; ".join(problems))
    q_start = np.asarray(q_start, dtype=float)[:n]
    if not model.within_limits(q_start, 0.0):
        raise PlanningError("the arm is outside its joint limits; jog it back before playing")

    if loop and len(points) > 1:
        # Closing move: joint motion back to the first point with the first point's speed.
        first = points[0]
        points.append(TaughtPoint(q=first.q, gripper=first.gripper, name=f"{first.name} (loop)",
                                  motion="joint", speed=first.speed, blend=False, dwell=first.dwell))

    gripper_prev = gripper_start
    pieces: list[Trajectory] = []
    point_times: list[float] = []
    elapsed = 0.0
    q_cur = q_start.copy()

    joint_block: list[TaughtPoint] = []
    joint_stops: list[bool] = []

    def flush_joint_block() -> None:
        nonlocal q_cur, elapsed, joint_block, joint_stops
        if not joint_block:
            return
        waypoints = np.vstack([q_cur] + [p.q_array() for p in joint_block])
        stops = [True] + [bool(s) for s in joint_stops]
        speeds = [p.speed for p in joint_block]
        res = plan_joint_path(waypoints, stops, vmax, amax, dt, speeds)
        traj = Trajectory(res.t, res.q, res.qd)
        pieces.append(traj)
        point_times.extend(elapsed + float(res.knot_times[k + 1]) for k in range(len(joint_block)))
        elapsed += traj.duration
        q_cur = waypoints[-1].copy()
        joint_block, joint_stops = [], []

    def stops_at(idx: int) -> bool:
        """A point stops unless it is a pass-through whose gripper and motion type continue unchanged."""
        p = points[idx]
        nxt = points[idx + 1] if idx + 1 < len(points) else None
        gripper_changes = gripper_prev is None or abs(p.gripper - gripper_prev) > 1e-6
        return (not p.blend or p.dwell > 0.0 or gripper_changes or nxt is None or nxt.motion != p.motion)

    def after_arrival(p: TaughtPoint) -> None:
        """Gripper event + settle/dwell holds after reaching a stopping point."""
        nonlocal gripper_prev, elapsed
        hold_time = p.dwell
        if gripper_prev is None or abs(p.gripper - gripper_prev) > 1e-6:
            pieces[-1].events.append(TrajectoryEvent(pieces[-1].duration, "gripper", float(p.gripper)))
            hold_time += cfg.gripper.settle_time
            gripper_prev = p.gripper
        h = _hold(q_cur, hold_time, dt)
        if h is not None:
            pieces.append(h)
            elapsed += h.duration

    i = 0
    while i < len(points):
        p = points[i]
        if p.motion == "linear":
            flush_joint_block()
            j = i
            while not stops_at(j):        # extend the block over pass-through linear points
                j += 1
            block = points[i:j + 1]
            res = plan_linear_block(
                model, q_cur, [b.q_array() for b in block], vmax, amax,
                lim.cartesian_linear_velocity, lim.cartesian_linear_acceleration,
                lim.cartesian_angular_velocity, lim.cartesian_angular_acceleration,
                dt, speed=min(b.speed for b in block), blend_radius=float(lim.linear_blend_radius),
            )
            pieces.append(res.trajectory)
            point_times.extend(elapsed + t for t in res.arrival_times)
            elapsed += res.trajectory.duration
            q_cur = block[-1].q_array()
            after_arrival(block[-1])
            i = j + 1
            continue
        stop = stops_at(i)
        joint_block.append(p)
        joint_stops.append(stop)
        if stop:
            flush_joint_block()
            after_arrival(p)
        i += 1
    flush_joint_block()

    traj = Trajectory.concatenate(pieces)
    traj.point_times = point_times
    traj.events.extend(TrajectoryEvent(t, "point", i) for i, t in enumerate(point_times))
    traj.events.sort(key=lambda e: e.time)
    traj.loop_start_time = point_times[0] if loop and len(points) > 1 else 0.0
    return traj
