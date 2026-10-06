"""Trajectory planning for playback.

* :mod:`profiles`      quintic Hermite polynomials (minimum-jerk when boundary
                       velocities and accelerations are zero).
* :mod:`joint_spline`  multi-waypoint joint-space planner, time-scaled until
                       per-joint velocity and acceleration limits hold.
* :mod:`cartesian`     straight-line end-effector moves, blended through
                       pass-through points, tracked with the upstream closed-loop IK.
* :mod:`playback`      turns a :class:`~robot_teaching.program.Program` into one
                       time-parameterized :class:`Trajectory` with gripper events.
"""

from .trajectory import Trajectory, TrajectoryEvent, PlanningError
from .profiles import quintic_coeffs, quintic_eval, min_jerk_duration
from .joint_spline import plan_joint_path, JointPathResult
from .cartesian import LinearBlockResult, plan_linear_block, plan_linear_segment
from .playback import plan_program

__all__ = [
    "Trajectory",
    "TrajectoryEvent",
    "PlanningError",
    "quintic_coeffs",
    "quintic_eval",
    "min_jerk_duration",
    "plan_joint_path",
    "JointPathResult",
    "plan_linear_segment",
    "plan_linear_block",
    "LinearBlockResult",
    "plan_program",
]
