# Offline validation of the trajectory planner

**Why.** Playback on the arm showed a wobble and an undershoot on the descent to the
fixture. The chain is planner → controller → motors and bus. The planner is the only
link that can be tested completely without hardware, so it was validated first.

**How.** `tools/validate_plan.py` plans a saved program exactly as the Play button does,
then checks the sampled trajectory and plots it:

```bash
uv run --with matplotlib python tools/validate_plan.py programs/<name>.json --out /tmp/plan
uv run python tools/validate_plan.py programs/<name>.json          # checks only
```

| check | what passes |
|---|---|
| timing | samples start at 0 and are strictly increasing |
| arrival | the trajectory lands on every taught joint configuration at its reported time |
| consistency | the stored velocity equals the finite difference of the stored positions |
| continuity | no position or velocity jump between samples (none larger than 1.5× the limits allow) |
| limits | per-segment peak velocity and acceleration stay within the limits scaled by the point speed |
| linear | linear moves stay on the straight chord and on the orientation geodesic |
| stream | the 500 Hz command stream (linear interpolation of the 100 Hz plan) adds no visible error |
| dt-robust | planning again with 2 ms sample spacing gives the same arrival times |

The plots show, per joint, position / velocity / acceleration / jerk against time, the
tool position and speeds, the distance of each linear move from its chord, and what the
500 Hz stream looks like next to the planned samples.

## Findings on `programs/example_pick_place.json` (2026-10-06)

**The planner is smooth and correct.** Seven of the eight checks pass: velocity matches
the finite difference of position, acceleration is continuous with one sign change per
rest-to-rest move, peaks respect the limits, linear moves are straight to 1e-5 mm, every
move ends exactly on the taught configuration, and the 500 Hz stream is off by at most
12 µrad. The plan is therefore not the source of the wobble; the next blocks to test are
the controller and the hardware.

**One weakness: linear-move re-timing is contaminated by IK noise** (`dt-robust` fails).
A linear move is sampled every 10 ms, each sample is solved by the closed-loop IK, and the
planner differentiates the result twice to check the joint velocity and acceleration
limits. The IK stops iterating once the pose error is under its tolerance of 1e-4, so
during the slow start and end of a move it often does zero iterations and the joint path
becomes a staircase with steps of about 150 µrad. Differenced twice, those steps look
like acceleration. Consequences:

* On the example descent the apparent acceleration is 0.58 of the budget; the true value
  is 0.27. The check is pessimistic, never permissive, so safety is not reduced.
* A linear move at a low point speed can be slowed below what was asked, or refused with
  the misleading message "linear move cannot satisfy the joint velocity limits (near a
  singularity?)". At 2 ms spacing the example program is refused outright.
* The velocity feed-forward on linear moves carries spikes of about 0.1 N·m through kd:
  harmless, and far too small to explain the wobble.

Planning the example with tighter IK tolerances (the planner's `_CLIK` parameters in
`src/robot_teaching/planning/cartesian.py`):

| IK tolerance | step left by the IK | apparent acceleration, descent | plans at 2 ms spacing | planning time |
|---|---|---|---|---|
| 1e-4 (current) | 152 µrad | 0.58 of budget | no | 0.17 s |
| 1e-6 | 17 µrad | 0.28 | yes | 0.38 s |
| 1e-7 | 16 µrad | 0.27 | yes | 0.55 s |

The planned durations are identical in every case.

**Fix applied (2026-10-06).** The IK tolerance is now 1e-6 (`_CLIK` in
`src/robot_teaching/planning/cartesian.py`). The same change also replaced the linear-move
timing: a block of linear points (blended through pass-through points) now runs under a
velocity-limit curve (Cartesian speed, angular speed, centripetal acceleration in the
blends) with forward/backward passes at the Cartesian acceleration limit, sampled with the
constant-acceleration law inside each 2 mm cell on a uniform time grid. That removed a
second, larger source of apparent acceleration: the old sampling left a speed step of about
0.017 m/s out of rest at both ends of a move and a 2 ms last sample, which the forced zero
end velocity turned into a spike of tens of rad/s². The cleaner long-term option remains to
compute the joint velocity analytically from the Cartesian velocity through the Jacobian
instead of differencing the IK output.

## Time law of linear moves (2026-10-15)

**Why.** The 6 October rebuild timed every linear block with forward and backward passes at the
constant Cartesian acceleration limit, a trapezoidal speed profile. Its acceleration switches on
and off within one sample at the four ends of the ramps, the jerk a trapezoid is known for.
Before that rebuild `sample_straight_line` had used the minimum-jerk quintic, which joint moves
still use. On the arm the jerk is felt at the start and stop of linear moves.

**Measured on the example.** The new `acc step` column of `tools/validate_plan.py` is the largest
change of the joint acceleration between two consecutive 10 ms samples, relative to the segment's
acceleration budget. Joint moves are unchanged; what remains on them is the numerical boundary with
the hold pieces.

| move | motion | acc step before | acc step after | T before [s] | T after [s] |
|---|---|---|---|---|---|
| above pick | joint | 0.074 | 0.044 | 2.48 | 2.48 |
| pick | linear | 0.275 | 0.016 | 1.97 | 2.75 |
| lift | linear | 0.254 | 0.021 | 2.50 | 2.96 |
| place | linear | 0.255 | 0.016 | 1.97 | 2.75 |
| retreat | linear | 0.254 | 0.021 | 2.50 | 2.96 |

On the pick descent the Cartesian acceleration used to reach its full budget (0.09 m/s² at speed
0.3) in two samples; it now rises linearly from zero, 0.003 m/s² per sample.

**Fix applied.** A linear block with a single waypoint, which is every linear move that is not
blended through a pass-through point, uses the rest-to-rest quintic in path length again. Its
duration comes from the Cartesian linear and angular velocity and acceleration limits
(`min_jerk_duration`, the same rule as joint moves). Blended blocks keep the velocity-limit curve
with constant-acceleration ramps, because a corner needs a mid-path speed dip that one quintic
cannot produce; a jerk-limited ramp there is the next step. The quintic is slower than the
trapezoid under the same limits (it never cruises at the speed limit): the example's linear moves
take 0.5-0.8 s longer each and the program 19.6 s instead of 17.1 s. All checks pass, and the IK
staircase on linear moves drops from 13 to 3 µrad because the move leaves rest more gently.
`tests/test_planning.py::test_linear_segment_time_law_is_minimum_jerk` pins the time law.
