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
