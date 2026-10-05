# robot_teaching

Teach-by-demonstration interface for the **Seeed reBot Arm B601-RS** (RobStride motors,
CAN bus), built on Seeed's [`reBotArm_control_py`](https://github.com/Seeed-Projects/reBotArm_control_py).

* **Teach by dragging.** Free-drive mode is gravity-compensated impedance control: you pull the
  end-effector where you want it, let go, and the arm stays (an end-effector velocity lock
  holds it when still and follows when pushed).
* **Fine adjustment from the UI.** Joint jog and Cartesian jog (base or tool frame) with
  selectable step sizes, plus gripper open/close/slider.
* **Record taught points** with a motion type (joint or straight-line), speed, blend
  (pass-through) and dwell. Programs are versioned JSON files.
* **Play back with trajectory planning**, not joint-angle interpolation: joint moves are
  C² quintic splines through the via-points, time-scaled to per-joint velocity and
  acceleration limits; linear moves are straight lines in position with geodesic
  orientation, tracked by closed-loop IK and re-timed to the limits. Setpoints stream at
  500 Hz in MIT mode with gravity feed-forward. Speed override, loop and smooth stop.
* **Simulator** with the same interface as the hardware, so the UI and planner run and
  are tested without the arm.

## Layout

```
robot_teaching/
├── config/teaching.yaml            limits, free-drive gains, jog steps, gripper positions
├── programs/                       saved programs (JSON); example_pick_place.json included
├── src/robot_teaching/
│   ├── model.py                    Pinocchio model wrapper: FK, IK, Jacobian, g(q), limits
│   ├── program.py                  TaughtPoint / Program / ProgramStore (JSON)
│   ├── planning/                   quintic profiles, joint spline, Cartesian line, program → trajectory
│   ├── controller.py               500 Hz state machine: HOLD / FREE_DRIVE / PLAYBACK / DISABLED
│   ├── backend/                    ArmBackend protocol, SimBackend, RebotArmBackend
│   ├── server/                     FastAPI (REST + WebSocket) and the static teach-pendant UI
│   └── cli.py                      `robot-teaching serve` / `robot-teaching plan`
├── tests/                          pytest suite (runs against the simulator)
└── third_party/reBotArm_control_py git submodule (vendor library: actuator, kinematics, dynamics)
```

The decision to build on `reBotArm_control_py` and the control design are recorded in
`robot_control/docs/adr/0002-teaching-interface-backend.md`.

## Install

Python 3.10 or 3.11 (the vendor library pins `<3.12`) and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules <this repo>
cd robot_teaching
uv sync --extra dev            # simulator + tests
uv sync --extra dev --extra hardware   # adds motorbridge (Linux) to drive the real arm
```

`third_party/reBotArm_control_py` is imported from the submodule path (it has no installable
packaging); `REBOTARM_CONTROL_PY_DIR` can point at another checkout.

## Run

Simulator (no hardware):

```bash
uv run robot-teaching serve --backend sim
# UI  http://127.0.0.1:8000/     API docs  http://127.0.0.1:8000/docs
```

Real arm. Bring up the CAN interface the way Seeed's wiki describes for your adapter
(SocketCAN `can0` at 1 Mbit/s, e.g. `sudo ip link set can0 up type can bitrate 1000000`),
check that `third_party/reBotArm_control_py/config/rebotarm.yaml` selects `rebotarm_rs.yaml`,
and make sure the motor zero has been set once with the vendor's `example/2_zero_and_read.py`
(the URDF zero is the fully extended rest pose). Then:

```bash
uv run robot-teaching serve --backend rebotarm --host 0.0.0.0   # reachable from a tablet on the LAN
```

On start the controller connects, switches every motor to MIT mode, enables them and holds
the current pose with the stiff gains from the vendor's hardware YAML.

Offline check of a saved program:

```bash
uv run robot-teaching plan programs/example_pick_place.json --csv /tmp/traj.csv
```

## Teaching workflow

1. Move the arm to a **clearance pose** (elbow up, nothing touching the table) and press
   **Free drive**. Gains fade from stiff to compliant over 0.5 s; the badge shows
   `LOCKED`. Push the end-effector: the arm follows (`FOLLOW`) and re-locks when you stop.
2. Press **● Record point**. The current joint configuration, end-effector pose and gripper
   target are stored with the chosen motion type / speed / blend / dwell.
3. For small corrections press **Hold**, then use the **joint** or **Cartesian** jog buttons
   and the **gripper** controls, and record (or **Update** an existing point).
4. Edit the list inline: rename, change motion type or speed, toggle blend, set dwell or the
   gripper value, reorder with ▲▼, **Go** to check a point, ✕ to delete. **Save** writes
   `programs/<name>.json`.
5. **Plan (preview)** shows the planned duration and arrival times. **▶ Play** runs the
   program from the current pose (hold mode only). The speed slider works live; **■ Stop**
   ramps the speed to zero and holds. **Space** (outside text fields) or **Esc** (anywhere)
   triggers the same graceful stop while the page has keyboard focus.

Point semantics: a point's motion type describes how it is *reached* from the previous
point (the first point from wherever the arm is). A **joint** point can be a **blend**
(passed through without stopping); the arm stops at a point that has a dwell, changes the
gripper target, or precedes a linear move. The **gripper** value is applied on arrival,
followed by a settle time (`gripper.settle_time`) and the dwell. **Linear** points always
stop at both ends and must be on the same IK branch as the previous point (otherwise the
planner refuses and suggests a joint move).

## Configuration (`config/teaching.yaml`)

| section | what |
|---|---|
| `control` | setpoint rate (500 Hz, matching the vendor config) and UI stream rate |
| `limits` | per-joint velocity / acceleration at speed 1.0, Cartesian speed limits, joint-limit margin, planner sample spacing |
| `free_drive` | compliant gains, integral term, `tau_scale` per joint, release / re-lock thresholds, velocity filter |
| `hold` | stiff gains (null = the vendor's per-joint MIT gains) |
| `jog` | step sizes offered in the UI, jog speed, max joint jump accepted from a Cartesian IK step |
| `gripper` | closed / open motor positions (rad), settle time, gains |
| `playback` | default speed scale |

Motor ids, CAN channel, MIT gains and the gravity profiles stay in the vendor's
`config/rebotarm_rs.yaml`.

**Calibrate on your arm before trusting the defaults:** the gripper `open_position`
(the vendor examples use 5.0 rad), the free-drive gains (start with one joint enabled,
from the "L" pose joint2 ≈ 0.7 rad, joint3 ≈ 1.1 rad), and the playback limits (the
defaults are deliberately slow).

## Safety

* The software **Stop motion** is a stiff position hold, not a safety function. Keep a
  hardware e-stop that cuts motor power within reach.
* **E-STOP** in the UI disables the motors: a loaded arm will fall.
* Free-drive relies on the URDF gravity model (≈5–11 % error measured by the vendor) plus
  joint friction. Start it from a clearance pose; contact with the table or between links
  corrupts the compensation.
* Playback starts only from hold mode and only when the planned trajectory starts where
  the arm actually is. Every error in the control loop falls back to a stiff hold and is
  shown in the UI banner.
* The RS firmware's velocity readout is not in rad/s (vendor finding); velocity is
  estimated by finite differences of position.

## Tests

```bash
uv run pytest
```

The suite covers the program model, the planners (limits, continuity, straightness),
the controller state machine on the simulator (hold, jog, free-drive lock/follow,
playback tracking and gripper events, stop ramp, e-stop, error fallback) and the HTTP /
WebSocket API.

## Known limitations / next steps

* Blends are joint-space only; linear segments stop at both ends.
* No collision checking: taught points are validated against joint limits only.
* Gripper positions are motor angles; a calibration helper and a force-limited close are
  not implemented.
* The vendor's position request every cycle plus the command frame is near the CAN
  bandwidth at 500 Hz with 7 motors; lower `control.rate` if the bus saturates.
