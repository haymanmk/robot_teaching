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
  orientation under the same minimum-jerk time law (blended linear blocks run under a
  velocity-limit curve), tracked by closed-loop IK and re-timed to the limits. Setpoints stream at
  500 Hz in MIT mode with gravity, Coulomb-friction and inverse-dynamics feed-forward
  (the URDF's inertia and Coriolis terms plus a per-joint armature for the reflected rotor
  inertia, evaluated on the commanded motion). Speed override, loop and smooth stop.
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
│   ├── server/                     FastAPI (REST + WebSocket), the static teach-pendant UI and the offline plotter
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

Real arm. Check that `third_party/reBotArm_control_py/config/rebotarm.yaml` selects
`rebotarm_rs.yaml`, and make sure the motor zero has been set once with the vendor's
`example/2_zero_and_read.py` (the URDF zero is the fully extended rest pose). Then:

```bash
uv run robot-teaching serve --backend rebotarm --host 0.0.0.0   # reachable from a tablet on the LAN
```

With the real arm the server starts **disconnected**: open the UI and press **Connect**
when the arm is powered and clear. Connecting brings the CAN interface up if it is down,
opens the bus, switches every motor to MIT mode, enables them and holds the current pose
with the stiff gains from the vendor's hardware YAML.

**CAN interface.** The SocketCAN interface named by the vendor YAML (`can0`) is brought up
on Connect with `ip link set can0 up type can bitrate 1000000` (`can.bring_up` and
`can.bitrate` in `config/teaching.yaml`; `--no-can-up` disables it for one run). `ip link`
needs root rights, so the server runs it through `sudo -n`, which only works without a
password prompt. Allow exactly that command once:

```bash
echo "$USER ALL=(root) NOPASSWD: $(which ip) link set can0 *" | sudo tee /etc/sudoers.d/robot-teaching-can
```

Without the rule, Connect fails with the command and this hint, and you can bring the
interface up by hand the way Seeed's wiki describes for your adapter. An interface that is
already up (by hand, or by a systemd-networkd `.network` file with `[CAN] BitRate=1M`) is
left alone. Closing the bus switches the motors off (upstream behaviour), so
**Disconnect** and **Ctrl+C** first move the arm **home**: a planned joint move to the home
position in `config/teaching.yaml` (`home.q`, default the URDF zero, like the vendor's
`safe_home`), waiting for the move to finish before the motors are released. Playback is
stopped and free drive is left first; an e-stopped arm cannot move (its motors are already
off). If the move cannot be planned, Disconnect refuses and the arm keeps holding; Ctrl+C
prints the reason and disconnects anyway. `home.on_shutdown: false` turns the shutdown move
off, and the **Go home** button runs the same move on its own. `--connect auto` connects
at startup instead; the simulator defaults to `auto` and accepts `--connect manual`.

**Home position and the Cartesian readout.** The XYZ/RPY shown in the UI, and the pose
column of the program, are relative to the home position by default: XYZ is the offset of
the gripper frame from its home location along the base axes, and RPY is the rotation
about the base axes from the home orientation, so both read zero at home. The **Base**
toggle shows the raw pose in the robot base frame instead. Programs store joint angles,
so this choice never affects playback.

Offline check of a saved program:

```bash
uv run robot-teaching plan programs/example_pick_place.json --csv /tmp/traj.csv
```

A fuller offline validation of the planner (smoothness, limits, straightness, plots) is
`tools/validate_plan.py`; its checks and findings are in `docs/planner_validation.md`.

## Teaching workflow

1. Press **Connect** (real arm; the simulator connects by itself). Move the arm to a
   **clearance pose** (elbow up, nothing touching the table) and press **Free drive**. Gains fade from stiff to compliant over 0.5 s; the badge shows
   `LOCKED`. Push the end-effector: the arm follows (`FOLLOW`) and re-locks when you stop.
2. Press **● Record point**. The current joint configuration, end-effector pose and gripper
   target are stored with the chosen motion type / speed / blend / dwell.
3. For small corrections press **Hold**, then use the **joint** or **Cartesian** jog buttons
   and record (or **Update** an existing point). The **gripper** controls work in hold and
   free drive: **Open**, **Close** and the slider command a stiff position; **Hand** lets you
   open or close the gripper by hand (light damping) and the recorded opening follows your
   hand. Hand mode stays on until the next Open/Close/slider command or playback, so a
   grasped object is not released by switching modes.
4. Edit the list inline: rename, change motion type or speed, toggle blend, set dwell or the
   gripper value, reorder with ▲▼, **Go** to check a point, ✕ to delete. **Save** writes
   `programs/<name>.json`.
5. **Plan (preview)** shows the planned duration and arrival times. **▶ Play** runs the
   program from the current pose (hold mode only). The speed slider works live; **■ Stop**
   ramps the speed to zero and holds. **Space** (outside text fields) or **Esc** (anywhere)
   triggers the same graceful stop while the page has keyboard focus.

**Scope.** The card at the bottom of the pendant shows one joint at the full control rate:
commanded and measured position, their difference, and the torque the MIT law asks for next
to the torque the motor reports. Pick the joint in its drop-down. It is the tool for telling
stick-slip (a sawtooth in the error at low speed) from bus trouble (gaps on every joint) from
low damping (a 3–4 Hz ring on one joint). **−** / **+** (or Ctrl+wheel over a trace) zoom the
window between 0.5 s and 60 s, the length of the history the server keeps. **Pause** freezes
the traces: the time axis then shows the controller clock and dragging a trace looks back in
time; **Resume** returns to live. The readout stays live. **Record** writes every sample of
the selected joint to a CSV file that downloads on **Stop**, with the columns `t_s`, `joint`,
`q_cmd_rad`, `q_rad`, `err_rad`, `tau_cmd_Nm`, `tau_meas_Nm` and `run` (`run` increments when
the server restarts, where `t_s` starts again from zero); **Plot…** opens the offline plotter for
such a file (below). **Spectrum** adds a fourth strip
with the amplitude spectrum (dB) of the tracking error, a position or a torque over the samples
on screen: Welch's estimate from Hann-windowed, detrended segments of about 2 s (Δf ≈ 0.5 Hz at
500 Hz; a segment with a hole is left out), the highest peak above 1 Hz marked and read out,
and − / + or Ctrl+wheel over it setting the frequency range from 5 Hz up to half the sample
rate. A wobble is a peak in the error; the same peak in the commanded torque means the loop
drives it. The estimate lives in `static/dsp.js`; `tests/test_dsp.py` checks it under Node.
The readout keeps the largest
tracking-error magnitude seen since the joint was picked, over every sample received rather
than only the ones on screen; **Reset max** clears it. A hole in a trace means samples were
missing. The scope survives a reconnect and a server restart without a page reload.
Programmatically, send `{"trace_joint": i}` on the `/ws/state` websocket and read the `trace`
chunk of each message; its `epoch` identifies the server run, so a new value means the clock
`t` started again from zero.

**Offline plotter.** http://127.0.0.1:8000/plot (the **Plot…** button of the Scope card) shows a
recorded CSV file with the same three strips and spectrum as the live scope, drawn by the same
code. The page needs no server: `src/robot_teaching/server/static/plot.html` can be opened
straight from disk, and the file is read by the browser, never uploaded. **Open CSV…** (or a
drop anywhere on the page) loads one or more recordings; a recording that spans a server restart
holds more than one *run* (the `run` column changes and `t_s` starts again from zero) and each
run is listed on its own. The strip at the top shows the whole run's tracking error with the
window outlined: click or drag it to jump, drag a trace to move through time, **−** / **+**,
Ctrl+wheel or the − and + keys zoom between 100 ms and the whole run (**Fit**), ← → move by half
a window, and hovering a trace reads the sample under the pointer. **Compare** overlays a second
recording, B, on the same axes, dashed and a little lighter, with its start on the start of the
first run; **offset** (or Shift+drag on a trace) slides B in time and **Align starts** puts it
back, so a move recorded before and after a gain change can be laid over itself. The summary
line gives each recording's length, sample rate, holes (steps of more than 0.25 s), rows without
a measurement and largest tracking error; the readout under the strips gives the same for the
samples on screen, the hover values and the spectrum show A and B together.
Positions are shown in degrees, as in the live scope. The CSV reader is `static/record.js`,
checked by `tests/test_record.py` under Node.

Point semantics: a point's motion type describes how it is *reached* from the previous
point (the first point from wherever the arm is). A **blend** point is passed through
without stopping: consecutive joint points blend in joint space (a C² spline), consecutive
**linear** points form one Cartesian path whose corners are cut within
`limits.linear_blend_radius` (an industrial blend zone), with continuous velocity and a
speed capped by the centripetal acceleration the blend needs. The arm stops at a point
that has a dwell, changes the gripper target, or precedes a point of the other motion
type. The **gripper** value is applied on arrival, followed by a settle time
(`gripper.settle_time`) and the dwell. A linear block must end on the same IK branch as
the taught configuration (otherwise the planner refuses and suggests a joint move).

## Configuration (`config/teaching.yaml`)

| section | what |
|---|---|
| `control` | setpoint rate (500 Hz, matching the vendor config) and UI stream rate |
| `limits` | per-joint velocity / acceleration at speed 1.0, Cartesian speed limits, joint-limit margin, planner sample spacing |
| `free_drive` | compliant gains, integral term, `tau_scale` per joint, release / re-lock thresholds, velocity filter |
| `hold` | stiff gains (null = the vendor's per-joint MIT gains) |
| `friction` | Coulomb friction per joint (N·m) fed forward during playback with the sign of the commanded velocity, so slow joints do not stick and slip; 0 disables |
| `feedforward` | inverse-dynamics feed-forward during playback: `inertia_scale` (0 disables), per-joint `armature` (rotor inertia × gear ratio², kg·m², which the URDF lacks), `max_torque` clip |
| `jog` | step sizes offered in the UI, jog speed, max joint jump accepted from a Cartesian IK step |
| `gripper` | closed / open motor positions (rad), settle time, gains |
| `can` | bring the SocketCAN interface up on Connect (`bring_up`) and at which `bitrate` |
| `playback` | default speed scale |
| `home` | home position (motor rad): Cartesian reference, target of **Go home**, and the pre-disconnect move; its speed and whether to move home on shutdown |

Motor ids, CAN channel, MIT gains and the gravity profiles stay in the vendor's
`config/rebotarm_rs.yaml`.

**Local overrides.** Values that belong to one machine, such as tuned gains, friction, the
armature or the hardware YAML, go in `config/teaching.local.yaml`, which git ignores. It is
merged on top of `config/teaching.yaml` section by section: a key in the local file replaces
that key, everything else keeps the shipped value. The server prints which files it loaded,
and the gains reload reads both again. For the motor gains of your arm, copy the vendor's
`config/rebotarm_rs.yaml` to `config/rebotarm_rs.local.yaml` and point at it:

```yaml
# config/teaching.local.yaml
hardware_yaml: rebotarm_rs.local.yaml     # relative to this file; --hw-yaml overrides
hold:
  kp: [60, 160, 160, 60, 60, 60]
feedforward:
  armature: [0.02, 0.05, 0.05, 0.01, 0.01, 0.01]
```

Both files are read when the server starts. The gains can be re-read without a restart:
edit the YAML and press **Reload gains** (Scope card) or `POST /api/gains/reload`. That
re-applies the hold, free-drive and gripper kp/kd, `free_drive.tau_scale`, the free-drive
thresholds, the friction feed-forward and the inverse-dynamics feed-forward, fading the gains in over
`free_drive.transition_duration`; it works in hold, free drive and while disconnected, not
during playback. Limits, jog steps and the home position still need a restart.
`GET /api/gains` shows the values in use.

**Calibrate on your arm before trusting the defaults:** the gripper `open_position`
(the vendor examples use 5.0 rad), the free-drive gains (start with one joint enabled,
from the "L" pose joint2 ≈ 0.7 rad, joint3 ≈ 1.1 rad), the playback limits (the
defaults are deliberately slow), and the friction feed-forward `friction.coulomb` (the
defaults are 75 % of the values the vendor measured on one arm; the vendor's
`tools/gravity_calibration` PD sweeps measure them on yours).

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
WebSocket API. `tests/test_dsp.py` runs the scope's spectrum maths under Node
(`node tests/dsp_check.js`) and `tests/test_record.py` the scope recording reader and the
drawing module's axis helpers (`node tests/record_check.js`); both are skipped when `node` is
not installed.

## Known limitations / next steps

* A blended linear corner is a quadratic Bézier cut; the blend radius is global, not per point.
* No collision checking: taught points are validated against joint limits only.
* Gripper positions are motor angles; a calibration helper and a force-limited close are
  not implemented.
* The vendor's position request every cycle plus the command frame is near the CAN
  bandwidth at 500 Hz with 7 motors; lower `control.rate` if the bus saturates.
