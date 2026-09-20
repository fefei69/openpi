# Async Hanoi deployment

[Hanoi overview](../README.md) · [Local checkpoint and episode validation](validation/README.md)

The client runs observation/control at 30 Hz while a background worker calls the OpenPI WebSocket server.
It implements timestamped future actions using the pattern from
[LeRobot async inference](https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference).
The local `tower_hanoi` code is API reference only; it is neither imported nor copied.

## Run

Use these launchers from the repository root. They also work by absolute path from another directory.
All accept `--help` and forward arguments to their Python entrypoints.

**Current experiment:** `./run_policy_client.sh` defaults to `--velocity-source recorded` for live runs.
It uses recorded velocity only in the policy input; image, XYZ, jaw stroke, and hardware feedback stay live.
Use `./run_policy_client.sh --velocity-source measured` to restore measured policy velocities.
Both retain the 30-second default. The server requires no change.

### 1. Align above rod A and test readback

```bash
./test_robot_initial_pose.sh
```

This **moves the robot**. It needs the arm connection, but no camera, ROS, or policy server.
Use the same clear startup path as dataset collection, with the board set to AAAA (all four rings on peg A).
If the gripper is already closed, startup first opens it before homing; this also handles retrying after a failed
grasp. It then uses the selected rod-A setup pose:

1. Move to six zero joint angles over 6.4 seconds and check joint readback.
2. Move in Cartesian space to XYZ **[0.492297590, -0.056030598, 0.191169396] m**,
   rotation vector **[0.0, 0.785398163, 0.0] rad**,
   over 4.208 seconds.
3. Open the jaw to **0.034 m** over 1 second, then pause for 0.5 seconds.
4. Read XYZ and compare with the **original desired pose**. If the Euclidean error exceeds **0.5 mm**, add
   `desired_xyz - measured_xyz` to the previous command and move again. Allow at most **three corrections**;
   stop before any command whose cumulative compensation from the desired pose would exceed **5 mm**.
   Each correction is blocking, takes 4.208 seconds, and keeps the fixed rotation and driver trajectory checks.
   Reject setup if final XYZ error exceeds 0.5 mm. Rotation must remain within **2 degrees**, and jaw within **3 mm**.
   Measured velocities are reported, but are not a pass/fail criterion for this pose test.
5. Return to the six-zero-joint home pose, check home readback, and disconnect.

The script prints `PASSED` or fails with a nonzero exit code and readback details. Every attempt writes a fresh
`data/hanoi/deployment/initial_pose_<timestamp>.json`. The user-selected target is the measured XYZ of
**K4326 medoid 1296, observation 87660**, above rod A. It replaces the earlier start behind peg B.
The same setup is used automatically by the pi0.5 policy client (the Cosmos client starts elsewhere, see
"Start pose" below); the desired pose is never replaced by a compensated
command. Reports contain `initial_proprio_alignment.final_error_mm`, `passed`, and individual `corrections`.
For a policy run, these fields are in `summary.json` and the successful `robot_initialized` event in `events.jsonl`.
Accuracy is based on controller readback; it does not establish independent physical accuracy or matching joints.
A pose mismatch also returns the arm home before reporting failure. Ctrl-C holds the arm and opens the gripper
in place. A controller/connection fault attempts a hold and disconnects. Reports include measured
pose and return-home status.

### 2. Start the selected policy server

In a separate terminal:

```bash
./run_policy_server.sh
```

The default checkpoint is `checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999`, with ten sampling steps.
The launcher uses the existing Python 3.11 `.venv` and validated CUDA/memory defaults, preserving environment overrides.
The server binds to localhost by default. Use `--host` for a separate client machine.

### 3. Run the live client for 30 seconds

Once the pose test passes, start the existing camera publisher with the collection view/settings and make its
image stream reachable from the client. Then:

```bash
./run_policy_client.sh
```

**Live robot control is now the default. No shadow step or extra arguments are required.**
The client first moves the arm through the same initialization sequence and checks the achieved pose.
It stays at that pose for inference warmup, then runs policy control. The standalone test returns home;
the client initializes again automatically, so no extra preparation arguments are needed.
`--duration-s` overrides the 30-second control duration; camera startup and model warmup are additional time.
Arm travel runs at **half the previous speed** by default: policy moves, initialization, and the pose test's
return-home moves take twice as long. Gripper closing ramps the force over **2.4 seconds**, followed by
**0.2 seconds** of settling before the arm can move again. No extra arguments are needed. Stop and opening
timings are preserved.

The default [workspace.json](workspace.json) bounds the recorded task region: the transferred episode's action XYZ
extrema expanded by 3 mm, with minimum Z clipped to the recorded table height. These are **record-derived bounds**,
not a separately commissioned obstacle model. `--workspace /path/to/bounds.json` overrides them with a JSON file
containing `xyz_min_m` and `xyz_max_m`. The driver also checks sampled trajectory feasibility.

The client requires the selected forward server identity, fresh images, valid measured feedback, and a verified
starting pose/jaw/orientation. Checkpoint weight hashes were verified separately; server metadata identifies the
configuration, export manifest, normalization, and sampling settings. The task prompt is fixed to AAAA-to-CCCC.

### How learned actions are executed

The live client uses `PolicyExecutor`. It skips prediction rows that elapsed during inference, takes up to nine
remaining rows with the same jaw intent, and selects the last XYZ as the next target. It sends one smooth,
rest-to-rest Cartesian move. It first computes a duration using speed, acceleration, and jerk limits of
**0.145 m/s, 0.23 m/s², and 1.3 m/s³**, then doubles that duration. Policy moves therefore last at least
**0.6 seconds**, with peak speed at most **0.0725 m/s**; acceleration and jerk also decrease. The target is not
clipped; workspace violations still stop execution with the offending range in the error message. Every move
finishes before the next one starts, while feedback and missed-grasp checks continue at 30 Hz.

There is no 0.1 mm polynomial reconstruction requirement for learned actions. At a jaw transition, the client
first moves to the XYZ paired with that open/close action, then operates the gripper at that position. This
move-and-grip pair finishes before a newer prediction can replace it. Later XYZ drift in the chunk is not
executed during the actual open/close dwell; fresh inference is required afterward. This is recorded as
`endpoint_rest_to_rest_v2` in the summary, with `gripper_alignment: true` on the approach command. Version 1
discarded the paired XYZ and operated the jaw at the previous endpoint, losing small final approach corrections.
The strict `ReferenceExecutor` remains in use only for offline teacher-trajectory reconstruction.

This adapter changes intermediate positions and timing relative to the training trajectory. It may move more
slowly and pause between targets; physical tracking and task success still need hardware verification.
The model observation/action format is unchanged; the nine-row prefix is now target lookahead, not a promise to
execute nine exact references in 0.3 seconds.

Minor scheduling lateness is logged as `control_late` / `dispatch_late`. Since each move ends at rest, the arm can
wait there for the next result. Controller faults, stale feedback/images, inference failures, invalid commands,
workspace violations, or excessive tracking error still stop the run and attempt a position hold.

### Recorded-velocity diagnostic

`--velocity-source recorded` substitutes only `observation/state[3:6]` (vx, vy, vz) with measured velocities
from the transferred training episode specified by `--episode`. It does not replay actions. The direct Python
entrypoint defaults to measured velocities; the shell launcher currently selects recorded velocities for this experiment.

The selector matches live XYZ to recorded measured XYZ, moving forward within the current recorded open/closed-jaw
interval. Actual dispatched gripper transitions advance that interval. After the live gripper dwell, matching starts
past the recording's corresponding dwell, so a lift is not confused with the preceding descent at the same XYZ.
It uses the exact recorded velocity values without scaling. The 30 Hz wall clock does not select the recorded row.
This is a position/phase-aligned diagnostic; the substituted values are **not measurements of the live arm's velocity**.
The assumed route is the transferred AAAA-to-CCCC episode. Matching distance and source row are logged to assess alignment.

All arm checks and recovery continue using real feedback. `events.jsonl` tick states remain measured. Each actual
request in `inferences.jsonl` includes `velocity_override`: original `measured_state`, `policy_velocity_m_s`,
`reference_row`, `gripper_phase`, and `match_distance_mm`. Its NPZ contains the exact modified policy input and live
image. The summary identifies `velocity_source` and the reference state/action checksum. Startup also prints the
active override. This test changes the policy input; the existing slow motion and gripper execution stay configured.

### Missed grasp and Ctrl-C

While a closed-gripper command is active, the client checks measured jaw stroke on every feedback read, including
during closing and later arm motion. **Stroke at or below 8 mm stops policy execution immediately.** This matches
the collector's threshold (20% of the calibrated 40 mm maximum stroke); `--min-grasp-stroke-m` overrides it.
This detects an empty/slipped grasp or a grip on something narrower than a ring; stroke alone cannot identify the
object or prove a successful grasp above the threshold. The supplied episode's successful closed-grip readbacks
were about 14–31 mm.

A missed grasp or **Ctrl-C** triggers the following recovery, without further policy commands:

1. Request an arm position hold and immediately switch the gripper to opening position control.
2. Open to 34 mm and verify the opening.
3. After a **missed/slipped grasp**, return to joint zero (all six arm joints = 0 radians), verify the return,
   then disconnect. The handled stop records `status: missed_grasp` without a traceback.
   **Ctrl-C** opens the gripper and leaves the arm at the stopped position.

Recovery uses arm feedback, not the camera or policy server. Ctrl-C also releases the gripper if it is holding a
ring. A second Ctrl-C can interrupt recovery; the client then attempts a hold and still closes its resources.
If opening or feedback verification fails, cleanup attempts a position hold. No controller errors are
automatically cleared and no torque-release command is sent.

Events include `missed_grasp`, `operator_stop`, gripper-release progress, and `return_home_started/finished/failed`.
The summary records `gripper_released`, `return_home_requested`, `returned_home`, and the offending stroke;
`task_success` is false for a detected missed grasp. A failed release prevents the return-home move.
Duration completion alone still
attempts a hold. Startup still homes before rod-A alignment, and a completed standalone pose test still returns home.

## Environments and offline replay

The client uses `.cache/hanoi-robot-venv` (Python 3.12), with ROS Jazzy from `/opt/ros/jazzy`.
The client launcher sources ROS automatically; `HANOI_ROS_SETUP` overrides the setup file. The initialization
launcher and CLI help do not source ROS. To recreate the client environment:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv --python /usr/bin/python3.12 .cache/hanoi-robot-venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install \
  --python .cache/hanoi-robot-venv/bin/python \
  -r examples/hanoi/deployment/requirements-robot.txt -e packages/openpi-client
```

The robot environment imports no JAX/Torch and leaves the existing `tower_hanoi` environment unchanged.
Recorded replay remains available explicitly, without ROS or hardware:

```bash
./run_policy_client.sh --mode replay --duration-s 10
```

Replay sends the transferred episode's RGB and measured state at 30 Hz, preserving image age and excluding stale
observations. Frames are already cropped. Predicted actions do not alter subsequent recorded observations;
this measures transport and command validation, not closed-loop behavior. `--start-row` selects a different segment.

## Timing and action behavior

- One request can be in flight. A single pending observation is replaced by newer observations; results are bounded
  to a single latest chunk. The control thread never calls inference or waits for a prediction.
- Each observation carries a local monotonic timestamp and 30 Hz tick. Row `k` is due at observation tick `+ k + 1`.
  Planning skips elapsed references. A new result replaces only uncommitted actions; a dispatched target move
  must finish. Its duration can exceed nine ticks to respect motion limits. Host tick jitter is logged.
- Planning is transactional: reference state advances only after successful dispatch. A dispatch failure exits;
  a new run must reconcile with measured state.
- Warmup runs after initialization and before policy actions. Its result and results from earlier generations are discarded.
- The executor starts from measured XYZ with zero reference velocity/acceleration after the completed initialization
  command. Instantaneous measured velocity does not gate startup and remains unchanged in policy observations.
  Gripper/hold transitions check the preceding reference velocity; measured tracking checks remain active.
- Gripper intent is thresholded at 0.5. Open: 0.034 m over 1 s. Close: ramp to -20 N over 2.4 s plus 0.2 s settling
  (78 ticks total). This is a slower force ramp; actual jaw speed depends on contact and load. The dataset contract
  and offline teacher executor retain the recorded 1.2 s close; the live driver and scheduler use 2.4 s.
  There is no Cartesian dispatch during the dwell. Repeated intent does not restart it. The old queue/in-flight
  results are invalidated, and the next observation must use an image received after dwell completion.
- State is measured tool XYZ, measured linear velocity XYZ, and measured gripper stroke. RobotOutput provides
  one feedback snapshot. No commanded jaw value or board/route label enters the model.
- Camera: depth-one best-effort `rgb8` subscription on `/camera/camera/color/image_raw`, 640×480 with stride honored,
  crop `[90:450, 151:511]`, `INTER_AREA` resize to 224×224. Callback receipt time is captured before decoding.
  Maximum image age at the observation anchor is 50 ms. ROS capture/header time is never mixed with monotonic time.
- Orientation is the collection's fixed RPY `(0, pi/4, 0)`, which for this pure-pitch rotation also equals the
  driver's angle-axis vector. WXAI standard follower tool configuration is used; verify its physical calibration.
- Runtime is bounded by `--duration-s` or operator stop. There is no automatic goal-board detector yet;
  `task_success` is `false` for a detected missed grasp and otherwise `null`. A model hold is not task completion.

## Local evidence (2026-09-15)

Before the half-speed setting, the target adapter was checked offline against the first failed live prediction in
`live_1789506803104332052`: it generated a 0.367-second move with peak speed 0.00502 m/s, acceleration 0.0421 m/s²,
and jerk 1.194 m/s³. Re-evaluating 90 saved replay predictions independently from their measured states accepted
82 Cartesian moves and 8 gripper commands, with no workspace or motion-limit violations. This is command-builder
validation using saved data, not a closed-loop rollout. Evidence is in
`data/hanoi/async_validation/target_adapter_validation.json`. The failed live prediction is also a regression fixture
under `tests/fixtures/first_live_prediction.json`.
The same regression fixture now takes 0.733 seconds, with half the speed, one-quarter the acceleration, and
one-eighth the jerk. The saved historical report retains the original timings.

The earlier ten-second real-checkpoint replay (using the original strict reconstruction adapter) in
`data/hanoi/async_validation/replay_1789502288692489408/` produced:

| Measurement | Result |
| --- | --- |
| Control ticks | 300 at 30 Hz |
| Async predictions | 90 |
| Observation-to-result latency, median / P95 | 125.2 / 140.1 ms |
| Maximum tick scheduling delay | 0.997 ms |
| Stale recorded observations excluded | 14 |
| Commands accepted / rejected | 0 / 90 |
| Rejection reasons | 79 reconstruction failures; 9 motion-during-jaw; 2 prefixes crossing jaw events |

This replay initializes each planning boundary from the current measured position/velocity and zero acceleration;
recorded observations do not follow hypothetical commands. It is not a continuous executed rollout; the rejection counts should be interpreted
accordingly. A separate 100-anchor check also failed the old reconstruction checks. Those failures motivated
the target adapter described above; they are retained here as historical evidence.

The ROS synthetic publish/subscribe result is in `data/hanoi/async_validation/ros_camera_test.json`.
The offline tests exercise latency skipping, immutable committed intervals, cancellation, resets, gripper barriers,
real WebSocket serialization/close, a complete slow-policy control loop, measured feedback/watchdogs, and driver
argument mapping. Run them with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/hanoi/tests/initialization_test.py \
  examples/hanoi/tests/async_inference_test.py \
  examples/hanoi/tests/execution_test.py src/openpi/policies/hanoi_policy_test.py
```

Each run creates a new directory under `data/hanoi/deployment/` (override with `--output`), containing configuration,
server metadata, first observation/crop, tick/state/prediction/command/error events, and a summary. The JSONL records
include predicted arrays and timings; `summary.json` distinguishes duration completion, operator stop, and failures.

### Exact inputs for every inference

Recording is automatic, including warmup. The inference worker writes the **actual request**, after selecting
the latest pending observation, before calling the server. Superseded observations that are never sent are not
recorded as requests. Disk writes run in the inference thread, separately from the arm control loop.

- `inference_inputs/000000.npz`, `000001.npz`, ...: lossless snapshots of the exact request keys:
  `observation/image` (224×224 RGB), `observation/state` (measured `[x, y, z, vx, vy, vz, jaw_stroke]` in metres and
  metres/second), and `prompt`. These are client inputs before the server's model transforms.
- `inferences.jsonl`: `inference_request` records contain the state, prompt, archive path, observation tick,
  generation, capture/image timestamps, and image age. Matching `inference_response` records contain all 63
  returned actions and request/response timings. `inference_error` preserves failures; `cancelled: true` marks a
  request interrupted during worker shutdown. A request without a terminal record indicates an interrupted run.
- `request_id` links these records to `prediction` and `command` events in `events.jsonl`. Command events also
  identify the source observation tick and generation. A saved response is not necessarily executed: warmup,
  invalidated results, and replaced chunks are retained for analysis. Request 0 is warmup in the default client.

Input archives are saved independently before inference, so previous inputs remain readable if a later request
fails. Nothing waits until run completion to save the history. For example:

```python
import numpy as np

with np.load("data/hanoi/deployment/live_<run>/inference_inputs/000001.npz", allow_pickle=False) as request:
    image = request["observation/image"]
    state = request["observation/state"]
    prompt = request["prompt"].item()
```

Use this history with the measured tick states and executed targets to compare live observations with the
recording and investigate accumulating errors. Existing runs only saved the first image; their missing camera
history cannot be reconstructed from the state logs.

## Cosmos waypoint client

`run_cosmos_client.sh` drives the arm with the Cosmos Policy waypoint_v4 checkpoint instead of pi0.5. It shares
this folder's camera, arm, workspace, watchdog and recovery code; only the policy contract differs. The server
lives in the Cosmos checkout (`../cosmos-policy`, branch `hardware`) and runs in its own environment:

```bash
# Terminal 1, in ../cosmos-policy
source examples/hanoi/env.sh && export COSMOS_POLICY_PLATFORM=hanoi_joint
.venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_waypoint --port 8001
# Terminal 2, here
./test_robot_initial_pose.sh   # unchanged
./run_cosmos_client.sh         # live, 60 seconds by default
```

Differences from the pi0.5 client:

- Observation: six measured joint angles plus jaw stroke from one driver snapshot (`TrossenArm.read_joints`),
  and the measured XYZ as separate decoding context. No velocity.
- Output: eight absolute destinations. The client executes the first `--commit-count` of them (default 1, the
  contract's execution prefix) before observing again. The summary's `execution_adapter` records the count.
- Loop: observe, infer (about 0.5 s on the RTX 5080), a rest-to-rest move through `move_to_target` (same limits
  and half-speed scaling), then a jaw change if the intent differs (same open/close timings), wait for
  completion, require a camera frame received after completion, repeat. There is no 30 Hz action timeline and no
  elapsed-row skipping. Feedback, tracking and missed-grasp checks still run at 30 Hz during every command.
- Identity: the client refuses a server whose `cosmos_hanoi.contract` differs from the waypoint_v4 deployment
  contract or whose export SHA-256 is not the selected `iter_000008000.pt`. `--expected-export-sha256 ""`
  disables the hash check for experiments.
- Modes: `--mode replay` streams rows of a Cosmos single-episode extract (`--episode`, `--replay-rows` or
  `--replay-stride`) with no ROS or arm; `--mode shadow` reads the live camera and arm and logs predictions
  without motion; `--mode live` moves.

Tolerances: in-flight tracking 8 mm (`--max-tracking-error-m`), arrival 3 mm after a 0.5 s settle window
(`--max-arrival-error-m`, `--arrival-settle-s`), missed grasp at or below 8 mm stroke. The in-flight limit is looser
than the pi0.5 client's 3 mm on purpose: the first live run (2026-09-18, `cosmos_live_1789762833813111489`) descended,
closed to a 14 mm grip on the ring, and was stopped 0.6 s into the 105 mm lift at 3.014 mm lag. An unloaded descent
of the same length tracked within 1.3 mm; a loaded lift lags the reference by several millimetres from the start.

Workspace: `cosmos_workspace.json`, derived from the waypoint_v4 training labels (10 mm lateral margin, 15 mm above
the hover height, floor at the table). The pi0.5 `workspace.json` ceiling rejected a 194.20 mm lift-away by 0.07 mm on
the second live run; live predictions sit 2 to 4 mm outside the recorded label range. A rejected command now opens the
gripper and returns home rather than holding, since nothing was dispatched.

First live result (2026-09-18, `cosmos_live_1789764033395957798`): the selected export completed the first ring
transfer, peg A to peg B, in one attempt: 14 mm grip, loaded-lift lag at most 2.6 mm, every arrival within 0.5 mm.

Raw execution is the default: the arm goes exactly where the model says. Runs 3 and 4 (2026-09-18) showed the
policy choosing the correct recorded destination every time but with a few millimetres of bias that its relative XYZ
encoding carries from one waypoint to the next: hovers 1.5 to 3.6 mm high, then a second grasp 3.5 mm above the ring
level (81.0 versus 77.5 mm) whose 16.9 mm edge grip slipped. Runs 5 and 6 (raw client, 2026-09-18) reproduced runs 2
and 3 step for step, so the errors are systematic. Regressing each step's prediction error on the offset the arm
started that step from gives a slope of about 0.6 in y and z: roughly 60% of an off-grid offset is carried into the
next destination, plus under 1 mm of fresh upward bias per step (z error along run 6: 0, +2.1, +1.8, +3.1, +2.9,
+3.6, +3.8 mm). That partial compounding is the model's real-world precision as it stands, against a grasp tolerance
of about 3 mm. The constant 3.5 mm x error was a start-pose mismatch, fixed below.

Start pose: the client starts at the recorded waypoint_v4 episode start, (414.0, 15.8, 191.2) mm behind peg B
(`--start-xyz-m`), not at the pi0.5 hover above rod A. In this recording the rod-A hover (492.6, -56.2, 191.1) is a
carry pose: 238 training windows begin there and none is labelled with a grasp descent, while the grasp hover over A
is at x = 496.1 mm. Started there, the model descended straight down from the wrong column, so every grasp and
release in runs 1 to 6 was 3.5 mm short in x (spread 0.06 mm) and the relative encoding kept that offset for the
whole chain. From the recorded start, replay predicts the hover over A within 0.5 mm of the 496.1 column. After
alignment the client also checks the six joint angles against the recorded start joints (`start_joints_verified`,
limit `--max-start-joint-error-rad`, 0.05 rad), since joints are the model's state input. Dry-run the pose on the arm
without the policy: `./test_robot_initial_pose.sh --start cosmos_v4`.

Dreams: start the server with `--dream` and every reply also carries the model's predicted future frame (what it
expects the camera to see after the whole eight-waypoint chunk) and its value estimate; the client saves the frame
as `inference_dreams/NNNNNN.png` and logs the value in `inferences.jsonl`. The destinations are unchanged (same
generated latent, one extra VAE decode). For a run recorded without it, `examples/hanoi/dream_run.py --run-dir <run>`
in the Cosmos checkout regenerates them from `inference_inputs/`; offline decisions match the live ones within 0.25 mm.

pi0.5 on the same dataset: `run_waypoint_server.sh` serves
`checkpoints/pi05_hanoi_waypoint_aaaa_to_cccc/hanoi_waypoint_full_20260917/exports/29999` (config
`pi05_hanoi_waypoint_aaaa_to_cccc`, trained on waypoint_v4, validation first-waypoint error 0.47 mm mean and
1.3 mm at the 99th percentile, jaw intent 100%) on port 8000 through `examples/hanoi/deployment/serve_waypoint.py`.
It publishes its identity as `hanoi_waypoint` with the same contract, so the same client, start pose, joint
check and execution apply: `./run_cosmos_client.sh --server ws://127.0.0.1:8000 --mode live --duration-s 150`.
The summary records `policy_family` and the adapter name starts with `pi05_`. The identity check selects the
export by family (`--expected-export-sha256 selected`, the default). The server warms up once before listening,
so the first request is fast; jaw intent is thresholded at 0.5 on the server as the contract states.

pi0.5 run 1 (2026-09-18, `cosmos_live_1789772013551974237`, raw execution, live camera and joints only): 11 of the
15 moves in 322 s, then the tracking watchdog stopped a descent between pegs A and B (8.3 mm error at 29% of the
move, contact with the ring on B). Along the way three null moves (a ring grasped and released back on the same
peg, or carried to its destination and brought back) and a few 1 to 3 mm hover dithers before a grasp. The null
moves are not sampling noise: eight fresh samples reproduce every decision. They are a state shortcut. In the
recording each peg has a grasp column and a release column a few millimetres apart (C: y 89.0 against 86.5 mm),
so after a lift the arm's position alone says whether it just grasped here (carry away) or just arrived carrying
(release here), and the policy learned that cue from its discretised joint state. Swapping the joint state
between request 28 (hover at y 86.7, released ring 1 back on C) and request 34 (y 88.8, carried it to A) swaps
the decisions with the images unchanged; a sweep of the state between them flips in one step at 88.6 to 88.8 mm;
the same swap turns the ring 3 return to B (request 79, hover at the grasp column) into the correct release, and
the wrong grasp at A (request 36) into a hover. Live lift hovers land within about 2 mm of the column, so the cue
sometimes reads the wrong side. Model-side fixes: train with state noise or without the discretised state so the
phase must come from the image and jaw stroke, or record grasps and releases from one column per peg.

Commit count: runs 7 to 9 (v4 start pose, re-plan after every waypoint) completed the same four moves and missed
the same fifth grasp on peg C, within 2 mm of each other at every step. Offline the export is precise on its own
data (under 1 mm at most steps, 2.5 mm at worst, on the validation episode), but at that one transition it moves
further off a column rather than back: from 4.7, 5.5 and 6.6 mm off the peg C column it descended 3.0, 3.1 and
3.7 mm further, and 3 to 6 mm too high. Since chunk positions 1 to 7 are as accurate as position 0 (median 1.1 to
1.6 mm), runs 10 to 12 tried `--commit-count 3` (hover, gripper action, lift per observation). That was worse: the
open-loop waypoints carry whatever offset the arm had at prediction time, and the next re-plan adds to it, so y
drifted at 0.15 mm per waypoint and z at 0.1 mm per waypoint (against no trend with commit 1, whose error
returned to zero by waypoint 20), and the grasp of the third ring at move 4 failed 2.4 to 3.0 mm high. Re-observing
after every waypoint does correct part of the offset at most steps; the anti-correction is specific to the peg C
grasp approach. The default stays 1; the option remains for ablations, `chunk_index` on each `command` event says
which waypoint of the chunk it was, and a chunk is cut short only when the duration ends with nothing held.

`--snap-to-recorded-destinations` is an ablation, off by default: it snaps each committed destination onto the 18
recorded points the server publishes (hovers by height only, releases and grasps to the nearest point, refusing a grasp
between two ring levels). It answers "does execution succeed when only the choice of point is the model's?" and must
not be on when reporting the policy's performance; the summary then records `execution_adapter` with a
`_snapped_ablation` suffix. The raw default never reads the published grid, so it runs against any server
build; only the ablation refuses to start when the server predates the grid (Cosmos commit 936ef05).

End of run: when `--duration-s` elapses the client finishes a placement in progress (a held ring is never
released mid-carry) for at most `--finish-grace-s` (30 s), then opens the gripper and returns to joint home, so the
arm ends parked like after `test_robot_initial_pose.sh`. `--no-return-home-after-duration` keeps the old hold. A
missed grasp also releases and homes; Ctrl-C releases where it stopped; any other failure holds in place for inspection.

Runs write `data/hanoi/deployment/cosmos_<mode>_<ns>/` with the same files as the pi0.5 client (`events.jsonl`,
`inference_inputs/`, `inferences.jsonl`, `summary.json`, `camera_crop.png`).

Verified 2026-09-18 on this machine: six replayed validation observations produced identical actions across two
client runs against one server process (0.000 mm), and within 0.2 mm of the offline evaluation run in a separate
process. That 0.2 mm is the model's own bf16 process-to-process variation, also seen between two offline runs.
Tests: `examples/hanoi/tests/cosmos_client_test.py`.

## Dense contract-five client (pi0.5 trained on the raw 30 Hz recording)

`run_dense_server.sh` serves `checkpoints/pi05_hanoi_dense_aaaa_to_cccc/hanoi_dense_20260919/exports/29999` (config
`pi05_hanoi_dense_aaaa_to_cccc`, dense contract five: every recorded row an observation, 30 absolute reference poses
at 10 Hz as the label, execution prefix 3; test split slot-1 error 0.59 mm, jaw 99.9%) on port 8000 through
[serve_dense.py](serve_dense.py). `run_dense_client.sh` drives the arm with [dense_client.py](dense_client.py):

```bash
./run_dense_server.sh                                  # warms up, then listens on 8000
./run_dense_client.sh --mode replay --duration-s 30    # raw single-episode extract, no ROS or arm
./run_dense_client.sh --mode live --duration-s 180     # v4 start pose, joint check, then policy control
```

How it executes, and how it differs from the waypoint clients:

- Observations are taken every 30 Hz tick while the arm moves (the training rows were taken from a moving arm),
  one inference is kept in flight, and a newer chunk replaces the uncommitted rows. Row `k` of a chunk is due
  `3 (k + 1)` ticks after its observation tick.
- At each segment boundary [dense_execution.py](dense_execution.py) takes the newest chunk's rows not yet due and
  executes the next `--prefix-rows` (default 9, 0.9 s) as one quintic segment: it starts from the commanded
  position, velocity and acceleration, ends at the last row when that row is due, with end velocity and
  acceleration read off the neighbouring rows (least-squares quadratic through five rows). Consecutive segments
  are therefore velocity-continuous, and the driver receives the endpoint, the duration and the feedforward end
  derivatives as before. Live run 1 (2026-09-20) used 3-row segments and stop-started at up to 10 Hz: the model's
  state input has no velocity, so it predicts the demonstration's cruise speed whatever the arm is doing, and once
  a segment was slowed the rows due next were unreachable within the limits, the executor braked, and the arm got
  slower still (90 brakes in 134 segments, median speed 7 mm/s). Re-planning that run's chunks with 9-row segments
  gives no brakes at a mean 27 mm/s; grasp poses still come from slot 1 to 2 of a fresh chunk because the prefix is
  cut at every jaw change.
- Limits: the recorder's velocity and acceleration limits with a 10% margin (0.16 m/s, 0.30 m/s²) and a tracking
  jerk bound of 20 m/s³ instead of the recorder's 1.3, which was chosen for whole 2 s legs; stitching short segments
  from rows with 0.7 mm noise needs the larger bound. A segment over the limits is slowed down (up to
  `--max-segment-stretch`, 6x);
  one that cannot be followed at all makes the executor brake to rest along its motion, after which the next chunk
  is predicted from a stopped arm. Both are counted in the summary (`stretched_segments`, `brakes`).
- A jaw change is executed at rest at its row's pose (alignment at the recorder's rest-to-rest pace if needed),
  followed by the commissioned dwell, and a fresh observation is required afterwards. Workspace bounds, the
  tracking watchdog (8 mm against the dispatched segment), the grasp-stroke check and the recovery on stop are the
  same as the waypoint client's, and the run ends only at a segment boundary with nothing held. Every stop, the
  duration, Ctrl-C, a missed grasp, a rejected command or an unexpected failure, opens the gripper and returns to
  joint home. Two checks were added after live run 2 (16-step variant, 2026-09-20): the measured tool tilt allowed
  while moving is `--max-orientation-error-deg` (3; the commissioned 2 degree gate stopped that run mid-carry at
  2.05, and each tick now logs `orientation_error_deg`), and a sideways move with the ring held below
  `--carry-min-z-m` (0.17) is rejected and the run stops. The recording carries at 191 mm and descends only above the
  target column; run 2's model descended to 151 mm while still over peg C and travelled between the pegs at that
  height, the "lift after grasp" versus "arrival before release" ambiguity the 16-step variant also showed offline.
  The rule is a safety stop on the model's plan, never a correction of it; it would have fired once in run 2 and never
  in the solved run.
- Replay plans from the recorded commanded state (position, velocity and acceleration of `reference_pose`) so its
  stretch and brake counts predict live behaviour, and reports the policy's per-slot error against the recorded
  reference (`reference_error_mm` on each prediction, `replay_slot1_error_mean_mm` in the summary). Episode 40,
  30 s: 57 segments, 7 stretched, 1 brake, 0 rejections, two ring moves, slot-1 error 0.65 mm.

Start pose: `--start above_peg_a` (the dense client's default) puts the arm at the recorded grasp hover over peg A,
(496.2, -57.4, 191.1) mm, joints (-0.1558, 1.7618, 1.5509, -0.5804, -0.1101, -0.1109) rad, jaw open. That is where
every training episode arrives after its first leg and from which the first grasp descends, so a run started there
is comparable with methods that begin above peg A; from that state the model's first chunk is the descent to the
top ring with the jaw closing at about 2 s. `--start episode_start` uses the episode start behind peg B instead.
The carry hover over A at (492.3, -56.3) is not offered: the recording only releases from it. Both poses are
available to the waypoint client (`cosmos_client.py --start`) and the pose tester (`test_robot_initial_pose.sh
--start above_peg_a`), and the joint-space start check uses the matching recorded joints.

First full completion on hardware (2026-09-20, `dense_live_1789936407839752778`, 30-step variant, 9-row segments, start
above peg A, raw output): all 15 moves of the solution in 270 s, every move legal, no null moves, no brakes, no rejected
commands, every grasp within 1.1 mm of its ring level, grip strokes 15 / 20 / 25 / 31 mm for rings 1 to 4, tracking
error 2.0 mm p95. The operator stopped the run 18 s after the last release and the arm returned home.

Camera recording: `--record-bag` on either client starts `ros2 bag record` of the full-frame camera stream and its
camera_info into `<run>/camera_bag` (mcap, zstd_fast by default) for the whole run, robot initialization through the
return home, and writes `bag_started`/`bag_stopped` events and a `camera_bag` summary entry with duration, size and
how the recorder closed. The camera publishes 640 x 480 rgb8 at 60 Hz: about 24 MB/s with zstd_fast (7 GB for five
minutes), 55 MB/s with `--bag-storage-preset none`. `--bag-topics` overrides the topic list. The recorder is a
separate process; the saved 224 x 224 crops under `inference_inputs/` remain the exact model inputs.

Cosmos dense server: the same client drives the Cosmos dense contract-five checkpoint
(`hanoi_cosmos_dense_20260919_video_init_cycle2/exports/iter_000016000.pt`, video-base init, two training cycles,
16-step chunk, execution prefix 8, test slot-1 error 0.80 mm) through `cosmos_policy.experiments.robot.hanoi.serve_dense`
in the Cosmos checkout, which publishes the `hanoi_dense` identity under config name `cosmos_hanoi_dense_v5_h16`:

```bash
# Terminal 1, in ../cosmos-policy
source examples/hanoi/env.sh
.venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_dense --port 8001
# Terminal 2, here
./run_dense_client.sh --server ws://127.0.0.1:8001 --mode live --duration-s 180 --record-bag
```

The client accepts execution prefixes 3 and 8 and requires `--prefix-rows` between the server's prefix and its chunk
length minus four rows of latency slack (9 fits both). Cosmos inference is 0.5 s on the RTX 5080, so 5 to 9 rows of
each 16-row chunk are already due when a segment starts; a chunk older than its horizon is skipped (`expired_predictions`
in the summary) and the client waits for a fresh one. Replay from the hover over A, 30 s: 22 segments, 1 slowed,
0 brakes, 0 rejections, four gripper events on the recorded points, slot-1 error 0.69 mm. The summary's `policy_family`
is `cosmos_dense` and the adapter name starts with it. Only one server fits on the GPU at a time.

Two trained variants exist, identical except for the chunk length, and the client sizes its executor and worker
from the `action_horizon` the server announces (30 or 16); the identity check accepts only the `hanoi_dense`
identity with contract version 5 and the selected export of the announced config name:

| Config | Chunk | Export | Test slot-1 error |
|---|---|---|---|
| `pi05_hanoi_dense_aaaa_to_cccc` (default) | 30 steps, 3.0 s | `hanoi_dense_20260919/exports/29999` | 0.59 mm |
| `pi05_hanoi_dense_h16_aaaa_to_cccc` | 16 steps, 1.6 s (the Cosmos horizon) | `hanoi_dense_h16_20260920/exports/29999` | 0.57 mm |

`./run_dense_server.sh --config-name pi05_hanoi_dense_h16_aaaa_to_cccc` serves the second; the client needs no flag
and records `config_name`, `action_horizon` and an adapter name with the horizon (`pi05_dense_v5_h16_track_prefix_3`).
The h16 run's `complete.json` says `serving_parity_passed: false`; the cluster's probe showed the served path
reproduces the single-observation evaluator exactly for both variants, and the reported 0.58 mm gap is bfloat16
batch-shape numerics inside the batched evaluator, so the served checkpoint is the one that was scored. Replay on
episode 40, 60 s: 127 segments, 9 stretched, 1 brake, 0 rejections, seven gripper events on the recorded points,
slot-1 error 0.66 mm (30-step variant: 121 segments, 12 stretched, 2 brakes, 0.64 mm).
Note from the training handover: the agent kept pi0.5's discretised state input (decision 5 of the guide) because
`discrete_state_input=False` drops the state from this model entirely, so the state-shortcut risk found in the
waypoint run still exists in principle; the dense data has no fixed grasp/release columns, which removes the cue.

## Files

| File | Purpose |
| --- | --- |
| [run_policy_server.sh](../../../run_policy_server.sh) | Server launcher with model environment and GPU defaults. |
| [run_policy_client.sh](../../../run_policy_client.sh) | Client launcher with ROS setup; defaults to live mode for 30 seconds. |
| [test_robot_initial_pose.sh](../../../test_robot_initial_pose.sh) | Move to a start pose (`--start rod_a` or `cosmos_v4`), test readback, and return home. |
| [initialize.py](initialize.py) | Camera-free initialization command and result report. |
| [workspace.json](workspace.json) | Default workspace bounds derived from the transferred episode. |
| [client.py](client.py) | Replay/shadow/live entrypoint, control loop, watchdogs, logs, and shutdown. |
| [serve.py](serve.py) | Policy server with selected-checkpoint identity metadata. |
| [async_inference.py](async_inference.py) | Background worker, OpenPI transport, and timestamped action buffer. |
| [hardware.py](hardware.py) | ROS camera, Trossen feedback/dispatch, and polynomial bounds. |
| [execution.py](execution.py) | Learned-target motion adapter and strict offline teacher reconstruction. |
| [../tests/async_inference_test.py](../tests/async_inference_test.py) | Timing, transport, control-loop, and hardware-adapter tests. |
| [requirements-robot.txt](requirements-robot.txt) | Python 3.12 robot client dependencies. |
| [validation/](validation/README.md) | Checkpoint smoke inference, recorded-episode accuracy, and plots. |

Shared contract helpers remain outside this deployment folder:

- [openpi_client/hanoi.py](../../../packages/openpi-client/src/openpi_client/hanoi.py): lightweight contract/camera
  helpers; the training policy re-exports the same helpers and retains its original behavior.
