# Hanoi pi0.5 deployment handoff

Local deployment code and current launch commands are in the [deployment guide](../examples/hanoi/deployment/README.md).

Prepared 2026-09-15 from the code and saved evaluation evidence on HPC. This handoff is for the local hardware agent.
The user has one Trossen arm, one external RealSense RGB camera over ROS, an RTX 5080, and an already working local
OpenPI policy server for another task. Reuse that server environment. Develop the Hanoi integration locally on
`hardware/trossen-hanoi`, based on `hpc` commit `ab73718472667ba7f97ea792f32ab25e0602ab26` in
[fefei69/openpi](https://github.com/fefei69/openpi/tree/hpc).

The immediate deployment target is the selected **AAAA to CCCC** policy. Its 30,000 optimizer updates, validation
selection, held-out test, and HPC serving parity have passed. Physical robot trials have not been performed.
This document does not certify the readiness of the reverse or multitask model; obtain their own selected exports
and completion evidence when available. Changing the prompt of the forward model does not make it the reverse model.

## Read these files first

| File | Purpose |
| --- | --- |
| [hanoi_policy.py](../src/openpi/policies/hanoi_policy.py) | Canonical camera decoder/crop, seven-value state, missing-camera masks, prompts, and contract constants. |
| [config.py](../src/openpi/training/config.py) | `LeRobotHanoiDataConfig` and the three `pi05_hanoi_*` configurations; XYZ delta/inverse transforms. |
| [execution.py](../examples/hanoi/deployment/execution.py) | Hardware-free trajectory and gripper command builder. It sends no robot commands. |
| [execution_test.py](../examples/hanoi/tests/execution_test.py) | Timing, gripper dwell, continuity, and rejection examples. |
| [policy_config.py](../src/openpi/policies/policy_config.py) | Public checkpoint-loading factory and normalization/output transforms. |
| [serve_policy.py](../scripts/serve_policy.py) | Existing WebSocket policy server. |
| [verify_serving.py](../examples/hanoi/evaluation/verify_serving.py) | Numerical comparison between evaluation and the public serving factory. |
| [Hanoi README](../examples/hanoi/README.md) | Directory overview and key entrypoints. |
| [Training guide](../examples/hanoi/training/README.md) | Pipeline operations and existing validation commands. |
| [Dataset handoff](hanoi_dataset_handoff.md) | Raw file locations, field definitions, alignment, and split details. |
| [Training plan](hanoi_training_plan.md) | Full training and deployment requirements; job observations there are dated. |

The [Trossen OpenPI tutorial](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/openpi.html) is useful
background for the server/client workflow. Its example uses a different camera mapping and robot configuration.
Use this repository's Hanoi contract for this single-arm, single-camera Cartesian policy.

## Copy the selected checkpoint

Run on the local PC from the repository root. `checkpoints/` is already ignored by Git.

```bash
mkdir -p checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999

rsync -avhP --append-verify \
  cw5167@login.torch.hpc.nyu.edu:/scratch/cw5167/workspace/openpi/checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999/ \
  checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999/

mkdir -p data/hanoi/deployment

rsync -avhP \
  cw5167@login.torch.hpc.nyu.edu:/scratch/cw5167/workspace/openpi/data/hanoi/runs/hanoi_20260914/evaluation_17838495_audited.json \
  data/hanoi/deployment/

rsync -avhP \
  cw5167@login.torch.hpc.nyu.edu:/scratch/cw5167/workspace/openpi/checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/hanoi_identity.json \
  data/hanoi/deployment/forward_hanoi_identity.json
```

The export is approximately **12.44 GB** and contains `params/`, `assets/`, and `export.json`. Copy the entire export,
including `assets/local/hanoi_roundtrip_20260910/norm_stats.json`. Serve the `29999` directory itself. The export uses
JAX/Orbax parameters; it is not a converted PyTorch checkpoint. Use the matching checkpoint loader in the existing
server, or separately validate any format/backend conversion.

Inference construction does not require the raw dataset, converted dataset, training indices, or optimizer state.
The checkpoint's own normalization must be used. Do not substitute statistics from another task or recompute them
from live observations. The parent `hanoi_identity.json` and audit record are provenance evidence, not model inputs.

Verify the local files against the saved native audit before loading them:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999")
audit = json.loads(Path("data/hanoi/deployment/evaluation_17838495_audited.json").read_text())
assert audit["passed"]
assert audit["model"]["selected_step"] == 29999
for name, expected in audit["model"]["checkpoint_files"].items():
    path = root / name
    assert path.is_file() and path.stat().st_size == expected["bytes"], path
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    assert digest.hexdigest() == expected["sha256"], path
print("Selected export file sizes and SHA-256 checksums match the HPC audit.")
PY
```

The audited forward normalization SHA-256 is
`ec82035dd64b6715d75644addc7b040bb4d347ec3a3f79b730ba7930590c1c48`.
The export manifest SHA-256 is `38b1e39be734df97a49836f8ede9238f1e97335af724e4a8448a50c9bf5b4b88`.
`validate_export` in `evaluation/metrics.py` checks manifest structure and file sizes; the code above additionally checks
content hashes. Preserve the numeric directory name `29999` if using `validate_export`.

## Load the policy in the existing server

With the user's existing working Python environment active, the repository server command is:

```bash
python scripts/serve_policy.py --port 8000 \
  --default-prompt 'Move all four rings from peg A to peg C following Tower of Hanoi rules.' \
  policy:checkpoint \
  --policy.config pi05_hanoi_aaaa_to_cccc \
  --policy.dir checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999
```

Use an available port if another policy server is running. For an existing custom launcher, load through
`policy_config.create_trained_policy(config.get_config("pi05_hanoi_aaaa_to_cccc"), checkpoint_dir,
sample_kwargs={"num_steps": 10})`. The stock JAX sampler already defaults to ten steps. Keep ten sampling steps for
the initial deployment because that is the evaluated setting. Record a local environment/version snapshot when the
Hanoi server works; this handoff does not require rebuilding the user's working GPU environment.

## Observation contract

Each request is an unbatched dictionary. Use the existing `openpi_client.websocket_client_policy.WebsocketClientPolicy`
and its `infer(observation)` method for transport; the client handles NumPy serialization.

```python
observation = {
    "observation/image": cropped_rgb_uint8,  # uint8, (224, 224, 3), RGB/HWC
    "observation/state": measured_state,    # float32, (7,), finite
    "prompt": "Move all four rings from peg A to peg C following Tower of Hanoi rules.",
}
```

| State index | Quantity | Units |
| --- | --- | --- |
| 0, 1, 2 | Measured tool X, Y, Z in the commissioned base/tool frame | metres |
| 3, 4, 5 | Measured tool linear velocity X, Y, Z in the same frame | metres/second |
| 6 | Measured driver jaw stroke | metres |

The state is Cartesian, not joint positions. Jaw stroke is the driver's stroke readback, not pad separation. Use
the original measurement/kinematics convention for velocity; commanded velocities are not measured velocities.
The raw dataset's eighth proprioception value is a jaw target and must never be sent as an observation. Symbolic
board states, routes, phase labels, and intended next moves are not policy inputs.

The RGB path is:

1. Subscribe to `/camera/camera/color/image_raw` with a latest-frame/depth-one queue.
2. Decode a 640x480 `rgb8` image using `decode_ros_rgb`, passing `height`, `width`, `step`, `encoding`, and data.
   It honors row padding and rejects a different encoding; explicitly convert a different ROS encoding if needed.
3. Call `preprocess_camera` once: crop `rgb[90:450, 151:511]`, then resize to 224x224 with `cv2.INTER_AREA`.
4. Send the resulting RGB bytes. HDF5/LeRobot images are already cropped and resized; do not crop them again.

`HanoiInputs` maps this one view to `base_0_rgb`, fills both absent wrist-camera slots with zeros, and sets their
masks to `False`. Do not duplicate the external camera into the wrist slots or expose them with true masks. The
shared policy transforms perform model normalization and padding; the robot client should not duplicate them.

Match the collection camera pose, intrinsics/resolution, peg layout, tool offset, coordinate conventions, and fixed
tool orientation RPY **(0, pi/4, 0)** radians. The code's `commissioned_base_tool_frame` is a contract label, not an
actual ROS TF frame name. Recover the concrete calibration and driver mapping from the user's local collection setup;
the HPC repository does not establish those machine-specific values.

## Action and timing contract

**Local implementation update (2026-09-15):** the hardware client now uses `PolicyExecutor`, described in the
[deployment guide](../examples/hanoi/deployment/README.md#how-learned-actions-are-executed). It selects a target
from up to nine future references and uses a rest-to-rest move whose duration respects the motion limits.
The strict reconstruction requirements below describe the original teacher-equivalence design, retained in
`ReferenceExecutor` for offline validation; they are no longer live rejection criteria. Endpoint execution changes
intermediate positions and timing, so the old teacher-equivalence audit does not certify this new adapter.

The serving result's `actions` is a finite **(63, 4)** array: absolute base-frame reference XYZ in metres, followed
by jaw-open intent. Threshold the jaw channel at **0.5**: values at or above it mean open; lower values mean close.
It is neither a joint command nor a requested jaw position in metres.

Training uses `action_abs[t:t+63]` with no added `t+1` label shift. XYZ is represented inside the model as a delta
from the observation's measured XYZ. The serving output transform restores absolute XYZ. Do not cumulatively sum
outputs or add the measured position again. Padding to 32 action dimensions is internal; only four channels reach
the client.

Although collection used sparse Cartesian waypoints, the saved training targets are the dense, noisy 30 Hz
next-reference trajectory. Keep its **63-reference / 2.1-second horizon** and initially commit a
**nine-reference / 0.3-second prefix** before replanning. Replacing the prefix by a rest-to-rest move to its final
point changes the trajectory; the recorded noise was intentional, and endpoint-only replacement added up to 5.7 mm
intermediate deviation in the audit.

`ReferenceExecutor.plan` produces one `cartesian`, `gripper`, `hold`, or `wait` command. For Cartesian motion it fits
all nine future XYZ references, preserves prior position/velocity/acceleration boundary conditions, and supplies
quintic coefficients. `Command.sample` provides position or derivatives along that command. The hardware adapter
must realize this reference trajectory over the committed duration using supported driver control. A generic
endpoint-only API is insufficient unless it can preserve these boundary conditions and intermediate references.

The offline builder rejects fits above 0.1 mm reference reconstruction error and sampled speed/acceleration/jerk
above 0.145 m/s, 0.23 m/s², and 1.3 m/s³ respectively. These are implemented reference checks, not measured hardware
tracking guarantees. Its strict checks have passed on teacher references; learned predictions can still be rejected.
Handle a rejection explicitly rather than silently clipping, weakening checks, or sending an unchecked endpoint.

Implement timing using one local monotonic clock and 30 Hz tick mapping:

- Record image receipt time and the observation anchor time. Training eligibility used anchor-command time minus
  image receipt time in the range 0 to 50 ms. Preserve this freshness definition; keep sensor capture/header time
  as additional diagnostics rather than mixing it with receipt time or clocks from another host.
- Pass actual inference/transport elapsed time through `observation_tick` and `now_tick`. Output row `k` is due at
  `observation_tick + k + 1`. The builder uses `ceil(now_tick)` and skips elapsed rows; require nine usable future
  references. Do not replay the start of an old chunk after inference latency.
- Honor `available_tick`: a pending command must finish before issuing another one. Measure end-to-end inference,
  transport, scheduling, and driver timing after warmup; the nine-tick execution interval is not an assumed GPU latency.
- `plan` advances its internal reference state when a command is accepted. If driver dispatch fails or tracking
  diverges, stop and reconcile/reset the executor with measured state before continuing. Do not retain a fictitious
  successfully executed trajectory.

The policy server does not enforce these timing rules. They belong in the local client/control loop.

## Gripper and episode behavior

| Event | Commissioned behavior |
| --- | --- |
| Open | Stroke 0.034 m over 1.0 s / 30 ticks. |
| Close | Effort -20 N over 1.2 s plus 0.2 s settling / 42 ticks total. |
| Repeated intent | Keep the state; do not restart the dwell. |
| During jaw actuation | No Cartesian movement; wait for the commissioned dwell. |
| After jaw actuation | Discard the old chunk and obtain a fresh observation before replanning. |

The actual driver call, sign/unit mapping, and measured completion handling remain to be implemented against the
local hardware API. Initialize executor position, velocity, acceleration, and jaw state consistently with a known
start condition. On episode or task changes, clear queued actions and reset executor state. WebSocket client `reset()`
is a no-op in this checkout; it does not reset the hardware executor for you.

Use confirmed goal-board state together with the commissioned settled/open arm condition to end an episode.
Terminal hold padding is not a learned stop signal. Runtime goal checking may use external logic; do not add its
privileged annotations to the policy's observation dictionary.

Exact task texts, available as `hanoi_policy.PROMPTS`, are:

- AAAA to CCCC: `Move all four rings from peg A to peg C following Tower of Hanoi rules.`
- CCCC to AAAA: `Move all four rings from peg C to peg A following Tower of Hanoi rules.`

The multitask configuration is `pi05_hanoi_multitask`; always send the chosen direction's prompt and reset between
tasks. Each model needs its own selected checkpoint and normalization assets.

## Local implementation and acceptance work

1. Verify the copied checkpoint checksums and make the existing server load the Hanoi configuration. Query server
   metadata and compare its fields with `hanoi_policy.CONTRACT`; validate actual request/output shapes as well.
2. Implement the ROS camera/state adapter and a recording mode that sends no robot commands. Compare a real live
   crop and the measured state with collection conventions, including timestamps, units, and camera masks.
3. Add the driver bridge around `ReferenceExecutor`: continuous Cartesian execution, gripper commands, waits,
   timing/rejection handling, tracking checks, episode reset, and goal confirmation. Reuse the user's calibrated
   workspace/IK bounds and stop mechanism. Their local values must come from the hardware setup.
4. Run the existing focused contract tests in the local environment:

   ```bash
   python -m pytest src/openpi/policies/hanoi_policy_test.py examples/hanoi/tests/execution_test.py -q
   ```

5. Repeat Hanoi-specific numerical serving parity locally with the same observation and explicit sampling noise.
   The HPC check uses a training anchor, NumPy RNG seed 44 with noise shape (63, 32), ten sampling steps,
   `atol=1e-6`, `rtol=1e-5`, and identical thresholded jaw decisions. The stock WebSocket request does not expose
   `Policy.infer(..., noise=...)`; use the in-process factory for this deterministic comparison and test transport
   separately. `evaluation/verify_serving.py` is an evaluation helper that requires the converted dataset and HPC run metadata,
   not a standalone dataset-free CLI. A portable training-anchor/expected-action fixture is still to be exported
   or the existing harness must be supplied with its data dependencies. The audit JSON contains metrics, not that
   fixture. Document any cross-runtime numerical difference before accepting it.
6. Record inference and command timing after warmup and replay teacher references through the local driver bridge
   without motion. Check all prefix offsets, gripper boundaries, stale frames, expired chunks, and disconnects.
   Then perform controlled physical trials using the user's commissioned limits and stop procedure. Report task
   success and tracking behavior separately from offline prediction error.

Follow existing code conventions: focused tests under `examples/hanoi/tests/`, existing imports/logging style,
dataclass configuration, and `tyro` for a new CLI. Keep hardware integration under `examples/hanoi/deployment/` and
reuse shared contract helpers. Keep
ROS/driver dependencies in the appropriate existing local client environment. Preserve the HPC training code and
running controllers while developing locally.

Deliver a documented launch command, calibration/configuration mapping, camera/state sample, checkpoint identity,
local environment snapshot, parity result, timing measurements, execution logs, and physical trial results.
Generic GPU setup is already handled by the user; the remaining work is the Hanoi-specific integration above.

## Existing evidence and its limits

The native forward audit passed at **2026-09-15 10:19:20 UTC** and verified 30,000 actual optimizer updates despite
the final export's zero-based label `29999`. Validation selected that export before held-out testing.
On 320 sampled held-out anchors, mean valid-reference XYZ error was **2.2702 mm**, first-reference error
**0.3572 mm**, and jaw balanced accuracy **99.9068%**. These are offline predictions, not hardware success rates.
HPC serving/evaluation parity passed with a maximum XYZ difference of about **2.25e-9 m** and identical jaw decisions.

Teacher-reference replay passed 63,000 Cartesian commands and 3,000 gripper events. Neither that replay nor the
serving audit exercised the real robot. The robot client, local numerical/timing validation, and physical task
assessment are the deployment agent's remaining work.
