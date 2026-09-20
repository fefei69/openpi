# Hanoi training and pipeline operations

[Directory overview](../README.md) · [Deployment guide](../deployment/README.md) ·
[Training plan](../../../docs/hanoi_training_plan.md)

This guide covers dataset preparation, qualification, training, job management, evaluation, and delivery.
Run commands from the repository root. Dated progress below is retained as an operational snapshot; see
`data/hanoi/PROGRESS.md` on HPC for subsequent observations.

## Saved run identities

This directory reorganization changes source paths and hashes. Saved run evidence is preserved, and source-identity
checks remain strict. Use the matching original source snapshot to audit or resume historical runs. This cleanup
includes no provenance migration or identity-check bypass.

## Recorded progress

- Source audit: all 100 episodes / 720,100 rows passed; 14,208 stale observation anchors are excluded without removing
  their underlying action rows. Forward/reverse pairs share chronological train/validation/test assignments.
- Teacher execution: all 63,000 Cartesian commands and 3,000 gripper events passed. Maximum added reference error is
  0.000401 mm; maximum speed, acceleration, and jerk are 0.129864 m/s, 0.197913 m/s², and 1.163221 m/s³.
- Three independent pi0.5 configurations, shared input/output transforms, and training-only normalization are present.
- An 18-frame visual spot check covers both directions in training, validation, and test. The sampled start/final
  stacks and transfer direction match the labels; details are in `data/hanoi/visual_review.json`.
  A separate color-region check passes all 200 episode start/final images for the expected peg and ring order.
  Its thresholds and per-frame measurements are saved in the same report; intermediate images are not exhaustively checked.
- CPU preparation job `17736379` completed successfully: all 100 episodes / 720,100 frames converted, normalization
  refreshed, trajectory replay passed, and full native-data verification passed. The latter checks every numeric row
  and 2,300 RGB/chunk/input-parity probes. The final pretraining gate also passed.
- The persistent manager is active in the existing CPU allocation. Its host PID and allocation expiration are recorded
  in `data/hanoi/runs/hanoi_20260914/monitor_process.json`; state and exact submitted arguments are in `manager.json`.
  H100 qualification `17758983` completed successfully, including batch-32/horizon-63 training, optimizer resume,
  EMA evaluation/sampling, and native serving. Optional two-A100 qualification `17760911` was deferred while pending.
  At **2026-09-14 23:29 UTC**, AAAA→CCCC is running as continuation `17789900` on H100 node `gh007`, with checkpoint
  16,125 committed and three retained EMA snapshots. The earlier allocation `17769392` was canceled; the current
  job has restored its state and repeatedly saved checkpoints successfully. Reverse and multitask have not started.
  Actual throughput including checkpoint saves is about 15 hours per model, with 37 training hours remaining
  across all three at this observation, plus queue, evaluation, and restart time. See
  `data/hanoi/PROGRESS.md` on HPC for later observations.
  The monitor manages subsequent training, checkpoint continuations, and evaluation stages.
- Inference export retention and offline validation/selection/test are implemented and locally tested. Final
  validation and test evaluation await the production checkpoints.
- The monitor bounds optional qualification waits and persists deferrals across restarts. Once an optional pilot is
  deferred, it proceeds with qualified hardware. It also checks storage before production submission and again at launch.

## W&B for the remaining runs

At the user's request, online metric tracking is enabled for `cccc_to_aaaa` and `multitask` through the independent
CPU logger `examples.hanoi.pipeline.wandb_logger`. It publishes loss, gradient/parameter norms, allocated-GPU telemetry and
checkpoint progress to `cw5167-nyu/openpi`. The original `aaaa_to_cccc` run remains on local logging.

The qualified trainer's native `wandb_enabled=False` flags stay unchanged: the CPU logger reads the actual Slurm
stdout and telemetry, preserving the original training-code and checkpoint identities. Training scalars retain the
stdout log's four-decimal precision. This avoids interrupting or
invalidating the ongoing fine-tune. It uploads metric/config metadata; dataset images and checkpoint weights are
not uploaded. Checkpoints remain on scratch.

The logger waits for each requested model's submission, uses one persistent W&B ID per model across continuations,
and uses the recorded optimizer step as a custom plot axis so checkpoint rollback does not discard metrics. On
logger restart, it reads acknowledged W&B history and replays missing local events. A network failure is retried
independently of training. The logger is separate from the Slurm manager and must be kept alive too.

Run from the repository root in a CPU allocation, after checking that the existing logger is not already running:

```bash
source examples/hanoi/scripts/env.sh
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hanoi.pipeline.wandb_logger \
  --entity cw5167-nyu --project openpi --exp-name hanoi_20260914
```

Its lock and state live in `data/hanoi/runs/hanoi_20260914/`: `wandb_logging.json` contains destinations and stable
run IDs; `wandb_logger.json` contains the PID, host, process start ticks, heartbeat and run URLs once created.
The logger uses existing W&B authentication; credentials are not stored in those files. W&B runs appear when
their models are submitted, and `pipeline_state`/`latest_job_state` distinguish waiting from training.

## Environment and CPU preparation

Run commands from the repository root. Batch scripts use the working directory supplied by Slurm; the manager sets
`--chdir` to that repository root. Use a CPU allocation with 8 CPUs and 128 GiB for bulk conversion; the native
LeRobot writer retains embedded image tables while appending episodes. The existing interactive `cpu-light` allocation
has only 2 CPUs and 4 GiB, regardless of the physical node's capacity.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group hanoi
source examples/hanoi/scripts/env.sh
export JAX_PLATFORMS=cpu
.venv/bin/python -m examples.hanoi.data.convert_hanoi_data_to_lerobot --resume
.venv/bin/python -m examples.hanoi.data.compute_norm_stats
.venv/bin/python -m examples.hanoi.evaluation.verify_execution
.venv/bin/python -m examples.hanoi.data.verify_data
```

`h5py==3.16.0` is needed
for the recording's HDF5 2.0 Boolean metadata. The original LeRobot Git revision and JAX/model dependencies are retained.
The converter reads raw files from `/scratch/cw5167/datasets/`, writes the local LeRobot dataset under
`$HF_LEROBOT_HOME/local/hanoi_roundtrip_20260910`, and stores audits and frame selections under `data/hanoi/`.
It never uploads data or overwrites a source recording. `--resume` checks source identity and preserves interrupted
generated files under `data/hanoi/interrupted/` before retrying their episode.

The submitted CPU preparation job is recorded in `data/hanoi/runs/hanoi_20260914/preparation.json`; it uses
`examples/hanoi/scripts/prepare.sbatch`. Its final gate checks every converted state/action row, sampled exact RGB/chunk values, episode
boundaries, task prompts, normalization inverse, and canonical training/serving input equality.

## Model and serving contract

Configurations are `pi05_hanoi_aaaa_to_cccc`, `pi05_hanoi_cccc_to_aaaa`, and `pi05_hanoi_multitask`. All use full JAX
fine-tuning, batch 32, horizon 63, action dimension 32 with padding, and 30,000 steps. Each has separate normalization
assets and checkpoints. Multitask episodes carry their direction's prompt from `hanoi_policy.PROMPTS`.

Serving inputs use the following keys:

```python
{
    "observation/image": cropped_rgb_uint8,  # (224, 224, 3)
    "observation/state": measured_state,    # XYZ, velocity XYZ, jaw stroke; shape (7,)
    "prompt": "Move all four rings from peg A to peg C following Tower of Hanoi rules.",
}
```

The shared adapter maps RGB to `base_0_rgb` and masks both absent wrist cameras. `decode_ros_rgb` honors `rgb8` row
stride, and `preprocess_camera` applies the collection crop once to a 640x480 source. HDF5/LeRobot images are already
cropped. The target-command field `proprio[7]` and privileged board/route labels never enter the model input.

Targets are `action_abs[t:t+63]`, with no additional time shift. XYZ deltas are relative to the anchor's measured
position; jaw intent stays absolute. The serving inverse returns 63 absolute XYZ/jaw references. The executor preserves
Cartesian endpoint derivatives, honors committed commands, skips expired references, and requires fresh observations
after 1.0-second opening or 1.4-second closing/settling. An endpoint-only rest-to-rest driver call does not meet this
contract. Sampled derivative checks in the offline builder do not replace live workspace, IK, tracking, or timing tests.

## Deployment and local validation

The deployment code and its launch instructions live in [deployment/](../deployment/README.md).
The [local validation tools](../deployment/validation/README.md) cover checkpoint smoke tests and action accuracy on
recorded episodes. This guide covers the training and offline pipeline separately.

## Qualification and training launchers

`examples/hanoi/scripts/qualify.sbatch` runs 100 measured full training steps after warmup, saves/restores optimizer state, checks another
optimizer step, restored EMA evaluation and inference, and the native serving factory. It records throughput, GPU
memory, and actual checkpoint sizes. `examples/hanoi/scripts/train.sbatch` checks preparation, normalization, selection, environment, and
qualification identities, obtains the pipeline/writer locks, verifies resume identity, and calls `scripts/train.py`.
Neither launcher selects a partition or QoS; Torch assigns them from account and resources.

These are example argument shapes; submit only after preparation gates pass and select resources from fresh estimates:

```bash
sbatch --account=torch_pr_595_tandon_advanced --nodes=1 --ntasks=1 \
  --gres=gpu:1 --constraint=h100 --cpus-per-task=8 --mem=128G --time=01:00:00 \
  examples/hanoi/scripts/qualify.sbatch --profile h100_1 --exp-name qualification_h100_1 \
  --output-path data/hanoi/runs/hanoi_20260914/h100_1.json --fsdp-devices 1

sbatch --account=torch_pr_595_tandon_advanced --nodes=1 --ntasks=1 \
  --gres=gpu:1 --constraint=h100 --cpus-per-task=8 --mem=128G --time=12:00:00 \
  examples/hanoi/scripts/train.sbatch --config-name pi05_hanoi_aaaa_to_cccc --exp-name hanoi_20260914 \
  --qualification-path data/hanoi/runs/hanoi_20260914/h100_1.json
```

## Monitoring, retention, and evaluation

The manager polls every ten minutes and owns only its recorded jobs. It reconciles uncertain submissions by unique
name, checks stage outputs before advancing, and requires confirmed pending cancellation before queue replacement.
It waits for scheduler requeue, caps retries and no-progress restarts, and preserves the experiment directory.
Shared GPU quota waits receive bounded replacement checks. Fresh queued runs can change qualified device count;
recorded placements and checkpoints retain the cross-mesh restore requirement.
The initial GPU budget is two; scratch sizing further restricts training/evaluation to one model at a time.
`telemetry.jsonl` records allocated GPU utilization/memory, logged training progress, and completed-checkpoint age.

```bash
source examples/hanoi/scripts/env.sh
# Inspect one pass without submitting work; the manager lock prevents a second controller.
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hanoi.pipeline.manage --exp-name hanoi_20260914
# Run an owned controller in an appropriate CPU session when no controller is active.
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hanoi.pipeline.manage --exp-name hanoi_20260914 --submit --watch
```

Every 5,000 steps and at the final step, atomic hard-linked exports retain EMA parameters and normalization assets
without retaining another optimizer state. The latest complete full checkpoint remains resumable. Abandoned export
staging directories are reclaimed when training resumes. A real pilot sizes the quota forecast before production.

`examples/hanoi/scripts/evaluate.sbatch` evaluates all eligible validation anchors, including the last partial batch, and samples up to 64
anchors per episode for physical XYZ/jaw metrics. It reports episode/direction breakdowns and a multitask prompt-swap
diagnostic. Validation flow loss selects one EMA snapshot; selection is locked before test. Completed metrics are
cached with identities, and cleanup can resume without repeating test. Unselected exports are removed after each model.
Each validation/test result records `timing_seconds` for setup, flow evaluation, physical sampling, cleanup, and
the total. Timings include data loading and compilation; they measure evaluation stages, not robot inference latency.
Flow-progress logs include elapsed seconds so the remaining evaluation time can be estimated during the first pass.
Before completion, [evaluation/verify_serving.py](../evaluation/verify_serving.py) reopens the selected checkpoint through `create_trained_policy` and compares
its absolute actions with the evaluation path using the same observation and explicit sampling noise. It tests both
directions for multitask and requires identical jaw decisions. These probes use training anchors and do not select
models using test outcomes.

```bash
.venv/bin/python -m examples.hanoi.evaluation.evaluate --config-name pi05_hanoi_multitask --exp-name hanoi_20260914
# After selection, use its checkpoint path from selection.json with the existing server:
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_hanoi_multitask --policy.dir CHECKPOINT_PATH_FROM_SELECTION
```

Run evaluation and serving on an appropriate GPU allocation. These commands do not implement the live robot client.

## Final delivery

To produce the package automatically after all three training/evaluation stages finish, start one CPU watcher:

```bash
source examples/hanoi/scripts/env.sh
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hanoi.pipeline.finalize --exp-name hanoi_20260914 --watch
```

The watcher polls every ten minutes and records its state in `data/hanoi/runs/hanoi_20260914/finalizer.json`.
Its lock prevents duplicate watchers. It invokes the existing delivery audit only after all three models are
completed and every managed job is reconciled. Audit failures stop finalization for diagnosis; an existing delivery
is preserved for inspection. `ready_for_review` means the package was produced; inspect its evidence before declaring
the overall goal complete. The watcher imports the training stack only in the final packaging subprocess.

When no watcher is running, the same audit can be invoked manually after all three stages finish:

```bash
source examples/hanoi/scripts/env.sh
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hanoi.pipeline.deliver --exp-name hanoi_20260914
```

The audit reads the actual saved optimizer-step scalar, verifies every required validation/test/serving result,
recomputes validation-only selection, checks normalization and code identities, and hashes all selected checkpoint
files. It creates `data/hanoi/runs/hanoi_20260914/delivery/` atomically, with a metrics report, copied evidence,
reproducible source archive, checkpoint checksums, and download/serving commands. Weights remain in their selected
export directories. An incomplete model or failed serving check prevents delivery.

Validation in a CPU allocation (worker IPC and asynchronous checkpoint saves require local socket permissions):

```bash
source examples/hanoi/scripts/env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest src/openpi/policies/hanoi_policy_test.py \
  src/openpi/training/data_loader_test.py src/openpi/training/checkpoints_test.py \
  examples/hanoi/tests -k 'not with_real_dataset' -q
```

Training is not complete until all three selected EMA checkpoints, validation/test metrics, manifests, normalization
assets, and reproducible serving commands have been delivered. Offline error metrics will not establish hardware success.
