# Hanoi pi0.5 training and job scheduling plan

Status observed at **2026-09-15 14:46 UTC**: AAAA→CCCC has completed all 30,000 updates, six-candidate validation,
held-out testing, selected-policy serving parity, and the native per-model audit. Its selected export is `29999`;
the audit passed at 10:19 UTC. Reverse training is running as job `17839751` on H100 node `gh015`, with checkpoint
8,500 committed and the 5,000-step EMA export retained. Multitask awaits reverse training and evaluation.
The manager, separate W&B logger, and finalizer were verified live on **cs673**. The current execution session is
on cs653; inspect the controllers on their recorded host rather than interpreting missing local PIDs as exits.
Current dated observations and process identities are recorded in
`data/hanoi/PROGRESS.md` on HPC and `data/hanoi/runs/hanoi_20260914/`.

Conversion, shared transforms, three configs, normalization, teacher replay, GPU launchers, durable orchestration,
inference exports, quota checks, and offline selection/evaluation are implemented. CPU preparation `17736379`
passed conversion, normalization, replay, and native-data verification for all 100 episodes / 720,100 frames.
H100 qualification `17758983` passed batch-32/horizon-63 training, optimizer resume, EMA evaluation/sampling, and
native serving. Optional two-A100 qualification `17760911` was deferred while pending. Initial production
`17769392` was canceled by Slurm after a host-transfer stall; the resumed job has repeatedly saved checkpoints
successfully with `NUMPY_MADVISE_HUGEPAGE=0`. The administrative cancellation cause remains unknown.

Reverse checkpoint-to-checkpoint timing over steps 125–2,000 was **1.806 seconds per update**, including saves:
approximately **15 hours per 30,000-step model**. At the current 8,500-step observation, roughly **11 training hours
remain for reverse**, followed by approximately **15 training hours for multitask**. Queue waits, restarts, and
evaluation add time. The forward model's complete evaluation allocation ran for about 70 minutes; this does not
establish the duration of reverse or multitask evaluation. The current reverse allocation ends at
2026-09-15 22:29:20 UTC and is expected to need a continuation under the existing bounded policy.
Keep the original, more conservative qualification measurements for the existing allocation/retry budgets.
See [implementation progress and commands](../examples/hanoi/training/README.md).
Updated 2026-09-15. Scheduling estimates below are dated observations, not reservations.

## Models and data contract

Train three independent full fine-tunes of `gs://openpi-assets/checkpoints/pi05_base/params` using the existing JAX
trainer and flow-matching objective:

| Configuration | Training episodes | Task prompt |
| --- | ---: | --- |
| `pi05_hanoi_aaaa_to_cccc` | 40 forward | Move all four rings from peg A to peg C following Tower of Hanoi rules. |
| `pi05_hanoi_cccc_to_aaaa` | 40 reverse | Move all four rings from peg C to peg A following Tower of Hanoi rules. |
| `pi05_hanoi_multitask` | 80, both directions | The corresponding prompt above, supplied by each episode. |

- Use `pi05=True`, discrete state input, model action dimension 32, horizon 63, and a 30 Hz reference timeline.
- Keep global batch 32, seed 42, 30,000 optimizer steps, EMA 0.99, and existing AdamW defaults. Warm up for 1,000 steps
  to `2.5e-5`, then decay to `2.5e-6`. Hardware fallback changes placement/sharding, not the learning configuration.
- Convert the two `hanoi_wm_roundtrip_20260910_195558_*.h5` files under `/scratch/cw5167/datasets/` to one local dataset
  using OpenPI's pinned LeRobot version, with lossless images and all original episode rows.
- Pair forward/reverse episode `k`. Pairs 0-39 are training, 40-44 validation, and 45-49 test.
- Model inputs are the existing 224x224 RGB `pixels`, measured `proprio[:7]` (XYZ, linear velocity XYZ, jaw stroke),
  and the task prompt. Exclude `proprio[7]`, the current target jaw command, and all privileged route/board metadata.
- Targets are `action_abs[t:t+63]`: absolute next-reference XYZ in base-frame metres plus binary jaw intent. There is
  no additional label shift. Apply XYZ-only deltas relative to the anchor's measured XYZ; leave jaw intent absolute.
  Serving applies the inverse transform and returns 63 four-dimensional absolute reference actions.
- Exclude observation anchors with command-minus-image-receipt age outside 0-50 ms. Keep their underlying action rows
  so chunk timing is unchanged. Use native within-episode terminal hold padding. Compute per-model normalization
  from that model's training anchors and package the statistics with its checkpoints.
- Preserve the deliberately added motion noise. The audit's 5.7 mm figure measures extra interpolation error from
  replacing the noisy reference with endpoint-only, rest-to-rest segments; it is not a reason to remove noisy labels.

The read-only audit passed all 100 episodes / 720,100 rows: finite numeric fields, legal routes, consistent labels,
and all 9,000 Cartesian endpoints within 1.2 mm of their references. Freshness filtering retains 564,714 training,
70,563 validation, and 70,615 test anchors across both directions. These counts must be reproduced by conversion.
An automated color-region check also finds the four rings on the expected peg, in size order, in all 200 start/final
images. An 18-frame manual spot check covers both directions and all splits. These image checks support label
consistency; they do not establish success throughout every demonstration or on the live robot.

## Implementation and deployment equivalence

Follow repository conventions: frozen transform/config dataclasses, `tyro` CLIs, existing logging and import style,
Ruff rules, and focused tests under `examples/hanoi/tests/`. Group Hanoi conversion, evaluation, deployment, and
launch utilities by purpose under `examples/hanoi/`; keep the Hanoi policy adapter and three configurations in the
existing policy and training registries. Add only an optional
`DataConfig.frame_indices_path` to select validated global indices through `Subset(full_native_LeRobot_dataset)`.
Resolve selection files during data loading, not policy construction, so serving needs no raw training dataset.

The robot contract is one Trossen arm and one external RealSense RGB stream over ROS 2:

- Default topic `/camera/camera/color/image_raw`, `rgb8`, latest-frame subscription with depth one. Decode row stride
  correctly, crop the 640x480 source as `rgb[90:450, 151:511]`, then resize to 224x224 with `cv2.INTER_AREA`.
  HDF5 images are already cropped and must not be cropped again.
- Map the real image to `base_0_rgb`; zero and mask both missing wrist cameras. Use the same canonical preprocessing,
  state ordering, task text, normalization, padding, and inverse action mapping during training and serving.
- Preserve the commissioned tool/grasp frame and fixed orientation. Jaw readback is driver stroke, not pad separation.
- Start with a nine-reference / 0.3-second execution prefix. The command builder must preserve trajectory velocity
  and acceleration continuity; endpoint-only segments are not equivalent to the recorded trajectories.
- Match gripper behavior: opening to 0.034 stroke over 1.0 s; force closing at -20 N over 1.2 s plus 0.2 s settling.
  Repeated intent must not restart the dwell. No Cartesian commands during jaw actuation. After a blocking dwell,
  discard the old chunk and replan from a fresh observation; account for inference latency and expired references.
- Use goal-board confirmation with settled/open arm state for episode completion; terminal hold alone is not a stop
  prediction. Reset queued actions on episode/task changes. Robot-driver integration and live trials remain deferred.

The user has an **RTX 5080** and an already working local OpenPI policy server for
another task. Reuse that server and environment; generic GPU setup and runtime
compatibility investigation are outside the current deployment work. Bring the
matching Hanoi configuration and shared transforms into the local hardware branch,
transfer the selected export together with its normalization assets, and connect
the single-arm ROS/Trossen client. Preserve the 63-reference horizon, ten sampling
steps, nine-reference / 0.3-second execution prefix, and input/output/gripper
contract. Hanoi-specific serving parity and execution timing still need checking
when integrating this model; a working server for another task does not constitute
that model-specific evidence. Keep the running HPC training environment frozen.

Before full training, require conversion/transform tests and hardware-free teacher-reference replay tests covering
all nine prefix offsets, motion/gripper boundaries, stale inputs, delayed inference, and terminal padding. Compare
the executor's intermediate references, derivatives, jaw events, and times against the recorded noisy trajectory;
target reference reconstruction error at most 0.1 mm and the recorded speed/acceleration/jerk limits. This is an
offline command-equivalence check, not a claim that hardware tracking will be exact. Keep intentional upstream
training-only image augmentation separate from canonical input-parity tests.

## Torch resource discovery and qualification

Use the live scheduler and the [current NYU submission guide](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/).
Normal submissions supply an account, `--gres=gpu:N`, and a feature constraint; let Torch assign partitions and QoS.
Do not copy the older hard-coded partition lists or force `--qos=normal`: the latter failed test-only validation.
The association table's `normal` entry does not identify a usable job QoS. Inspect the actual submitted job's QoS and
its live limits rather than relying on older scripts or documentation quota numbers.

Use `torch_pr_595_tandon_advanced`, whose description matches this project. Other visible accounts are
`torch_pr_518_general`, `torch_pr_519_general`, and `users`; automatic rerouting stays on the project account.

At 2026-09-14 04:13-04:14 UTC (00:13-00:14 EDT), read-only checks found:

| Candidate | Unallocated, unreserved GPUs on serviceable nodes | Test-only estimated wait for an 8-CPU, 128-GiB, 12-hour job |
| --- | ---: | --- |
| 1 H200 | 10 | About 26 hours |
| 1 H100 | 10 | About 3 hours 12 minutes |
| 2 H100, same node | Same pool | About 3 hours 12 minutes |
| 1 or 2 A100, same node | 5 | About 1 hour 31-32 minutes |
| 2 L40S, same node | 2, plus 4 on planned nodes | About 8 hours 23 minutes |
| RTX 6000 pool | 24 | Not qualified for this pinned JAX environment |

Another 35 unallocated H200s were on reserved nodes. Physical free counts do not establish account access or start
time. Reducing the H200 request to 30 minutes or 2 hours, or enabling the preemption comment, did not improve its
estimate in this snapshot. No GPU training jobs were queued for this user; the existing `cpu-light` job is unrelated.

### Qualification profiles

1. Candidate profiles: 1 H200 (`fsdp_devices=1`), 1 H100 (`1`), 2 H100 (`2`), and 2 A100 (`2`), always on one node.
   Single-A100 and paired-L40S profiles may be added only after the same qualification. RTX 6000 requires explicit
   compatibility qualification with the pinned CUDA/JAX stack and is excluded from automatic fallback initially.
2. Before qualification, use `my_slurm_accounts`, live node features/allocated TRES/reservations, and
   `sbatch --test-only` with the actual resource request. Query GPU VRAM with `nvidia-smi` inside the allocated smoke
   job; the spec sheet's node RAM column is not GPU VRAM. Treat heterogeneous device-memory variants separately.
3. Run one 60-minute qualification job at a time, starting with the candidate with the earliest credible estimate.
   Test real batch-32/horizon-63 training, 100 measured steps after compilation, peak memory, checkpoint save/reload,
   finite gradients, and inference. If a device fails memory/compatibility, mark that exact profile unsupported and
   try the next candidate; do not repeatedly submit an unchanged OOM job or silently switch to LoRA.
4. For two-device profiles, use existing same-node JAX FSDP and keep global batch 32. Qualify restore into that mesh;
   resuming on a different device count additionally requires a cross-mesh checkpoint-restore test.
5. Prepare dependencies, converted data, normalization, and downloaded weights before GPU allocation. CPU stages use
   CPU resources. Profile data loading during the smoke run to avoid low GPU utilization from setup or I/O stalls.

Current estimates justify testing A100/H100 alternatives before waiting for H200; they do not prove that those
alternatives fit the model or will finish training sooner.

## Submission, monitoring, and replacement policy

Scratch storage must also qualify before production. At 2026-09-14 04:47 UTC, `myquota` reported 4.50 TB used of
5.00 TB. Retaining every milestone's full optimizer state for three models may exceed the remaining space.
Measure checkpoint sizes in the pilot; retain inference-only EMA snapshots at selection milestones and the latest
full resumable state, with room for an overlapping asynchronous save, data/cache growth, and qualification artifacts.
Atomic hard-linked snapshot export is implemented and tested against real Orbax saving, pruning, and restoration.
After each model's validation/test selection, compact this pipeline's generated candidates. Require a
storage-budget check before production; never infer available user quota from the filesystem-wide `df` output.
The conservative peak forecast is four full states plus eight inference exports, less already occupied pipeline
storage, plus a 50 GiB reserve. Train/evaluate serially and measure actual checkpoint sizes during qualification.

The manager in `examples/hanoi/pipeline/manage.py` uses a durable run manifest and a lock scoped to this Hanoi pipeline.
It records per-stage Slurm states, ambiguous submission/requeue/replacement states, and model progress. Record config,
code/lockfile/data-contract identity, exact sbatch arguments, hardware profile, experiment/checkpoint directory,
job IDs, submission/eligible times, scheduler reasons, estimates, replacements, retries, and completed checkpoints.

### Submission

- Stage order: environment/weights preparation; CPU conversion and contract checks; per-model CPU normalization;
  hardware qualification; three training runs; per-model validation/selection/test. Submit a downstream stage only
  after its predecessor has completed and its artifacts validate. This avoids stale dependency IDs after replacement.
- Initially allow at most two GPUs total across this pipeline. The measured scratch constraint additionally requires
  one training/evaluation run at a time, enforced by both the manager and the launchers' shared lock. Apply stricter
  live QoS/account limits.
- For qualified profiles, rank estimated completion as estimated start plus measured startup, remaining-step time,
  and checkpoint overhead. Without throughput measurements, rank only qualification jobs by estimated start.
- Use 8 CPUs, 128 GiB host RAM, and 4 loader workers initially. Size production walltime from the pilot: startup plus
  1.25 times measured remaining training time, every periodic save at its measured latency, and two additional
  checkpoint-save durations. Round up to 15 minutes, minimum 1 hour and maximum 12 hours per allocation. Longer
  training resumes across allocations. Use the same periodic-save budget for completion ranking and allocation bounds.
- Create log directories before `sbatch`; use absolute paths and `sbatch --parsable`. Record the returned job ID.
  A transient/ambiguous submit response requires querying the unique submission token before retrying.
- Before submitting each new production model, refresh the storage forecast using its selected qualified profile.
  A capacity shortfall waits for another monitor poll without consuming a GPU allocation; malformed quota data or
  other validation failures require diagnosis. The launcher repeats validation when the allocated job starts.
- Never use `--overwrite` for managed training. Keep the same experiment directory across retries and use `--resume`.
  Verify manifest identity before restoring. If interrupted before the first complete checkpoint, restart from base
  in that same managed run; preserve any partial files for Orbax's own handling.

Example request shape, once the implementation and qualification exist:

```bash
sbatch --parsable \
  --account=torch_pr_595_tandon_advanced \
  --nodes=1 --ntasks=1 --gres=gpu:1 --constraint=h100 \
  --cpus-per-task=8 --mem=128G --time=12:00:00 \
  --chdir=/scratch/cw5167/workspace/openpi \
  --output=/scratch/cw5167/workspace/openpi/slurm/%x-%j.out \
  --error=/scratch/cw5167/workspace/openpi/slurm/%x-%j.err \
  examples/hanoi/scripts/train.sbatch --config-name pi05_hanoi_aaaa_to_cccc --exp-name hanoi_20260914 \
  --qualification-path data/hanoi/runs/hanoi_20260914/h100_1.json
```

`examples/hanoi/scripts/train.sbatch` requires completed preparation and qualification artifacts. The manager will calculate
walltime and select the qualified profile rather than always using this example's values.

### Long-wait policy

- Poll tracked jobs every 10 minutes using batched `squeue` queries; use `sacct` after a job leaves the queue.
  Record `squeue --start` and pending reason. An unknown estimated start is not an infinite wait.
- After 30 minutes of eligible pending time, reassess if the remaining estimated wait exceeds 2 hours. If no estimate
  exists, reassess after 2 actual hours pending. Shared `QOSGrpGRES` capacity waits count toward these limits;
  dependency/held time does not count as eligible wait.
- Once one profile is qualified, the second qualification is optional. Defer it after 2 hours of eligible waiting,
  or after 30 minutes when its remaining estimated wait exceeds 2 hours. Capacity reasons include the observed
  `QOSGrpGRES` limit; dependency/user-held jobs do not trigger this rule. Persist deferral intent before exact-job
  pending-only cancellation, recover interrupted cancellations, and withdraw the intent if the job starts.
  A confirmed deferral ends additional qualification attempts for this run and proceeds with qualified hardware.
  It does not label the deferred GPU profile unsupported.
- An operator may defer that optional pilot earlier after at least 1 hour of eligible capacity waiting when a fresh
  test-only estimate shows qualified production can start within 5 minutes and the production storage gate passes.
  Record the estimate and reason before using the same locked, pending-only cancellation and recovery path. This
  operator decision does not change the monitor's automatic 2-hour threshold or cancel a running qualification.
- Check partition QoS capacity alongside test-only estimates. On September 14 at 10:51 UTC, test-only predicted
  an immediate H100 start, but the actual production submission remained pending on `QOSGrpGRES`. At 10:57 UTC,
  the `h100_tandon` and `a100_tandon` partitions each used their full 60-GPU group limit. An immediate test-only
  estimate alone is therefore insufficient evidence that either partition can admit a job. Retain the real queue
  entry while waiting for capacity; scheduled releases are observations, not promised starts.
- Test only qualified alternatives, at most once per hour. Replace when the alternative's estimated completion is
  at least 1 hour earlier, or, for an unknown current estimate after 2 hours pending, when a qualified alternative has
  a credible estimated start within 1 hour. Prefer the current job on ties or missing evidence.
- A fresh run with neither a recorded placement nor a completed checkpoint may switch device count between
  qualified profiles. Once either exists, changing device count requires explicit cross-mesh restore qualification.
  Global batch, optimizer settings, and the training/serving contract stay fixed.
- Never cancel a running job merely because another GPU becomes available. Never submit an identical duplicate to
  try to gain priority. Cancel/resubmit loses accumulated eligibility age.
- If a documented pending-job feature update can broaden an equivalent one-GPU profile, apply it only when Torch
  routing is verified afterward. Otherwise use a replacement, since changing features/comments does not guarantee
  that Torch's submission-time routing is recalculated.
- For replacement, record intent under the manager lock; verify the exact job is owned by this run; use
  `scancel --ctld --state=PENDING JOBID`; confirm that job was canceled before submitting its replacement. If it has
  started, keep it. If state/cancellation is ambiguous, do not launch another writer. Never target jobs by user-wide
  cancellation or a loose name pattern.
- Allow one automatic queue replacement per model per 6 hours, at most two total before reporting persistent queue
  blockage. With no better qualified alternative, retain the pending job and reassess instead of churning the queue.

### Preemption, retries, and monitoring

- After a long-wait trigger, include normal-plus-preemptible placement only for profiles whose checkpoint/restart
  smoke test passed, using Torch's `--comment="preemption=yes;requeue=true"`. Re-test its estimate; enable it only
  when the improvement rule above is met. Do not automatically use preemption-only placement or GPU MPS.
- After the pilot, choose `save_interval` as the largest divisor of 5,000 no greater than both 1,000 and
  `floor(300 / p95_step_seconds)`, with a minimum of one step. This targets saves within approximately 5 minutes
  while actually producing the retained 5,000-step EMA snapshots; retain those plus the newest full checkpoint.
  Include measured save latency when checking checkpoint age. Verify the first checkpoint finishes well before
  preemption eligibility. An incomplete asynchronous save is not a resumable checkpoint.
- Slurm requeue reruns the batch script; the launcher must select the same run and actually restore its training
  state. The current trainer has periodic saves, not an emergency signal checkpoint. Do not assume a last-minute
  signal guarantees a fresh checkpoint. Hardware/profile switching preserves global batch and optimizer progress;
  it does not promise bit-identical data ordering after restart.
- Wait for scheduler auto-requeue when enabled; do not also submit a replacement. If the scheduler reports a terminal
  `PREEMPTED`, `NODE_FAIL`, or `TIMEOUT` without requeue, submit one continuation from the last completed checkpoint,
  subject to the following budgets. Count scheduler auto-requeues as attempts too, and stop owned requeues when a
  budget is exhausted.
- Separate healthy planned walltime continuations from failure retries. Bound planned allocations using the pilot's
  total runtime budget divided by usable training time per allocation, rounded up, plus one contingency allocation;
  record that bound before production. Cap unplanned infrastructure retries at three per model. Across both kinds,
  stop after two consecutive restarts fail to advance the last complete checkpoint. Report exhausted budgets before
  extending them.
- OOM, invalid data, NaN, import errors, and repeated low-utilization cancellation require diagnosis/profile correction;
  they are not blind-retry cases. Keep logs and report the cause.
- Monitor GPU utilization/memory, measured steps per second, loss/gradient finiteness, checkpoint age, and remaining
  walltime. Keep preparation on CPU and fix pipeline bottlenecks rather than generating artificial GPU load.
- Source `examples/hanoi/scripts/env.sh` before each stage; it sets `NUMPY_MADVISE_HUGEPAGE=0` before NumPy imports.
  On September 14, production checkpoint 3250 stalled in JAX host transfer with heavy kernel compaction. A bounded
  CPU probe on the same node/allocation filled and checked a 2 GiB array in 1.04 seconds with huge-page advice off;
  the default advice exceeded 20 seconds. The pinned XLA transfer implementation allocates its buffer through
  NumPy. This supports a scoped allocation repair; it does not establish the administrative cause of the later
  `CANCELLED by 0` event. [NumPy documents this import-time setting](https://numpy.org/doc/1.26/reference/global_state.html).
  Preserve the original qualification evidence, record this environment difference, and verify a full restored
  training step and subsequent committed checkpoint before declaring the repair effective. Full-model settings,
  dependencies, optimizer/EMA state, normalization, and the data/serving contract remain identical.
- `telemetry.jsonl` records the GPUs associated with the training PID, not every GPU reported on the node. NVML process
  UUIDs avoid confusing Slurm's remapped CUDA device indices. The manager records scheduler runtime/walltime details.
- Run the manager explicitly in an appropriate CPU session/allocation. A written policy alone is not a running
  monitor. All submission/cancellation operations are limited to job IDs recorded by this pipeline.
- Current monitor details and its hosting allocation expiration are in `monitor_process.json`. A detached process
  launched with host permission is invisible inside the tool's isolated PID namespace; verify it on the host before
  deciding it has exited. A redundant queued controller (`17743243`) was canceled after that host check.
- User-requested W&B tracking applies to the still-unsubmitted reverse and multitask models. Keep the independent
  `examples.hanoi.pipeline.wandb_logger` CPU process alive alongside the manager. Its stable per-model IDs and destination
  `cw5167-nyu/openpi` are recorded in `wandb_logging.json`; its heartbeat/PID/URLs are in `wandb_logger.json`.
  It publishes actual stdout metrics and telemetry, while native trainer logging flags and the qualified training
  identity stay unchanged. Use recorded optimizer steps as the plot axis and recover missing events from local
  logs after a logger restart. The already-started forward model keeps local logging. See the Hanoi README for
  the launch command and scope; final evaluation metrics and selected weights remain the delivery pipeline's job.

## Validation and completion

Test manager behavior with a fake scheduler: unknown estimates, quota/held reasons, pending-to-running cancellation
races, ambiguous submission responses, manager restarts, duplicate prevention, preemption requeue versus terminal
failure, incomplete checkpoints, retry caps, and resource-budget enforcement. Use real `sbatch --test-only` for
request validation; it produces estimates but does not create queued jobs.

After each model finishes, evaluate retained complete EMA checkpoints using reproducible validation flow loss.
Discover checkpoint directories rather than assuming the final label is `30000` (the current loop usually saves
`29999`). Include the final partial evaluation batch. On up to 64 evenly spaced eligible anchors per episode, report
first/mean/last-valid XYZ reference error, jaw balanced accuracy/confusion, and direction/episode breakdowns; mask
extrapolated terminal positions. Add a multitask prompt-swap diagnostic. Select using validation only, then evaluate
the selected checkpoint once on test. Deliver all three checkpoints, normalization assets, manifests, metrics, and
reproducible commands. Hardware task success remains a separate live evaluation.

Before marking a selected model complete, compare its native `create_trained_policy` output with the evaluation
path on eligible training observations, with identical explicit noise and ten sampling steps. Require finite
63-by-4 absolute actions, numerical agreement, and identical binary jaw decisions. Test both prompts for multitask.

`examples/hanoi/pipeline/deliver.py` performs the final three-model audit. It restores only the saved optimizer-step scalar
to confirm 30,000 completed updates, verifies the six validation results and selected test/serving evidence, recomputes
selection, and hashes every selected checkpoint file. It emits an atomic delivery directory with source archive,
normalization/checkpoint paths, metrics, copied manifests, checksum files, and portable download/serving commands.
The source archive excludes recordings and checkpoints. A package-import check validates the archived runtime files.

`examples/hanoi/pipeline/finalize.py --watch` runs in the CPU allocation and invokes that audit once the manager has completed
all three models and reconciled every job. It holds a separate watcher lock and records a heartbeat in `finalizer.json`.
It preserves an existing delivery and stops on audit failure. After successful packaging, its `ready_for_review` state
still requires inspection of the generated evidence before declaring the goal complete. This watcher does not change
the qualified training implementation or scheduling/retry budgets.

## References

- [NYU Torch job submission and preemption](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)
- [Slurm sbatch and test-only mode](https://slurm.schedmd.com/sbatch.html)
- [Slurm pending start estimates](https://slurm.schedmd.com/squeue.html)
- [Slurm cancellation filters](https://slurm.schedmd.com/scancel.html)
- [Slurm priority and eligibility age](https://slurm.schedmd.com/priority_multifactor.html)
- [Trossen OpenPI integration](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/openpi.html)
- [Trossen interpolation and units](https://docs.trossenrobotics.com/trossen_arm/main/programming_guide/concepts.html)
