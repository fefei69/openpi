# Hanoi dense contract-five handover: pi0.5 pipeline

Written 2026-09-19 by the openpi-side agent. Scope: the pi0.5 half of
[`hanoi_dense_training_guide.md`](hanoi_dense_training_guide.md). The Cosmos half is handled separately in
`cosmos-policy`; nothing here touches that repository. Results sections are filled in after the evaluate stage.
Nothing in this document claims live success; that is measured on the arm.

## 1. What changed against waypoint_v4

| | waypoint_v4 | dense_v5 (this run) |
|---|---|---|
| Observations | 3,589 curated decision windows | every fresh 30 Hz row: 345,280 (train 275,847) |
| Label | 8 recorded leg endpoints, relative to measured XYZ | 30 commanded reference poses at 10 Hz, absolute XYZ + jaw intent |
| State | 6 joints + jaw stroke, discrete prompt tokens | same values, same tokens (standard pi0.5 input; see §3) |
| Cartesian context | required for encoding/decoding | none |
| Frames | copied into a LeRobot dataset | indexed from the raw HDF5 by row (no copies) |
| Execution at deployment | first target only | 3-step prefix (0.3 s) of a 30-step chunk, asynchronous |

## 2. Dataset: `data/hanoi/dense_v5_pi05/`

Built by `examples/hanoi/dense_dataset.py` from
`/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260915_223424_AAAA_to_CCCC.h5` (sha256 `d235dbc8…`).
Every row whose image is neither stale nor repeated is an observation; its label is the chunk
`reference_pose[t + 3j, 0:3]`, `action_abs[t + 3j, 3]` for `j = 1..30`; rows past the episode end repeat the
last row and are flagged in `actions_is_pad`. No row was dropped, curated, re-weighted or relabelled.

- `indices/aaaa_to_cccc_{train,val,test}.npz`: `source_observation_indices`, `states` (n, 7), `actions`
  (n, 30, 4), `source_action_indices`, `actions_is_pad`, `episode_indices`, `source_episode_bounds`,
  `measured_xyz` (audits only), `speed_m_s`, `stationary` (finite-difference speed < 2 mm/s),
  `observation_jaw_intent`, plus `validated`, `horizon`, `frameskip`, `split`, `source_path`, `prompt`.
- Split (decision 9, unchanged from v4): episodes 0–39 / 40–44 / 45–49.
- `audit.json` / `audit.md`: the section-3 audit. `action_abs[t,:3]` equals `reference_pose[t,:3]` exactly on the
  same row (attribute `post_action_reference`); the next-row residual is one tick of commanded motion (median
  0.73 mm, max 4.3 mm). 1,500 jaw flips on exactly the 1,500 command rows. Stationary rows 17.15% (episode 40:
  17.34%). Stale 7,852 ⊂ repeated 14,770; 345,280 observations remain; 141 tail rows have a padded slot 1.
- `dataset_statistics.json`: min/max/mean/std of state and of actions over valid slots (training split).
- `assets/pi05_hanoi_dense_aaaa_to_cccc/local/hanoi_dense_v5/norm_stats.json`: the openpi normalizer file.
  **Deviation:** its `q01`/`q99` are the training-split min/max, not percentiles. The board sits at one x, so
  the 1st–99th percentile of x spans under 5 mm while the home pose (first ~2 s of each episode, 0.4% of slots)
  lies 8 cm away and would normalize to about −33 under the stock quantile bounds. Mean/std are unchanged.
- `verification.json`: every archive re-derived from the raw file matched exactly; frames and the real loader
  path were exercised (`dense_prepare.sbatch` on a 48 GB CPU job; the 4 GB cpu-light allocation OOMs).
- `dryrun.json`: 4 loader workers deliver ~30 batches/s of batch 32 on CPU.

## 3. Decision 5 (state input)

In openpi, pi0.5 feeds the state as discretized prompt tokens (`Task: …, State: 178 176 236 9 176 176 255;`);
`discrete_state_input=False` removes the state from the prompt and nothing else consumes it (the stock
`pi05_libero` recipe is state-free). There is no continuous state path for pi0.5. On 2026-09-19 the user chose
to keep the standard pi0.5 tokens (option 1). The alternatives considered were adding π0's continuous state
projection to π0.5 (a model edit) or training state-free.

## 4. Code

| File | Role |
|---|---|
| `src/openpi/policies/hanoi_dense_policy.py` | `HORIZON=30`, contract v5, `HanoiDenseInputs`/`HanoiDenseOutputs` (jaw thresholded at 0.5, `reference_rate_hz`, `execution_prefix` in the reply), `serving_metadata()` |
| `src/openpi/training/config.py` | `LeRobotHanoiDenseDataConfig`, `DataConfig.dense_archive_path`, `TrainConfig.policy_metadata_module`, config `pi05_hanoi_dense_aaaa_to_cccc` |
| `src/openpi/training/data_loader.py` | `RawFrameChunkDataset`: validates the archive, reads `pixels[row]` lazily per worker process |
| `src/openpi/policies/policy_config.py` | per-checkpoint metadata hook (`hanoi_dense` block with export/normalization hashes) |
| `examples/hanoi/dense_dataset.py` | build, audit, verify, `require_validated_data()` gate |
| `examples/hanoi/dense_validation.py` | section-7 metrics (`physical_report`), subset sampling, in-training `Validation` hook, decision-11 `selection_key` |
| `examples/hanoi/dense_training.py` + `dense_train.sbatch` | authorized run guard (30,000 updates, batch 32, H=30, W&B), storage gate, identity file `hanoi_dense_identity.json`, learning curve |
| `examples/hanoi/dense_evaluate.py` + `dense_evaluate.sbatch` | per-export validation, selection, test, serving parity, `complete.json`, transfer list |
| `examples/hanoi/dense_manage.py` | Slurm continuation: train (1440 min) → evaluate (300 min), ≤3 restarts on TIMEOUT/PREEMPTED/NODE_FAIL |
| `examples/hanoi/dense_dryrun.py` + `dense_prepare.sbatch` | CPU verification and loader rehearsal |

`examples/hanoi/dataset.py::training_code_identity()` now also hashes the dense modules; together with the
`src/openpi` edits this means the finished v4 evaluate cannot be re-run unchanged (its results are written).

## 5. Training

- Config `pi05_hanoi_dense_aaaa_to_cccc`: `Pi0Config(pi05=True, action_horizon=30, discrete_state_input=True)`,
  weights `gs://openpi-assets/checkpoints/pi05_base/params`, cosine schedule (warmup 1,000, peak 2.5e-5, decay
  to 2.5e-6 over 30,000), batch 32, 30,000 updates, EMA on, save/export every 2,000 plus 29,999, 4 loader workers.
- Run `hanoi_dense_20260919`, one H200 (`h200_1`), 24 h walltime. Manager PID 3406486 on cs616 (cpu-light
  allocation 17900893); log `.cache/hanoi/logs/hanoi_dense_20260919-manager.log`; state
  `data/hanoi/runs/hanoi_dense_20260919/dense_manager.json`. Train job 18019092 submitted 10:46 EDT.
- In-training learning curve: every 9th validation row (3,848 observations, 670 stationary), EMA weights,
  10 sampling steps, at steps 0/250/500/1000/2000/4000 and every export; seeds 43 and 44 at 29,999. Reports in
  `data/hanoi/runs/hanoi_dense_20260919/pi05_hanoi_dense_aaaa_to_cccc/learning_curve/`.
- Checkpoints/exports: `checkpoints/pi05_hanoi_dense_aaaa_to_cccc/hanoi_dense_20260919/exports/<step>/`
  with `export.json`, `params/`, `assets/local/hanoi_dense_v5/norm_stats.json`.

## 6. Evaluation protocol (section 7 of the guide)

`dense_evaluate` scores every export on every 3rd validation row (~11.5k observations, phase randomized), with
`physical_report` giving, for all / stationary / moving observations: per-step XYZ error over valid slots
(mean, p95, per-slot means, slot 1 separately), chunk endpoint error, jaw accuracy per slot and balanced, and
event timing (first intent flip relative to the observation's intent: detected / missed / false alarms, timing
error in rows). Selection (decision 11): lowest validation mean per-step XYZ error among exports with jaw
accuracy ≥ 0.99, ties to the earlier step; `selection.json` is written before the test archive is opened. The
selected export is then scored once on every 3rd test row (`test.json`) and checked for serving parity: 200
validation observations through `policy_config.create_trained_policy` (the `serve_policy.py` factory) must
reproduce the evaluator's slot 1 within 0.5 mm with equal jaw decisions (`serving_validation.json`).
"Ready to deploy (offline)" = slot-1 mean under 2 mm on both motion subsets, jaw accuracy ≥ 0.99, parity passed.

## 7. Serving

```bash
python scripts/serve_policy.py --port 8000 \
  --default-prompt 'Move all four rings from peg A to peg C following Tower of Hanoi rules.' \
  policy:checkpoint \
  --policy.config pi05_hanoi_dense_aaaa_to_cccc \
  --policy.dir checkpoints/pi05_hanoi_dense_aaaa_to_cccc/hanoi_dense_20260919/exports/<selected>
```

Request: `observation/image` (224×224×3 uint8, the deployment crop), `observation/state` (7,), `prompt`.
Reply: `actions` (30, 4) float32 absolute base-frame XYZ metres + jaw intent 0/1, `reference_rate_hz` 10,
`execution_prefix` 3. Metadata: the contract dict plus `hanoi_dense` {contract, prompt, config_name, checkpoint,
export_sha256, export_step, normalization_sha256, num_steps 10}.

## 8. Results (offline; live success not measured)

Full tables: [`hanoi_dense_pi05_results_20260919.md`](hanoi_dense_pi05_results_20260919.md).

- Selected export **29999** (decision 11; every export from 2,000 on was eligible, curve monotone).
- Test split (episodes 45–49, every 3rd row, 11,604 observations): slot-1 0.59 mm mean / 1.43 mm p95, 98.3%
  within 2 mm; stationary 0.46 mm, moving 0.62 mm; chunk mean 2.39 mm; endpoint 2.60 mm; jaw 99.90%;
  4,344 of 4,345 gripper events detected, mean absolute timing error 0.22 rows.
- Executed prefix (slots 1–3): 0.59 / 0.64 / 0.78 mm. Far-slot error concentrates in dwell-exit timing from
  stationary observations (stationary chunk mean 5.7 mm, p95 29.9 mm; moving 1.7 mm, 5.9 mm).
- Serving parity: passed, 200 observations, slot-1 max 0.27 mm (tolerance 0.5 mm), jaw decisions identical.
- W&B: https://wandb.ai/cw5167-nyu/openpi/runs/jg44v2cf
- Transfer list `data/hanoi/deployment_debug/hanoi_dense_20260919_transfer.txt` (12.4 GB; export, evaluation
  records, dataset metadata, serving code, this handover and the results note). Run on the deployment PC inside
  its openpi clone, after committing local edits (five tracked files are overwritten):

```bash
rsync -avhPr --append-verify \
  --files-from=:/scratch/cw5167/workspace/openpi/data/hanoi/deployment_debug/hanoi_dense_20260919_transfer.txt \
  cw5167@login.torch.hpc.nyu.edu:/scratch/cw5167/workspace/openpi/ \
  ./
```

## 8b. Comparison run: 16-step chunk (2026-09-20)

At the user's request a second run, identical except `action_horizon=16`, was trained and evaluated
(`hanoi_dense_h16_20260920`; see [`hanoi_dense_pi05_h16_results_20260920.md`](hanoi_dense_pi05_h16_results_20260920.md)).
On the identical test rows position is a tie (slot 1 0.567 vs 0.592 mm; slots 1–3 identical; per-slot curves
coincide through slot 16) while the gripper is better with the shorter chunk (jaw 99.98% vs 99.90%; event timing
exact on 98.3% vs 95.2% of moving rows; 0.11 vs 1.49 rows error from dwells). Its parity check exceeded the
0.5 mm tolerance (0.584 mm max) but a batch-1 probe shows the served path is exact for both runs and the gap is
batched-evaluator numerics. Both exports are transfer-ready; v5 is the guide's specified horizon, H=16 requires
the client to accept `action_horizon: 16`.

## 9. Deviations from the guide

1. Data root `data/hanoi/dense_v5_pi05/` instead of `data/hanoi/dense_v5/` (separate from the Cosmos build).
2. Normalization bounds are the training min/max (see §2).
3. Only the pi0.5 archive (H=30) was produced; the builder takes `--horizon 16` if a Cosmos archive is wanted.
4. In-training validation uses every 9th row; the evaluate stage uses every 3rd row (allowed by section 7).
5. Decision 5 implemented as the standard pi0.5 discrete state tokens (see §3), by the user's choice.
6. Image augmentation: decision 10 says "none", and none was added, but openpi's training-time preprocessing
   (`src/openpi/models/model.py::preprocess_observation`, unchanged in this fork) applies its library default to
   the base camera when training: random crop to 95% then resize, rotation within ±5°, and colour jitter
   (brightness 0.3, contrast 0.4, saturation 0.5). The v4 run trained under the same default. Serving and
   evaluation apply no augmentation.
