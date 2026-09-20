# Hanoi dense contract-five comparison run: 16-step chunk (`hanoi_dense_h16_20260920`)

Written 2026-09-20. Identical to the v5 run (`hanoi_dense_20260919`, 30-step chunk) except `action_horizon=16`
(1.6 s at 10 Hz, the Cosmos horizon) and exports every 4,000 updates. Same recording, observations, split,
normalization rule, model, optimiser, budget, validation subsets and test rows. Training job 18060964
(10 h 40 min on one H200), evaluate job 18079376. Offline only; **live success has not been measured.**

## Comparison on the identical test rows (episodes 45–49, every 3rd fresh row, 11,604 observations)

| metric | v5, H=30 | H=16 |
|---|---|---|
| slot-1 XYZ error mean / p95 | 0.592 / 1.432 mm | 0.567 / 1.419 mm |
| slot-1 stationary / moving | 0.456 / 0.619 mm | 0.456 / 0.589 mm |
| executed slots 1–3 | 0.59 / 0.64 / 0.78 mm | 0.57 / 0.64 / 0.78 mm |
| mean over the common slots 1–16 | 2.19 mm | 2.22 mm |
| chunk mean over its own horizon | 2.39 mm (30 slots) | 2.22 mm (16 slots) |
| jaw accuracy (all slots / slot 1) | 0.9990 / 0.9973 | 0.9998 / 0.9978 |
| gripper events detected | 4,344 / 4,345 | 2,319 / 2,319 |
| event timing exact, moving rows | 95.2% | 98.3% |
| event timing |error|, stationary rows | 1.49 rows | 0.11 rows |

Position is a tie: the per-slot curves coincide to a hundredth of a millimetre through slot 16. The shorter
chunk is better on the gripper: fewer wrong intents and tighter timing, especially from dwells, because 1.6 s
ahead is far less ambiguous than 3 s. Fewer events are "expected" for H=16 simply because a 1.6 s chunk
contains fewer future flips.

## Serving parity: served path exact, batched evaluator numerics over tolerance

The evaluate stage's parity check (200 validation observations, served path vs the batched evaluator) reported a
slot-1 difference of max 0.584 mm / mean 0.198 mm, above the guide's 0.5 mm tolerance (v5: 0.272 / 0.110 mm),
with identical jaw decisions. `complete.json` therefore records `serving_parity_passed: false` and
`ready_to_deploy_offline: false`, and the manager stopped on that check as designed.

`examples/hanoi/dense_parity_probe.py` (job 18080419, `parity_probe.json` in each run directory) separated the
two effects on 40 of those observations for both runs:

| | v5, H=30 | H=16 |
|---|---|---|
| evaluator at batch 1 vs served path, slot 1 and whole chunk | 0.000 mm | 0.000 mm |
| evaluator at batch 32 vs batch 1, slot 1 (max / mean) | 0.262 / 0.101 mm | 0.438 / 0.200 mm |
| jaw decisions equal | all | all |

The served path reproduces the single-observation evaluator exactly for both runs. The whole reported gap is
bfloat16 batch-shape numerics inside the batched evaluator, amplified over ten denoising steps and about twice
as large for the 16-slot suffix. It is not a serving defect; the 0.5 mm tolerance was set assuming the two paths
share a batch shape. The H=16 export is served identically to how it was scored, one observation at a time.

## Artifacts

- Selected export `checkpoints/pi05_hanoi_dense_h16_aaaa_to_cccc/hanoi_dense_h16_20260920/exports/29999` (12.4 GB).
- Evaluation records under `data/hanoi/runs/hanoi_dense_h16_20260920/pi05_hanoi_dense_h16_aaaa_to_cccc/`.
- Transfer list `data/hanoi/deployment_debug/hanoi_dense_h16_20260920_transfer.txt`. The served contract carries
  `action_horizon: 16`; the deployment client must accept that value (the execution prefix stays 3).
- W&B: https://wandb.ai/cw5167-nyu/openpi/runs/8l9wi3op

## Results: `hanoi_dense_h16_20260920`

