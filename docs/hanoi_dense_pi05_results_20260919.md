# Hanoi dense contract-five results: pi0.5 run `hanoi_dense_20260919`

Written 2026-09-20 by the openpi-side agent. Training ran 2026-09-19 10:55–21:47 EDT on one H200 (job
18019092, 30,000 updates, 10 h 53 min); the evaluate stage ran 21:48–22:31 EDT (job 18046917). Everything
below is offline. **Live success has not been measured.**

## Summary

- **Selected export: 29999** (the final one), by decision 11: lowest validation mean per-step XYZ error among
  exports with jaw accuracy ≥ 0.99, ties to the earlier step. Every export from 2,000 on was eligible and the
  curve never turned upward, so the last export won.
- **Test split** (episodes 45–49, every 3rd fresh row, 11,604 observations, selected export, EMA, 10 sampling
  steps): slot-1 XYZ error 0.59 mm mean / 1.43 mm p95, 98.3% within 2 mm; stationary observations 0.46 mm,
  moving 0.62 mm; chunk mean over the 30 slots 2.39 mm; chunk endpoint (3 s ahead) 2.60 mm; jaw accuracy
  99.90%; gripper events 4,344 of 4,345 detected, 93% at the exact slot, mean absolute timing error 0.22 rows.
- **Serving parity passed**: 200 validation observations through the `serve_policy.py` factory reproduced the
  evaluator's slot 1 within 0.27 mm (tolerance 0.5 mm), whole chunk within 0.39 mm, jaw decisions identical.
- **Offline ready-to-deploy criteria met** (slot-1 mean under 2 mm on both motion subsets, jaw ≥ 0.99,
  parity passed, hashes recorded in `complete.json`).

## What the arm executes

The client runs the first three slots (0.3 s) and re-plans. Test-split per-slot mean XYZ error, mm:

| slot | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| error mm | 0.59 | 0.64 | 0.78 | 1.20 | 2.67 | 3.68 | 3.11 | 2.03 | 2.63 |

## Where the remaining error sits

Far slots from **stationary** observations. When the observation is a dwell (gripper closing/opening, 17% of
rows), the policy cannot know exactly when the dwell ends, so slots that fall on the next leg's departure are
time-shifted: stationary chunk mean 5.7 mm and p95 29.9 mm, versus 1.7 mm and 5.9 mm from moving observations.
The same shows in event timing: from stationary rows 57% of predicted flips land on the exact slot (mean
absolute error 1.5 rows, p95 6 rows); from moving rows 95% are exact. Slot 1 is unaffected (0.46 mm from
stationary rows), so the executed prefix does not carry this uncertainty. Position error on the executed
slots is therefore sub-millimetre in both regimes; what remains is timing of dwell exits several slots ahead.

## Against waypoint_v4 (for orientation, not a like-for-like metric)

v4 reported 0.46 mm first-target error on 18 fixed destination points; v5 reports 0.59 mm on the commanded
pose 0.1 s ahead of every fresh row, with no curated windows and no measured-XYZ context. v4's live failures
(phase flips on state swaps, null moves) were not reproducible offline and are not claimed fixed here.

## Deviations from the guide

1. Data root `data/hanoi/dense_v5_pi05/` (own folder, separate from the Cosmos build).
2. Normalization bounds are the training-split min/max instead of the 1st/99th percentiles (the board sits at
   one x, so the percentile range of x is under 5 mm while the home pose lies 8 cm away; see the handover §2).
3. Only the pi0.5 archive (H=30) was built; `dense_dataset.py --horizon 16` produces the Cosmos one.
4. In-training validation used every 9th validation row; the evaluate stage used every 3rd (allowed by §7).
5. Decision 5 implemented as the standard pi0.5 discrete state tokens, by the user's choice (option 1).
6. Serving parity used the guide's 200-observation / 0.5 mm protocol rather than v4's exact single-probe check.
7. Image augmentation: decision 10 says "none", and none was added, but openpi's training-time preprocessing
   (`src/openpi/models/model.py::preprocess_observation`, unchanged in this fork) applies its library default to
   the base camera when training: random crop to 95% then resize, rotation within ±5°, and colour jitter
   (brightness 0.3, contrast 0.4, saturation 0.5). The v4 run trained under the same default. Serving and
   evaluation apply no augmentation.

## Artifacts

- Selected export: `checkpoints/pi05_hanoi_dense_aaaa_to_cccc/hanoi_dense_20260919/exports/29999` (12.4 GB;
  `export.json` sha256 `2084b7375cb8b445…`, `assets/local/hanoi_dense_v5/norm_stats.json` sha256 `ef8c5fb9e4be9a4a…`).
- Evaluation: `data/hanoi/runs/hanoi_dense_20260919/pi05_hanoi_dense_aaaa_to_cccc/{validation_<step>.json,
  selection.json, test.json, serving_validation.json, complete.json, learning_curve/}` with prediction arrays as `.npz`.
- Transfer list: `data/hanoi/deployment_debug/hanoi_dense_20260919_transfer.txt`.
- W&B: https://wandb.ai/cw5167-nyu/openpi/runs/jg44v2cf
- Handover: `docs/hanoi_dense_pi05_handoff.md`.

## Results: `hanoi_dense_20260919`

### In-training learning curve (every 9th validation row, EMA, seed 43)

| step | slot 1 all mean / p95 mm | stationary | moving | chunk mean mm | chunk p95 mm | endpoint mm | jaw acc | events detected (timing) | ready |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 85.760 / 140.369 | 87.013 / 145.271 | 85.496 / 138.010 | 88.017 | 142.693 | 89.563 | 0.3927 | 695/1444 (749 missed, 1705 false), 48.30 rows | no |
| 250 | 16.850 / 40.810 | 14.928 / 31.132 | 17.254 / 41.975 | 27.346 | 81.002 | 38.240 | 0.9346 | 1147/1444 (297 missed, 214 false), 10.60 rows | no |
| 500 | 7.583 / 18.741 | 5.793 / 12.988 | 7.960 / 19.651 | 11.825 | 37.674 | 15.384 | 0.9766 | 1371/1444 (73 missed, 66 false), 5.21 rows | no |
| 1000 | 4.303 / 10.312 | 3.342 / 7.503 | 4.506 / 10.926 | 7.447 | 25.507 | 8.777 | 0.9851 | 1408/1444 (36 missed, 44 false), 3.38 rows | no |
| 2000 | 2.861 / 7.023 | 2.217 / 4.782 | 2.997 / 7.419 | 5.147 | 18.514 | 5.856 | 0.9928 | 1429/1444 (15 missed, 9 false), 1.68 rows | no |
| 4000 | 2.075 / 5.058 | 1.799 / 4.353 | 2.134 / 5.208 | 4.065 | 15.224 | 4.623 | 0.9959 | 1442/1444 (2 missed, 11 false), 0.96 rows | no |
| 6000 | 1.716 / 4.141 | 1.332 / 2.902 | 1.797 / 4.349 | 3.699 | 14.646 | 4.161 | 0.9972 | 1442/1444 (2 missed, 10 false), 0.64 rows | yes |
| 8000 | 1.533 / 3.876 | 1.171 / 2.315 | 1.609 / 4.093 | 3.493 | 14.198 | 3.893 | 0.9978 | 1443/1444 (1 missed, 5 false), 0.50 rows | yes |
| 10000 | 1.392 / 3.534 | 1.045 / 2.218 | 1.465 / 3.717 | 3.294 | 13.987 | 3.653 | 0.9983 | 1443/1444 (1 missed, 5 false), 0.38 rows | yes |
| 12000 | 1.176 / 3.028 | 0.922 / 1.876 | 1.230 / 3.168 | 3.146 | 13.473 | 3.489 | 0.9985 | 1442/1444 (2 missed, 1 false), 0.35 rows | yes |
| 14000 | 1.067 / 2.684 | 0.844 / 1.796 | 1.113 / 2.780 | 3.082 | 13.837 | 3.438 | 0.9986 | 1442/1444 (2 missed, 3 false), 0.32 rows | yes |
| 16000 | 1.000 / 2.461 | 0.762 / 1.688 | 1.051 / 2.620 | 2.976 | 14.052 | 3.334 | 0.9986 | 1443/1444 (1 missed, 3 false), 0.31 rows | yes |
| 18000 | 0.902 / 2.292 | 0.686 / 1.462 | 0.947 / 2.412 | 2.900 | 13.671 | 3.212 | 0.9987 | 1442/1444 (2 missed, 2 false), 0.31 rows | yes |
| 20000 | 0.814 / 2.066 | 0.613 / 1.365 | 0.856 / 2.153 | 2.785 | 13.685 | 3.112 | 0.9988 | 1443/1444 (1 missed, 3 false), 0.27 rows | yes |
| 22000 | 0.739 / 1.866 | 0.573 / 1.372 | 0.774 / 1.956 | 2.738 | 13.596 | 3.094 | 0.9989 | 1443/1444 (1 missed, 2 false), 0.25 rows | yes |
| 24000 | 0.686 / 1.765 | 0.518 / 1.320 | 0.721 / 1.876 | 2.703 | 14.104 | 3.005 | 0.9989 | 1443/1444 (1 missed, 1 false), 0.27 rows | yes |
| 26000 | 0.635 / 1.646 | 0.499 / 1.279 | 0.663 / 1.689 | 2.636 | 13.676 | 2.958 | 0.9990 | 1443/1444 (1 missed, 1 false), 0.25 rows | yes |
| 28000 | 0.623 / 1.638 | 0.460 / 1.190 | 0.658 / 1.720 | 2.608 | 13.696 | 2.923 | 0.9989 | 1443/1444 (1 missed, 1 false), 0.26 rows | yes |
| 29999 | 0.596 / 1.463 | 0.452 / 1.243 | 0.626 / 1.529 | 2.555 | 13.504 | 2.802 | 0.9989 | 1443/1444 (1 missed, 1 false), 0.27 rows | yes |

### Evaluate stage: every export on every 3rd validation row

| step | slot 1 all mean / p95 mm | stationary | moving | chunk mean mm | chunk p95 mm | endpoint mm | jaw acc | events detected (timing) | ready |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 2.854 / 7.112 | 2.209 / 4.852 | 2.989 / 7.492 | 5.190 | 18.554 | 5.955 | 0.9926 | 4261/4326 (65 missed, 28 false), 1.71 rows | no |
| 4000 | 2.070 / 4.960 | 1.715 / 3.802 | 2.144 / 5.164 | 4.053 | 15.070 | 4.549 | 0.9959 | 4315/4326 (11 missed, 34 false), 0.93 rows | no |
| 6000 | 1.699 / 4.098 | 1.312 / 2.948 | 1.780 / 4.286 | 3.652 | 14.234 | 4.049 | 0.9972 | 4320/4326 (6 missed, 23 false), 0.65 rows | yes |
| 8000 | 1.548 / 3.878 | 1.154 / 2.460 | 1.631 / 4.108 | 3.443 | 13.784 | 3.824 | 0.9978 | 4321/4326 (5 missed, 15 false), 0.52 rows | yes |
| 10000 | 1.397 / 3.518 | 1.025 / 2.240 | 1.475 / 3.755 | 3.221 | 13.409 | 3.521 | 0.9982 | 4323/4326 (3 missed, 17 false), 0.42 rows | yes |
| 12000 | 1.186 / 3.021 | 0.886 / 2.001 | 1.249 / 3.189 | 3.102 | 13.264 | 3.369 | 0.9984 | 4316/4326 (10 missed, 11 false), 0.36 rows | yes |
| 14000 | 1.066 / 2.561 | 0.828 / 1.845 | 1.116 / 2.693 | 3.013 | 13.483 | 3.357 | 0.9986 | 4320/4326 (6 missed, 8 false), 0.34 rows | yes |
| 16000 | 1.012 / 2.504 | 0.757 / 1.680 | 1.065 / 2.640 | 2.900 | 13.067 | 3.219 | 0.9986 | 4322/4326 (4 missed, 8 false), 0.33 rows | yes |
| 18000 | 0.896 / 2.239 | 0.685 / 1.572 | 0.940 / 2.340 | 2.840 | 13.076 | 3.130 | 0.9987 | 4320/4326 (6 missed, 7 false), 0.31 rows | yes |
| 20000 | 0.827 / 2.101 | 0.634 / 1.498 | 0.867 / 2.210 | 2.729 | 12.956 | 3.019 | 0.9987 | 4321/4326 (5 missed, 6 false), 0.31 rows | yes |
| 22000 | 0.743 / 1.879 | 0.580 / 1.375 | 0.777 / 1.933 | 2.687 | 13.177 | 2.977 | 0.9988 | 4323/4326 (3 missed, 9 false), 0.27 rows | yes |
| 24000 | 0.687 / 1.738 | 0.523 / 1.329 | 0.721 / 1.788 | 2.624 | 13.104 | 2.887 | 0.9988 | 4322/4326 (4 missed, 5 false), 0.29 rows | yes |
| 26000 | 0.632 / 1.619 | 0.497 / 1.294 | 0.661 / 1.674 | 2.566 | 12.932 | 2.831 | 0.9988 | 4322/4326 (4 missed, 9 false), 0.27 rows | yes |
| 28000 | 0.622 / 1.610 | 0.480 / 1.277 | 0.652 / 1.664 | 2.538 | 13.030 | 2.807 | 0.9989 | 4322/4326 (4 missed, 4 false), 0.26 rows | yes |
| 29999 | 0.591 / 1.442 | 0.464 / 1.215 | 0.618 / 1.474 | 2.488 | 12.855 | 2.709 | 0.9988 | 4322/4326 (4 missed, 7 false), 0.27 rows | yes |

### Selection

- Selected export **29999** by: lowest validation mean per-step XYZ error with jaw accuracy >= 0.99; ties to the earlier step.
- Validation mean per-step XYZ error 2.488 mm, slot 1 0.591 mm, jaw accuracy 0.9988.
- Eligible exports (jaw ≥ 0.99): [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 29999].

### Test split (every 3rd row, 11604 observations, selected export 29999)

| step | slot 1 all mean / p95 mm | stationary | moving | chunk mean mm | chunk p95 mm | endpoint mm | jaw acc | events detected (timing) | ready |
|---|---|---|---|---|---|---|---|---|---|
| 29999 | 0.592 / 1.432 | 0.456 / 1.134 | 0.619 / 1.478 | 2.393 | 11.964 | 2.600 | 0.9990 | 4344/4345 (1 missed, 2 false), 0.22 rows | yes |

### Serving parity (public `serve_policy` factory vs evaluator)

- 200 validation observations, tolerance 0.5 mm on slot 1: **passed**.
- Slot-1 difference max 0.2723 mm, mean 0.1104 mm; whole-chunk max 0.3880 mm; slot-1 jaw agreement 1.000.

### Completion record

- Selected checkpoint `/scratch/cw5167/workspace/openpi/checkpoints/pi05_hanoi_dense_aaaa_to_cccc/hanoi_dense_20260919/exports/29999` (export.json sha256 `2084b7375cb8b445…`, normalization sha256 `ef8c5fb9e4be9a4a…`).
- Ready to deploy (offline criteria): **True**. Hardware success measured: **False**.
- W&B: https://wandb.ai/cw5167-nyu/openpi/runs/jg44v2cf

