# Joint-state Hanoi error audit, September 17, 2026

Run: `hanoi_joint_20260917`, config `pi05_hanoi_joint_aaaa_to_cccc`.
This is a read-only analysis of the saved validation/test reports, evaluation
implementation, prepared labels and raw recordings. No retraining, checkpoint
selection changes or Cosmos workspace changes were made.
Sections from "Correction applied" onward record later work on September 17:
the version-four label fix and its pilot run.

## Findings

The prepared labels have a consistency problem that the original admission audit
did not test. Event-preserving RDP limits geometric path error but can select very
different next destinations for nearly identical starting configurations. This
is evidence of ambiguous supervision, not proof that every model error is caused
by it. Per-prediction diagnostics are needed to quantify its contribution.

Concrete example from training episodes 25 and 39:

| Quantity | Episode 25 | Episode 39 |
| --- | --- | --- |
| Previous selected local reference row | 4799 | 4799 |
| Observation local row | 4805 | 4803 |
| First target local row | 4892 | 4841 |
| First target XYZ, metres | [0.492560, 0.088990, 0.191130] | [0.493900, -0.049352, 0.191130] |
| Target motion stage | approach_source | approach_source |
| Target gripper intent | open | open |

Both observations have the same recorded board label `[2,0,1,2]` and held-disk
label 0. Their measured XYZ differs by **0.387 mm**, and the maximum joint-angle
difference is **0.000763 rad**. Their first target labels differ by **138.35 mm**.
Camera equivalence was not checked; proximity of state alone does not establish
identical policy observations.

For the raw commanded paths from local rows 4799 through 4892, the maximum
bidirectional vertex-to-segment distance is only **1.623 mm**. Episode 39's first
target is **1.283 mm** from episode 25's recorded polyline. Thus the large label
difference is predominantly different progress along a similar path, rather than
a similarly large geometric deviation of the demonstrations.

The simplifier retains an early intermediate point on a varied approach, whereas
a straighter approach can retain only its endpoint. Later target slots can then
refer to different progress and gripper phases. A fixed-index XYZ/gripper metric
penalizes that difference. It remains useful for measuring imitation of the given
labels, but does not by itself distinguish wrong paths from different waypoint
spacing. This does not certify either predicted path as safe or successful.

An exploratory check across training episodes grouped admitted observations by
the same preceding episode-local reference row. It compared pairs with measured
XYZ distance at most 1 mm, maximum joint-angle difference at most 0.01 rad, and
equal preceding gripper intent. Among 75,339 such pairs, **14.63%** had first-target
XYZ labels more than 10 mm apart. Median label distance was zero; the 90th
percentile was **61.20 mm**. These pairs are dependent and do not constitute a
model error rate or an irreducible-error estimate. Images were not matched.

## Interpreting the reported metrics

The evaluator computes Euclidean XYZ error in metres, multiplies by 1,000, and
excludes terminal padding from physical metrics. It uses absolute targets and
restores the same observation XYZ anchor to model predictions. These checks did
not identify an obvious unit, padding or Cartesian/joint conversion mistake.
The saved serving parity check covers one training anchor with common sampling
noise; it is not a general semantic or hardware qualification.

For selected checkpoint 4000 on the 612-observation test set:

- First-target mean: 12.069 mm.
- Close events: 3.384 mm across 75 observations.
- Open events: 2.469 mm across 75 observations.
- Other 462 first targets: **15.038 mm**, derived by subtracting event error sums.
- Mean valid-horizon error: 40.077 mm.
- Gripper balanced accuracy: 79.04%, pooled across all real positions in all
  eight-target chunks. It is not a first-action-only accuracy.
- All 150 first-target gripper-event intents were correct. This does not measure
  false gripper transitions at non-event observations.

The saved reports contain aggregate metrics, not individual prediction arrays or
per-slot gripper confusion matrices. Consequently they cannot establish how much
error comes from target-index shifts, along-path variation, perpendicular error,
or wrong task decisions. Those require a separate prediction diagnostic.

## Checkpoint selection correction

Selection used minimum validation flow loss, which chose step 4000. It did not
choose minimum physical first-target or grasp error. The claim that rising flow
loss meant later checkpoints were generally worse was too broad.

| Checkpoint | Validation flow loss | First XYZ mm | Close XYZ mm | Open XYZ mm | Mean horizon XYZ mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4000 | 0.020523 | 10.746 | 3.310 | 2.528 | 41.092 |
| 20000 | 0.029402 | 9.773 | 1.157 | 1.023 | 38.856 |
| 29999 | 0.032304 | 10.072 | 0.859 | 0.741 | 39.286 |

These comparisons use **validation**, not new test results. Test was evaluated
only for selected step 4000. Native flow loss scores noisy normalized flow over
all eight targets and 32 padded model dimensions; it is a different objective
from sampled first-target physical error. No per-dimension loss decomposition
was collected, so the role of padding in the selection mismatch is not known.

## Next diagnostic and design decisions

Retain the raw data, episode splits, measured joint/gripper state, external
Cartesian contract and H=8/prefix=1 interface. Before another training run:

1. Save predictions on validation, with per-slot XYZ errors and gripper confusion,
   separating first-action accuracy, along-path distance and perpendicular error.
2. Audit a revised extraction rule with consistent spatial progress across similar
   paths, preserving gripper events and the geometric error budget. Keep variable
   target counts per transfer; do not impose a fixed eight-step transfer template.
3. Specify checkpoint selection using deployment-relevant validation metrics.
   Preserve the already reported test result and distinguish later exploratory
   evaluations from the original locked selection.

References: `examples/hanoi/sparse_dataset.py::select_waypoints`,
`examples/hanoi/evaluation.py::Metrics`, `examples/hanoi/evaluate.py`,
`data/hanoi/joint_v3/audit.json`, and the run's `validation_*.json`/`test.json`.

## Correction applied: version-four recorded-destination labels and pilot

Added September 17, 2026, 16:00 EDT, after the audit above.

**Extraction change.** `examples/hanoi/waypoint_extraction.py::select_waypoints`
replaces RDP simplification for the new `pi05_hanoi_waypoint_aaaa_to_cccc`
config. It keeps the recording's own motion-leg endpoints (`leg_idx` boundaries)
plus every gripper transition, and merges an arrival with the gripper change that
immediately follows it into one destination. Intermediate motion noise stays in
the observations but is never a destination label. Nothing else in the contract
changed: same joint state, same Cartesian context, H=8, prefix 1, 30 Hz images.

**Resulting data (`data/hanoi/waypoint_v4`).**

| Quantity | Value |
| --- | ---: |
| Destinations per episode | 90 (6 per transfer: approach, descend+close, lift, approach, descend+open, lift) |
| Admitted observations (train / val / test) | 3,589 / 445 / 450 (4,484 total) |
| Unmatched boundaries | 16, all "no fresh frame after gripper dwell" |
| Gripper events retained in labels | 1,500 of 1,500 |
| Max recorded-path deviation from label polyline | 2.00 mm (budget 2.5 mm) |
| Similar-configuration training pairs audited | 69,773 |
| Max next-destination label distance among them | 0.0 mm (v3: 61 mm at p90) |

All 50 episodes share exactly the same 90 destination XYZs, and only 18 distinct
destination points exist in the whole recording. The first-target task is
therefore effectively a choice among discrete points plus a jaw intent; the
shortest move between an observation and its first target is 40.6 mm. Mean XYZ
error will be nearly bimodal, so p95, the within-5 mm fraction and first-action
jaw accuracy are the informative numbers, not the mean alone.

**Pilot run.** `hanoi_waypoint_pilot_20260917`, Slurm job 17926752, one H200,
4,001 updates, batch 32, same learning-rate schedule as the v3 run. Full
checkpoints and exports at 2000 and 4000. In-training validation
(`examples/hanoi/waypoint_validation.py`) samples the EMA parameters on all 445
validation observations at steps 0, 250, 500, 1000, 2000 and 4000 with fixed
noise (seed 43; seed 44 added at 4000) and writes
`data/hanoi/runs/hanoi_waypoint_pilot_20260917/pi05_hanoi_waypoint_aaaa_to_cccc/learning_curve/validation_<step>_seed<seed>.{json,npz}`
with first-target mean/percentile XYZ error, along- and cross-track components,
per-slot errors, close/open event errors, and first-action jaw accuracy. The
run ends with a native serving-parity check on the step-4000 export. The test
split is not consulted. Results are appended below when available.

### Pilot results (validation split, 445 observations, EMA parameters, 10 sampling steps)

Job 17926752 ran on one H200 from 19:13 EDT; step 4000 was reached at about
20:33 EDT (roughly 1.1 s per update including validation passes).

| Step | First-target mean | Median | p95 | Within 5 mm | Jaw, first action | Close / open event | All 8 slots mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 136.0 mm | 126.9 | 245.4 | 0% | 39% | 187.9 / 123.2 mm | 138.6 mm |
| 250 | 64.0 mm | 56.2 | 137.6 | 0.4% | 76% | 94.1 / 51.0 mm | 73.2 mm |
| 500 | 15.7 mm | 14.4 | 31.5 | 7% | 100% | 17.3 / 17.2 mm | 21.1 mm |
| 1000 | 8.4 mm | 7.7 | 16.6 | 28% | 100% | 10.0 / 8.5 mm | 10.1 mm |
| 2000 | 4.2 mm | 3.8 | 8.5 | 67% | 100% | 5.5 / 3.5 mm | 5.0 mm |
| 4000, seed 43 | **2.41 mm** | 2.16 | 5.00 | 94.8% | 100% | 2.63 / 1.85 mm | 2.70 mm |
| 4000, seed 44 | 2.42 mm | 2.06 | 5.29 | 93.9% | 100% | 2.90 / 2.15 mm | 2.73 mm |

At step 4000 the largest first-target error over both seeds is 9.8 mm and no
prediction is on the wrong peg (no XY error exceeds 20 mm; pegs are 40 mm
apart). The 18 destinations are not all 40 mm apart, though: hover points at
the same peg differ by 2.5 to 3.6 mm and stack heights by about 10 mm. Judged
by nearest destination, 42 of 445 (seed 43) and 41 of 445 (seed 44) first
targets are closer to a neighbouring point than to their label, almost all
among the closely spaced hover points; at grasp/release events, 0 of 147
(seed 43) and 6 of 147 (seed 44) are nearer the adjacent stack height, with
event z errors up to 5.1 / 7.5 mm. Per-slot mean error rises only from 2.4 mm
(slot 1) to 3.1 mm (slot 8), and jaw balanced accuracy is 100% in every slot.
Along-track and cross-track components of the first-target error are 1.75 and
1.28 mm. The two sampling seeds agree within 0.1 mm on every mean.

For reference, the v3 run on RDP labels had, at its own step 4000 on the same
validation episodes: 10.75 mm first target, 41.1 mm over eight slots, 53.8 mm
on the last valid slot, 79% pooled jaw balanced accuracy; and at 29,999 updates
still 10.07 mm / 39.3 mm. The v4 pilot passed those numbers before step 1000.

The previous agent's diagnostic threshold (mean ≤3 mm, p95 ≤10 mm, jaw ≥98%,
close/open ≤2 mm) is missed only on the close-event term (2.63 and 2.90 mm
versus 2.0), and the curve was still falling at 4000. This is offline label
imitation on held-out episodes of the same recording; it is not a measurement
of hardware success, controller clearance, or robustness to visual changes.
Test-split evaluation was not run, as specified for the pilot.

### Pilot results (validation split, 445 observations, EMA parameters, 10 sampling steps)

Run `hanoi_waypoint_pilot_20260917`, Slurm 17926752, one H200, started 19:13 EDT
September 17. Fixed sampling noise, seed 43 (seed 44 added at step 4000).

| Step | First-target XYZ mean | Median | p95 | Within 5 mm | Close / open event error | Mean over 8 slots | Jaw accuracy, 1st action / all slots | Train loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 136.0 mm | 126.9 | 245.4 | 0% | 187.9 / 123.2 | 138.6 | 39% / 48% | 0.291 |
| 250 | 64.0 | 56.2 | 137.6 | 0.4% | 94.1 / 51.0 | 73.2 | 76% / 68% | |
| 500 | 15.7 | 14.4 | 31.5 | 7% | 17.3 / 17.2 | 21.1 | 100% / 99.9% | 0.0093 |
| 1000 | 8.4 | 7.7 | 16.6 | 28% | 10.0 / 8.5 | 10.1 | 100% / 100% | 0.0054 |
| 2000 | 4.2 | 3.8 | 8.5 | 66% | 5.5 / 3.5 | 5.0 | 100% / 100% | 0.0026 |
| 4000, seed 43 | **2.41** | 2.16 | 5.00 | 94.8% | 2.63 / 1.85 | 2.70 | 100% / 100% | 0.0016 |
| 4000, seed 44 | **2.42** | 2.06 | 5.29 | 93.9% | 2.90 / 2.15 | 2.73 | 100% / 100% | |

For reference, the v3 run on RDP labels (same split, same evaluator conventions):
first target 10.75 mm at step 4000 and 10.07 mm at step 29999; mean over eight
slots 41.1 mm and 39.3 mm; pooled jaw balanced accuracy 79.3% and 80.0%.

Observations:

- The curve halved roughly every 500 updates from step 500 onward and had not
  flattened at 4000 (loss was still falling: 0.0026 at 2000, 0.0018 at 3500).
- At step 4000 no validation prediction is more than 10 mm from its target
  (worst 9.8 mm, seed 43; 8.4 mm, seed 44). There are no wrong-destination
  predictions; the residual is small imprecision around the correct point,
  1.75 mm along the motion direction and 1.3 mm across it.
- Per-slot error at 4000 grows only mildly with horizon: 2.4, 2.4, 2.7, 2.8,
  2.8, 2.8, 2.8, 3.1 mm for slots 1 to 8 (v3 at step 4000: 10.7 mm for slot 1,
  53.8 mm for the last valid slot).
- Gripper intent has been 100% correct on every slot since step 500, versus
  79% pooled balanced accuracy for v3 after 30,000 updates.
- The previous agent's "diagnostic target" (mean <=3 mm, p95 <=10 mm, jaw >=98%,
  close/open <=2 mm) was met on all but the close/open threshold (2.63 to
  2.90 mm close, 1.85 to 2.15 mm open). It is a threshold set before the run,
  not a hardware-success criterion.

Conclusion: the ~10 mm floor of the v3 run was caused by inconsistent
next-waypoint labels, not by model capacity or training length. With consistent
labels the same model, data volume and schedule reach 2.4 mm mean first-target
error in 4,000 updates. No test-split evaluation and no hardware execution has
been done for this checkpoint. Full checkpoints/exports exist at steps 2000 and
4000 under `checkpoints/pi05_hanoi_waypoint_aaaa_to_cccc/hanoi_waypoint_pilot_20260917/`.

Completion: job 17926752 finished at 20:41 EDT (1 h 28 min wall). The manager
marked the pilot complete; the native serving factory reproduced the evaluation
path's output on the step-4000 export exactly (0.0 m XYZ difference, identical
jaw decisions). Exports: `checkpoints/pi05_hanoi_waypoint_aaaa_to_cccc/hanoi_waypoint_pilot_20260917/exports/{2000,4000}`
(54 GB total including the resumable step-4000 state).

Close-out: Slurm 17926752 completed at 20:41 EDT after 1 h 28 min (exit 0).
The step-4000 export passed the native serving-parity check (identical XYZ and
jaw decisions between the evaluation path and `create_trained_policy`). The
manager recorded the pilot as complete with no attention flags; the test split
was not consulted. W&B run: https://wandb.ai/cw5167-nyu/openpi/runs/7s5tn1gg.
Serving: `scripts/serve_policy.py` with config `pi05_hanoi_waypoint_aaaa_to_cccc`
and `checkpoints/pi05_hanoi_waypoint_aaaa_to_cccc/hanoi_waypoint_pilot_20260917/exports/4000`.

### Pilot results (validation split, 445 observations, EMA parameters, 10 sampling steps)

Job 17926752 completed on one H200 in 1 h 28 min (about 1.5 s per update).
Serving parity between the native evaluation path and `create_trained_policy`
passed on the step-4000 export (XYZ and jaw differences 0.0). The test split was
not consulted. Run: https://wandb.ai/cw5167-nyu/openpi/runs/7s5tn1gg

| Step | First XYZ mean | Median | p95 | Within 5 mm | Within 10 mm | Close / open event XYZ | Mean 8-slot XYZ | Jaw balanced acc. (first / all slots) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 136.0 mm | 126.9 | 245.4 | 0% | 0% | 187.9 / 123.2 mm | 138.6 mm | 39% / 48% |
| 250 | 64.0 mm | 56.2 | 137.6 | 0% | 2% | 94.1 / 51.0 mm | 73.2 mm | 76% / 68% |
| 500 | 15.7 mm | 14.4 | 31.5 | 7% | 31% | 17.3 / 17.2 mm | 21.1 mm | 100% / 99.9% |
| 1000 | 8.4 mm | 7.7 | 16.6 | 28% | 70% | 10.0 / 8.5 mm | 10.1 mm | 100% / 100% |
| 2000 | 4.2 mm | 3.8 | 8.5 | 66% | 98% | 5.5 / 3.5 mm | 5.0 mm | 100% / 100% |
| 4000, seed 43 | 2.41 mm | 2.16 | 5.00 | 95% | 100% | 2.63 / 1.85 mm | 2.70 mm | 100% / 100% |
| 4000, seed 44 | 2.42 mm | 2.06 | 5.29 | 94% | 100% | 2.90 / 2.15 mm | 2.73 mm | 100% / 100% |

Reference, v3 RDP labels on its validation split: first XYZ 10.75 mm at step
4000 and 10.07 mm at step 29999; mean 8-slot XYZ 41.1 / 39.3 mm; pooled jaw
balanced accuracy 79% at step 4000; close / open 3.31 / 2.53 mm at 4000 and
0.86 / 0.74 mm at 29999.

Observations:

- No wrong-peg predictions: the largest first-target error at step 4000 is
  9.8 mm (seed 43) and 8.4 mm (seed 44), while destinations above different
  pegs are at least 20 mm apart (the smallest observation-to-target move is
  40.6 mm). The v3 ambiguity was therefore the dominant error source, not the
  model or the Cartesian/joint contract.
- Fine destination structure matters for grasps. The 18 distinct destination
  points include near-duplicate hover points 2.5-3.6 mm apart (same peg, same
  height; interchangeable in practice) and stack-level grasp/release heights
  9.7-10.7 mm apart. At step 4000, 1 of 147 (seed 43) and 6 of 147 (seed 44)
  grasp/release predictions lie nearer to the adjacent stack height than to the
  labelled one; maximum event z error is 5.1 mm and 7.5 mm respectively. So a
  few percent of grasp heights are still off by about half a ring. The close
  and open errors were still falling at step 4000; this is the metric to watch
  in a longer run, together with the hardware adapter's arrival tolerance.
- Per-slot first-target error at step 4000 rises only from 2.4 mm (slot 1) to
  3.1 mm (slot 8); every slot has 100% jaw accuracy on both seeds.
- Along-track and cross-track components stay balanced (1.75 / 1.28 mm at
  step 4000), so the residual is isotropic imprecision around the correct
  destination rather than systematic overshoot or undershoot.
- Error roughly halves per doubling of updates through step 4000 and had not
  flattened. Training loss was 0.0016 at step 4000 versus 0.0187 for v3.
- The pilot's diagnostic target (first mean <= 3 mm, p95 <= 10 mm, jaw >= 98%,
  close/open <= 2 mm) was missed only on the 2 mm event criterion (close events
  2.6 / 2.9 mm). This is the previous analyst's threshold, not a hardware
  requirement; the hardware adapter's arrival tolerance governs execution.

Artifacts: exports at
`checkpoints/pi05_hanoi_waypoint_aaaa_to_cccc/hanoi_waypoint_pilot_20260917/exports/{2000,4000}`
(12 GB each) plus the resumable step-4000 state (54 GB total); prediction arrays
in the `learning_curve/*.npz` files; `pilot_complete.json` and
`serving_validation.json` in the run directory.

Open items before deployment: test-split evaluation of the step-4000 export
(unchanged protocol from v3), a longer run if sub-2 mm grasp/release precision
is required, and real-hardware execution of the v4 contract. Nothing here
measures task success on the robot.

## Full run on version-four labels (launched 20:59 EDT)

`hanoi_waypoint_full_20260917`, Slurm job 17934688, one H200, 30,000 updates,
batch 32, saves and inference exports every 2,000 updates plus 29999, same
learning-rate schedule as the v3 run. The in-training physical validation now
runs at steps 0/250/500/1000 and at every export step, with two sampling seeds
at the final step. The evaluation stage selects the checkpoint by **validation
mean first-target XYZ error** (ties prefer the later checkpoint) instead of flow
loss for this config; the test split is evaluated once for the selected
checkpoint, followed by the serving-parity check. Code: `examples/hanoi/joint_training.py`
(`--full`), `examples/hanoi/joint_manage.py` (`--full`, evaluate on the same
GPU family), `examples/hanoi/waypoint_validation.py::schedule`,
`examples/hanoi/evaluate.py` (criterion). Storage preflight: 430 GB available
against a 341 GB forecast, shared with the running Cosmos v4 job.

### Full-run learning curve (validation split, 445 observations, EMA parameters, seed 43; both seeds at 29999)

Job 17941147 (the first submission, 17934688, was cancelled by a cluster-wide
admin purge while still pending) ran on one H200 from 07:27 EDT on September 18;
the final update completed at about 17:35 EDT (roughly 10 h 10 min including
15 validation passes and 15 save/export pairs).

| Step | First-target mean | p95 | Within 5 mm | Close / open event | All 8 slots mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 4.15 mm | 8.32 | 69.4% | 4.75 / 3.70 mm | 4.80 mm |
| 4000 | 2.40 mm | 4.73 | 96.6% | 2.30 / 1.76 mm | 2.68 mm |
| 6000 | 1.91 mm | 4.04 | 97.5% | 2.13 / 1.82 mm | 2.09 mm |
| 8000 | 1.47 mm | 3.03 | 98.9% | 1.63 / 1.11 mm | 1.62 mm |
| 10000 | 1.22 mm | 2.56 | 99.1% | 1.22 / 0.89 mm | 1.32 mm |
| 12000 | 1.18 mm | 2.60 | 99.8% | 1.28 / 0.97 mm | 1.27 mm |
| 14000 | 1.02 mm | 2.09 | 100% | 1.23 / 0.90 mm | 1.02 mm |
| 16000 | 0.84 mm | 1.83 | 100% | 1.03 / 0.64 mm | 0.84 mm |
| 18000 | 0.74 mm | 1.45 | 100% | 0.85 / 0.66 mm | 0.79 mm |
| 20000 | 0.63 mm | 1.25 | 99.6% | 0.64 / 0.51 mm | 0.74 mm |
| 22000 | 0.65 mm | 1.34 | 100% | 0.68 / 0.50 mm | 0.66 mm |
| 24000 | 0.53 mm | 1.08 | 100% | 0.59 / 0.42 mm | 0.57 mm |
| 26000 | 0.50 mm | 1.04 | 100% | 0.62 / 0.40 mm | 0.52 mm |
| 28000 | 0.50 mm | 1.06 | 100% | 0.57 / 0.45 mm | 0.50 mm |
| 29999, seed 43 | **0.47 mm** | **0.94** | 100% | 0.47 / 0.40 mm | 0.49 mm |
| 29999, seed 44 | 0.44 mm | 0.92 | 100% | 0.49 / 0.35 mm | 0.48 mm |

Jaw intent was 100% correct in every slot from step 500 onward. At 29999 the
largest first-target error over both seeds is 1.80 mm, so no prediction is
within reach of a neighbouring stack height (about 10 mm apart) or hover point
(2.5 to 3.6 mm apart) on the validation split. The v3 run on RDP labels ended at
10.07 mm first-target / 39.3 mm eight-slot error after the same 30,000 updates.
Checkpoint selection on all fifteen exports, the single test-split evaluation
and the serving-parity check are run by the automatic evaluation stage and are
reported below when complete. Hardware success remains unmeasured.

### Full-run evaluation stage (job 17965762, one H200, 54 min)

All fifteen exports were re-evaluated on the validation split by the evaluation
stage; its numbers agree with the in-training validation to within 0.01 mm.
Selection by validation mean first-target XYZ error locked **step 29999**
(0.465 mm). The test split (450 observations, episodes 45 to 49) was then
evaluated once for that checkpoint:

| Test metric (selected step 29999) | v4 full run | v3 run (selected step 4000) |
| --- | ---: | ---: |
| First-target XYZ error, mean | **0.459 mm** | 12.07 mm |
| Mean over all valid slots | 0.493 mm | 40.08 mm |
| Last valid slot | 0.501 mm | |
| Close-event first target (75) | 0.501 mm | 3.38 mm |
| Open-event first target (75) | 0.428 mm | 2.47 mm |
| Jaw balanced accuracy, all slots | 100% (confusion [[1725, 0], [0, 1735]]) | 79.0% |
| Per-episode first-target mean | 0.44 to 0.47 mm | |

The native serving factory reproduced the evaluation path on the selected export
exactly (0.0 m XYZ difference, identical jaw decisions). The manager marked the
run complete. Selected checkpoint:
`checkpoints/pi05_hanoi_waypoint_aaaa_to_cccc/hanoi_waypoint_full_20260917/exports/29999`
(serve with `scripts/serve_policy.py` and config `pi05_hanoi_waypoint_aaaa_to_cccc`).
All fifteen exports and the resumable step-29999 state are retained (204 GB);
scratch stood at 95.7% of quota afterwards. Hardware success remains unmeasured.
