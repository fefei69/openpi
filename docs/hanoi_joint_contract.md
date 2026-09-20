# Hanoi joint-state forward training, version 3

The user authorized the September 15 recollection for AAAA to CCCC, with a full
30,000-update fine-tune from pi0.5 base. There is no 5,000-update pilot stage.
The latest save request is every 2,000 updates, plus the final update.

For the data pipeline, training settings, and a standalone loader for another
trainer, see the [Cosmos training handover](hanoi_cosmos_training_handoff.md) and
its [machine-readable snapshot](hanoi_cosmos_training_handoff.json).

## Observation and action contract

The policy request contains one cropped external RGB image, seven measured state
values, the measured Cartesian XYZ context, and the forward prompt:

```python
{
    "observation/image": rgb,  # uint8 (224, 224, 3), RGB
    "observation/state": np.array([q0, q1, q2, q3, q4, q5, gripper_stroke], np.float32),
    "observation/cartesian_position": np.array([x, y, z], np.float32),
    "prompt": PROMPTS["aaaa_to_cccc"],
}
```

Use `openpi.policies.hanoi_policy.PROMPTS["aaaa_to_cccc"]` for the exact prompt.
Joint angles are measured SDK arm indices 0–5 in radians. Gripper feedback is
the driver's stroke in metres. State and Cartesian XYZ must come from the same
SDK snapshot. No velocity, force, temperature, board label or future command is
fed to the model. Cartesian XYZ is used only for action conversion and execution;
the learned state has seven values. The generic policy server retains this
auxiliary XYZ through its output transform without adding it to model state.

The camera crop stays `square_roi x=151 y=90 size=360 -> 224`; reuse the existing
ROS decoding/camera preprocessing and do not crop an already cropped image twice.
Both missing wrist cameras are masked. The checkpoint supplies the normalization.

Output remains `(8, 4)` absolute base-frame Cartesian destinations and jaw intent:
`[x_m, y_m, z_m, open_after_arrival]`. Internally XYZ labels are offsets from the
separate measured Cartesian context, never from joint angles. Intent uses 0/1,
threshold 0.5. Execute only the first waypoint, then observe and replan.

`examples/hanoi/joint_execution.py` packs a matching request for the existing
Cartesian `SparseExecutor`. Arrival, controller readiness, gripper dwell, fresh
image timing and stop handling retain the version-two execution contract. The
hardware adapter still supplies a calibrated arrival tolerance; the data's
matching tolerance is not an execution tolerance. No elapsed-time target skipping.

## Data and verification

Raw forward source:
`/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260915_223424_AAAA_to_CCCC.h5`
and its matching JSON. Fifty successful episodes contain 360,050 dense rows.

The converter preserves gripper events while simplifying `action_abs` within
0.5 mm. It retains the accepted nearby-observation approximation: a fresh real
image after the preceding reference/gripper dwell, within a local 0.2 s window
and 2 mm measured matching distance. It does not claim those moving observations
are physically stopped. Command-return timestamps are not treated as arrivals.

The 6,300 selected targets yield 6,254 admitted decision observations. Forty-six
unmatched boundaries are reported; all 1,500 gripper events remain in the labels.
Episode-pair indices 0–39 train, 40–44 validate, and 45–49 test: respectively
5,031 / 611 / 612 observations. Normalization uses only the training split.

Native LeRobot data: `data/lerobot/local/hanoi_joint_20260915_aaaa_to_cccc`.
Only selected decision images are copied (about 332 MB of Parquet). NPZ archives
under `data/hanoi/joint_v3/indices` contain explicit eight-target chunks, padding,
native observation indices, raw observation/action indices and both episode bounds.
Native nominal timestamps are storage metadata, not physical waypoint durations;
the loader never queries actions using timestamp offsets. The final target repeats
for terminal padding, contributes to training loss, and is excluded from accuracy.

Every selected image and numeric observation was compared with the raw source.
Verification independently checks raw target indices, freshness, gripper dwell,
matching distances, train/held-out episode separation and native transforms.
Source and converted file hashes are recorded in `joint_v3/audit.json`; admission
and file-inventory checks run again before training or evaluation.

## Full training and continuation

Config: `pi05_hanoi_joint_aaaa_to_cccc`; experiment: `hanoi_joint_20260917`.
Full fine-tuning starts from pi0.5 base, global batch 32, H=8, 30,000 updates.
Learning rate warms up for 1,000 updates to 2.5e-5 and decays to 2.5e-6 over the
30,000-update schedule. W&B logs metrics/configuration, without image/weight uploads.

Save and export every 2,000 updates, plus final `29999`: fifteen inference exports
and one retained resumable state. Forecast includes a second full state during
save and 50 GiB reserve. No other workspace is deleted for this run.

H200 nodes remained drained after maintenance. The full run requests two H100s
with FSDP=2, eight CPUs, 128 GB RAM and a 24-hour allocation. Global batch stays
32. This avoids the tight memory margin of one H100: the previous native H=8
run peaked at 75.5 GB on H200. No separate GPU training pilot is submitted.

`examples.hanoi.joint_manage` submits only forward training, then evaluation.
It persists submission intent, reconciles unknown submission responses, and
resumes scheduler timeouts/preemptions/node failures at most three times from
this experiment's latest checkpoint. Model failures stop for diagnosis. Its
state is `data/hanoi/runs/hanoi_joint_20260917/joint_manager.json`.

After all 30,000 updates, evaluate the fifteen exports on held-out validation,
lock selection by validation flow loss, then evaluate test once and compare the
selected model with the native serving factory. Report first-waypoint and
gripper-event accuracy as well as full-chunk metrics. No task-accuracy threshold
or real-hardware success is implied by completing the training budget.

Checkpoints live under
`checkpoints/pi05_hanoi_joint_aaaa_to_cccc/hanoi_joint_20260917`.
Use `scripts/serve_policy.py` with this config and its matching exported checkpoint.
The earlier Cartesian-state checkpoint is a separate baseline and is not resumed.
