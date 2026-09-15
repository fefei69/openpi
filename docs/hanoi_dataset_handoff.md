# Hanoi raw dataset handoff

Verified file locations and HDF5 schema on September 14, 2026. Open the source files read-only.

The four original files are directly in `/scratch/cw5167/datasets/` on NYU Torch:

```text
/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260910_195558_AAAA_to_CCCC.h5
/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260910_195558_AAAA_to_CCCC.json
/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260910_195558_CCCC_to_AAAA.h5
/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260910_195558_CCCC_to_AAAA.json
```

Each HDF5 file is approximately 28 GB in decimal units (26.1 GiB); each JSON sidecar is about 58 KiB.
The filenames retain the collection timestamp. `paired_file` and `roundtrip_id` attributes link the directions.

## What was recorded

- One physical Trossen arm and one fixed external RealSense RGB camera. No wrist camera or depth stream is used.
- Four-ring Tower of Hanoi: `AAAA` means all four rings on peg A; `CCCC` means all four on peg C.
  Board entries run from smallest to largest ring, with peg IDs 0=A, 1=B, 2=C.
- 50 episodes per direction, 100 total. Each episode has 7,201 rows at a nominal 30 Hz, about four minutes,
  and follows a 15-move route. Each HDF5 file has 360,050 rows; together they have 720,100.
- Forward and reverse episode `k` are a collection pair. The JSON sidecars contain per-episode routes, observed
  end boards, success flags, motion seeds, stale/repeated-frame counts, timing and tracking diagnostics, and config.
- Collection used segmented quintic Cartesian motion with deliberate trajectory noise. The files contain dense
  30 Hz reference samples even though robot commands were issued as motion segments. For the noisy motion profile,
  seven segments of nine ticks span a motion leg. Preserve the reference samples and intended noise.

## HDF5 fields

Here `N=360050` within each direction file. Episodes are concatenated along the first dimension.

| Field | Shape / dtype | Meaning and use |
| --- | --- | --- |
| `pixels` | `(N,224,224,3)`, uint8 | RGB, HWC, already cropped/resized during collection. |
| `proprio` | `(N,8)`, float32 | Measured XYZ in metres, measured linear velocity XYZ in m/s, measured jaw stroke in metres, then commanded jaw target. BC observes only columns `:7`. |
| `state` | `(N,6)`, float32 | Raw recorded pose; its first three columns match measured XYZ. This is different from the seven-dimensional BC state assembled from `proprio[:7]`. |
| `action_abs` | `(N,4)`, float32 | Absolute next-reference XYZ in base-frame metres and binary jaw intent: 0=close, 1=open. This is the source of BC targets. |
| `action` | `(N,4)`, float32 | Consecutive reference XYZ differences within an episode; the jaw channel remains absolute binary intent. It is not the BC chunk's anchor-relative XYZ representation. |
| `ep_offset`, `ep_len` | `(50,)`, int64 / int32 | Episode start row and length. Use these to slice episodes and prevent chunks crossing boundaries. |
| `episode_idx`, `step_idx` | `(N,)`, int64 | Direction-local episode ID and within-episode row index. |
| `command_monotonic_ns`, `image_receipt_monotonic_ns` | `(N,)`, int64 | Same-clock timestamps used to calculate observation freshness. |
| `image_timestamp_ns` | `(N,)`, int64 | Image timestamp; do not substitute it into the monotonic receipt-age calculation. |
| `board`, `goal_board` | `(N,4)`, int8 | Symbolic board metadata. Excluded from policy observations. |
| `held_disk`, `move_idx`, `phase`, `episode_success` | `(N,)`, integer | Collector/task annotations and diagnostics. Excluded from policy observations. |

The HDF5 attributes include `schema_version=2`, `action_abs_alignment=post_action_reference`,
`action_profile=segmented_quintic_velocity_noise_v1`, `rate_hz=30`, and `dry_run=False`.
`motion_config_json` and `execution_config_json` contain serialized collection/execution settings.

## BC interpretation already implemented in this repository

1. Observe `pixels[t]`, measured `proprio[t,:7]`, and the direction prompt. Exclude `proprio[t,7]`: it contains
   the commanded jaw target and would leak the action label. Symbolic board/route/phase annotations are not inputs.
2. The aligned target starts at the same row: `action_abs[t:t+63]`. Do not add a `t+1` shift. At an episode boundary,
   repeat that episode's final target to fill the horizon; never draw targets from the next episode.
3. Convert every target XYZ in that chunk to a delta from the single anchor's measured XYZ. Keep jaw intent
   absolute. At serving, invert this transform to recover absolute Cartesian references.
4. Model horizon is 63 references, approximately 2.1 seconds at 30 Hz. The planned executor commits nine references
   (0.3 seconds) before replanning, with explicit motion continuity and gripper dwell handling.
5. HDF5 RGB is already transformed. For a live 640x480 ROS `rgb8` frame, decode byte stride, crop
   `rgb[90:450,151:511]`, then resize to 224x224 with `cv2.INTER_AREA`. Do not crop the stored images again.
   The shared policy adapter maps this image to `base_0_rgb` and masks both absent wrist cameras.
6. Keep only observation anchors satisfying `0 <= command_monotonic_ns - image_receipt_monotonic_ns <= 50_000_000`.
   Keep all underlying action rows so temporal spacing and chunks remain correct. This excludes 14,208 anchors
   across the raw files; it does not remove those rows from the reference timeline.
7. Split collection pairs 0-39 as train, 40-44 as validation, and 45-49 as test, in both directions. Avoid random
   frame splitting, which would put adjacent observations from the same demonstration in different splits.

Both directions together yield 564,714 eligible training anchors, 70,563 validation anchors and 70,615 test anchors.
The earlier approximately 5.7 mm figure concerned extra error from endpoint-only trajectory approximation;
it is not a justification for deleting the deliberately noisy references.

## Existing derived data and verification

- Converted LeRobot dataset: `/scratch/cw5167/workspace/openpi/data/lerobot/local/hanoi_roundtrip_20260910/`.
- Source hashes, episode mapping and split provenance: `data/hanoi/conversion.json`.
- Completed validation evidence: `data/hanoi/data_validation.json` (`passed=true`, 720,100 numeric rows checked,
  2,300 image/chunk probes, training/serving input parity passed).
- Audit/indexing/chunk helpers: `examples/hanoi/dataset.py`.
- Conversion: `examples/hanoi/convert_hanoi_data_to_lerobot.py`.
- Shared camera/state/action contract: `src/openpi/policies/hanoi_policy.py`.
- Full training/deployment plan: `docs/hanoi_training_plan.md`.

The existing audit checked finite numeric values, all 100 legal routes, label consistency and endpoint tracking.
Visual checks covered all start/final images plus manual spot checks; they do not constitute inspection of every
intermediate frame or live policy success. Raw inputs were preserved during conversion.

For a small read-only inspection from the repository root, use the existing `.venv/bin/python` (h5py 3.16):

```python
from pathlib import Path
import h5py

path = Path('/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260910_195558_AAAA_to_CCCC.h5')
with h5py.File(path, 'r') as f:
    print({name: (value.shape, str(value.dtype)) for name, value in f.items()})
    start = int(f['ep_offset'][0])
    rgb = f['pixels'][start]                 # One image, not the full image array.
    measured_state = f['proprio'][start, :7]
    reference_chunk = f['action_abs'][start:start + 63]
```
