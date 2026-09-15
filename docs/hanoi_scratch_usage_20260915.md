# Scratch usage audit — September 15, 2026

Most space belongs to older world-model experiments. The September OpenPI and
Cosmos runs account for a much smaller part of the total.

The complete scan finished at **05:16 UTC (01:16 EDT)** and measured
**4,965,671,535,616 allocated bytes (4.966 TB)** after the earlier OpenPI cleanup.
Shared hard-linked files were counted once across the entire scratch tree.
**Subsequent approved cleanup, 05:19 UTC:** removed only `stable-wm/checkpoints`,
reclaiming 299,284,472,320 bytes. Datasets, bundles, and all Cosmos files were
preserved. The table below is the audit snapshot before this deletion; subtracting
the removed files estimates usage at about **4.666 TB**. The quota service refreshed
at 05:25 UTC to **4.67 TB / 5.00 TB**. OpenPI's storage preflight passed, and
continuation job **17829548** is queued to resume from checkpoint 24,375.
The quota service still reported 5.010 TB at 05:08 UTC. Small writes had resumed,
but the OpenPI storage preflight remained failed. The audit itself was read-only;
the checkpoint deletion above was subsequently authorized separately by the user.

## Largest project directories

These totals come from one whole-tree scan and reconcile with the overall usage.
GB/TB use decimal units. Shared files are attributed to whichever directory the
scan encountered first. Detailed subdirectory measurements below were performed
separately and may include shared files counted elsewhere in this table.

| Directory under `/scratch/cw5167` | Size | Main contents |
| --- | ---: | --- |
| `workspace/discrete-la-wm` | 3,049.2 GB | Older experiment runs, Git objects, quantization results and handoff packages |
| `stable-wm` | 1,046.4 GB | 663.8 GB of datasets, 299.3 GB of saved models, 70.1 GB of bundles |
| `workspace/cosmos-policy` | 281.7 GB | 219.2 GB of pilot checkpoints, 49.7 GB of base weights, dependencies |
| `workspace/openpi` | 193.9 GB | 94.5 GB of retained checkpoints, 37.9 GB of data, 60.5 GB of caches/dependencies |
| `workspace/Isaac-GR00T` | 185.3 GB | 123.7 GB of fine-tuning outputs, a large filesystem overlay, environment and code |
| All other directories | 209.1 GB | Current raw datasets, caches, containers, overlays and smaller projects |
| **Total** | **4,965.7 GB** | **About 4.966 TB** |

Other individually measured folders include the current raw datasets (56.0 GB), Apptainer cache (40.7 GB), general
cache (29.5 GB), containers (28.5 GB), overlays (16.3 GB), Hugging Face cache
(15.9 GB), UV cache (13.5 GB), and VS Code server files (7.1 GB). Some installed
dependencies share hard links with caches.

## What fills the old projects

### `discrete-la-wm/runs/hanoi-states`: 1,518.3 GB

This directory contains **131 old Slurm job directories**. It is mostly saved
models, rather than Slurm text logs. There are 3,634 checkpoint paths representing
3,626 distinct checkpoint files, occupying **1,476.0 GB** after hard-link
deduplication. The `.ckpt` payload is split nearly equally between:

- Files ending in `_object.ckpt`: **738.0 GB**.
- Files ending in `_weights.ckpt`: **738.0 GB**.

Many runs save both forms at best, numbered-epoch, and final checkpoints. For
example, the August 7 job `15455558` keeps six approximately 4.39 GB files for
best/epoch-3/final object and weights variants. Different formats have not been
proven interchangeable or byte-identical; choosing what to remove requires
checking the corresponding loaders and which models the user wants to retain.

### Other large `discrete-la-wm` outputs

| Path relative to `workspace/discrete-la-wm` | Size | Observed contents |
| --- | ---: | --- |
| `runs/pushbox-pilot` | 279.2 GB | 25 job directories; 740 `.ckpt` files occupy 272.7 GB, with 370 object and 370 weights paths |
| `runs/non_curated_quantization` | 188.0 GB | Includes 137.4 GB in the feature directory |
| `runs/hanoi_noisy_action_retraining` | 129.4 GB | Multiple runs, including model outputs, a 32.4 GB cache and 26.1 GB of bundles |
| `runs/hanoi_20260812_14_abstraction_sweep` | 106.9 GB | Older experiment sweep outputs |
| `runs/training-acceleration` | 81.6 GB | Older experiment outputs |
| `runs/hanoi_ec_quoted_capacity` | 69.5 GB | Older experiment outputs |
| `quantized_states` | 67.7 GB | Derived quantization outputs |
| `local_handoff` | 62.5 GB | Several model/handoff packages |

### `discrete-la-wm/.git`: 200.3 GB

`git count-objects -vH` reports 33.16 GiB of loose objects, 2.35 GiB of normal
packfiles, and **151.03 GiB of garbage**. That garbage consists of these seven
temporary pack files, totaling **162,176,091,136 allocated bytes (162.2 GB)**:

```text
.git/objects/pack/tmp_pack_1EpDVG
.git/objects/pack/tmp_pack_3vXggF
.git/objects/pack/tmp_pack_9jf06J
.git/objects/pack/tmp_pack_GABSmP
.git/objects/pack/tmp_pack_Mzl1AC
.git/objects/pack/tmp_pack_XU4k2R
.git/objects/pack/tmp_pack_diueUH
```

Their modification dates range from September 3 to September 13. They are a
promising cleanup candidate, subject to confirming no active Git operation owns
them and obtaining authorization for cleanup in this separate repository.
The ordinary `.pack` and `.idx` files are repository data and were not modified.

### `stable-wm`: 1,046.4 GB

The user identified stable-wm as deprecated and then explicitly approved deleting
only its checkpoints. Dependency checks found that `discrete-la-wm/data/stablewm`
points to this tree's `datasets` directory, multiple discrete-la-wm configurations
use these recordings, deployment scripts reference its bundles, and older le-wm
scripts reference its checkpoints. The current OpenPI runtime and conversion
metadata contain no stable-wm dependency; its September recordings are elsewhere.

The approved cleanup of **only `/scratch/cw5167/stable-wm/checkpoints`** completed
at 05:19:43 UTC, reclaiming 299,284,472,320 bytes while retaining datasets and bundles.
Its removal manifest is in `data/hanoi/runs/hanoi_20260914/stable_wm_checkpoint_cleanup.json`.
Before that removal, the full tree would have released approximately
1,020,406,479,360 bytes: 26.014 GB of bundle files have hard links outside the
tree and would remain allocated elsewhere. Removing the whole tree would also
break the dataset symlink and references described above.
Evidence: `data/hanoi/runs/hanoi_20260914/stable_wm_retirement_assessment.json`.

- **570.9 GB** in `datasets/tower_hanoi`: nine older HDF5 recordings from August,
  including the 136.8 GB August 20 recording and 99.0 GB August 15 recording.
- **298.4 GB** in `checkpoints/pushbox`, almost entirely `.pt` files. These include
  many numbered per-epoch weights; individual files reach approximately 263 MB.
- Additional datasets and bundles account for the remainder.

The current raw September AAAA-to-CCCC and CCCC-to-AAAA files are separately
located in `/scratch/cw5167/datasets` and occupy approximately 56.0 GB together.

## Cleanup priorities to review

1. The **162.2 GB of temporary Git pack files** above.
2. Retention of old world-model checkpoints, particularly the 1.476 TB of Hanoi
   snapshots and the older Pushbox snapshots.

**Cosmos is excluded from cleanup.** The user explicitly instructed, "leave Cosmos
alone." The earlier proposal to remove six Cosmos checkpoints was not approved;
all Cosmos files remain untouched.

Raw recordings and selected/needed model artifacts should be identified before
choosing broader removals. No message was sent to another agent. Completed cleanup
consists of the 44.7 GB failed OpenPI temporary checkpoint and the explicitly
approved 299.3 GB stable-wm checkpoint directory; no Cosmos file was changed.
See `hanoi_storage_incident_20260915.md` and the removal manifest above.

The raw inventories and summary are saved under
`data/hanoi/runs/hanoi_20260914/`: `scratch_usage_audit_20260915.json`,
`hanoi-scratch-total-20260915.tsv`, `hanoi-scratch-detail-20260915.tsv`,
`hanoi-largest-runs-20260915.tsv`, `hanoi-large-file-types-20260915.json`, and
`hanoi-checkpoint-format-totals-20260915.json`.
