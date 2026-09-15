# Scratch quota incident, September 15, 2026

Observed at 05:04 UTC (01:04 EDT). No training job is currently running or queued.
OpenPI job `17789900` failed at 03:43:14 UTC; Cosmos pilot `17824058` failed at
03:41:53 UTC. Only the user's two CPU allocations remain running.

Scratch quota reported 5,010,237,401,432 bytes used against a 5,000,000,000,000-byte
limit. OpenPI's log ends during the step-24,500 save, without a final exception
traceback. Quota exhaustion is consistent with the failure; its exact exception
was not captured. Both OpenPI background monitors are absent on their recorded
host, cs653. Their previous heartbeat records are historical.

## OpenPI recovery performed

The committed checkpoint is
`checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/24375`.
Its commit metadata is present and its saved optimizer-step scalar reads 24,376.
This verifies saved progress, not a full restore of every model/optimizer tensor.
All four inference exports at 5,000, 10,000, 15,000, and 20,000 remain.

Only the failed job's uncommitted `24500.orbax-checkpoint-tmp-680` directory was
removed. Before removal, accounting confirmed the exact job terminal, no Hanoi
job was queued/running, and the manager, pipeline, and checkpoint-writer locks
were acquired. Removal released files occupying 44,694,510,592 allocated bytes.
Raw data, converted data, complete checkpoints, logs, and other projects were
preserved. Small scratch writes subsequently succeeded, but the quota display
still showed its previous full reading; available headroom is not yet verified.
The full scratch scan subsequently totaled 4,965,671,465,472 allocated bytes,
suggesting only about 34.3 GB of remaining space. At 05:06:30 UTC, the existing
storage validator recorded a failed preflight: zero verified quota headroom
against 237.717 GB additional peak need including reserve. Its result is in
`data/hanoi/runs/hanoi_20260914/storage_after_quota_cleanup.json`.

Evidence is in `data/hanoi/runs/hanoi_20260914/production_17789900_quota_cleanup.json`
and `pre_resume_24375_verification.json`. Manager state now records the failed job
and an explicit quota hold. No replacement was submitted.

## Measured storage

These are allocated decimal bytes measured with `du`, with hard links counted
once per scan. The rows overlap and should not be added together.

| Location | Size | Meaning |
| --- | ---: | --- |
| OpenPI checkpoints before cleanup | 139.2 GB | One committed full state, one failed temporary state, four exports |
| OpenPI checkpoints after cleanup | 94.5 GB | One committed full state and four inference exports |
| Entire OpenPI repository after cleanup | 193.9 GB | Includes checkpoints, converted data, dependencies and caches |
| Separate `workspace/cosmos-policy` repository | 281.7 GB | Includes its pilot checkpoints and base weights |
| Cosmos pilot's checkpoint series | 219.2 GB | Eight full checkpoints, approximately 27.4 GB each |
| Older `/scratch/cw5167/stable-wm` tree | 1,046.4 GB | Existing datasets, checkpoints and bundles; untouched |

OpenPI's full-checkpoint retention is `max_to_keep=1`: the frequent 125-step saves
were not all retained. A new save still needs approximately 44.7 GB of temporary
space before replacing the previous full state. Saving every 500 steps would
reduce I/O overhead, but would not remove that temporary-space requirement.
The previously recommended cadence change has not been applied.

## Cosmos cleanup not approved; leave Cosmos alone

The user subsequently instructed **"leave Cosmos alone."** This supersedes the
earlier pending question. No Cosmos files were changed or deleted, and the
proposal below is retained only as a record of what was discussed.

The separate run directory is:

```text
/scratch/cw5167/workspace/cosmos-policy/data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_aaaa_to_cccc_20260914_pilot_retry1/checkpoints
```

It contains full checkpoints at steps 2, 100, 200, 300, 400, 500, 600, and 700,
written between approximately 03:05 and 03:40 UTC. The log reports step 700 saved,
and `latest_checkpoint.txt` points to it. Both 600 and 700 have model metadata;
no full restore of these Cosmos checkpoints was performed by this OpenPI agent.

The proposed cleanup removes only `iter_000000002`, `iter_000000100`,
`iter_000000200`, `iter_000000300`, `iter_000000400`, and `iter_000000500`:
approximately 164.4 GB. It preserves 600 and 700, base weights, datasets, and logs.
This cleanup alone would still leave the full three-model storage reserve short
if the directory-size estimate is accurate. These trained checkpoints belong to
another project, and their removal was not approved. No Cosmos files have been
changed, and no message was sent to another agent.

The subsequent full scratch audit is in `hanoi_scratch_usage_20260915.md`. Older
world-model experiments account for most of the 4.966 TB total, including
1.476 TB of old Hanoi `.ckpt` files and 162.2 GB of temporary Git pack files.

## Before resuming OpenPI

At 05:19:43 UTC, the user-approved deletion of only
`/scratch/cw5167/stable-wm/checkpoints` completed, reclaiming 299.284 GB.
Datasets and bundles were preserved. Cosmos remains untouched. At 05:25:41 UTC,
the quota service reflected the cleanup and the existing preflight passed:
320 GB conservative headroom against 237.717 GB additional peak need. Usage was
4.67 TB / 5.00 TB. Continuation job **17829548** was submitted to resume checkpoint
24,375. At 05:26:24 UTC it was PENDING (Priority). The manager and W&B bridge were
restored and verified on cs673; full GPU restore remains to be verified after the
job starts. Recovery evidence is `quota_recovery_17789900.json` in the run directory.
Evidence: `data/hanoi/runs/hanoi_20260914/stable_wm_checkpoint_cleanup.json`.

Obtain a fresh passing quota forecast; the pipeline reserves enough for all three
serial runs, their temporary saves and inference candidates. Do not bypass this
gate or assume the optional cleanup alone is sufficient. Resume from checkpoint
24,375 with the original training/data identity, charging the diagnosed retry
against the existing allocation budget. Then verify the full GPU restore and
restart exactly one manager plus the configured W&B bridge. Reverse, multitask,
all production evaluation, and final delivery remain outstanding.
