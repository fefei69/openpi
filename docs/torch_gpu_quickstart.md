# Torch GPU quick guide for another agent

Verified September 14, 2026 for user `cw5167`. Use a Torch login/CPU shell with Slurm commands available.
From outside Torch, connect with `ssh cw5167@login.torch.hpc.nyu.edu`.

Use project account `torch_pr_595_tandon_advanced` for this project. The account is present in the live association
list, and an H100/8-CPU/128-GiB/2-hour `sbatch --test-only` request passed. No job was submitted by that check.

## Inspect access and requests

```bash
my_slurm_accounts
squeue -u cw5167
sinfo -h -o '%P %G %f %t'
```

Request `--gres=gpu:1` and a feature such as `--constraint=h100`. Current feature names include `h100`, `h200`,
`a100`, `l40s`, and `rtx6000`; visibility does not guarantee immediate availability or workload compatibility.
Torch chooses partitions for normal submissions. Leave `--partition` and `--qos` unset; in this session, copying
the association listing's `normal` QoS into a request was rejected. See the
[NYU submission guide](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/).

## Submit a batch workload

Prepare dependencies, weights and data before requesting GPU time. In your own project directory, save this as
`gpu_job.sbatch`, adjusting CPUs, host RAM and time to the actual workload. `--mem` specifies host RAM, not GPU VRAM.
The resource sizes below match the validated example request; smaller workloads should request less.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=other-agent-gpu
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=agent-%j.out
#SBATCH --error=agent-%j.err

set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec "$@"
```

Pass the absolute path to your prepared Python executable and the real workload script. These commands use
placeholders which the agent must replace:

```bash
# Validate request and obtain an advisory estimate; this creates no queued job.
sbatch --test-only --chdir="$PWD" gpu_job.sbatch /absolute/path/to/python your_script.py

# Submit once. Record the returned job ID and exact arguments.
sbatch --parsable --chdir="$PWD" gpu_job.sbatch /absolute/path/to/python your_script.py
```

The batch command returns before execution starts; the workload runs when Slurm grants the allocation.
See the [sbatch reference](https://slurm.schedmd.com/sbatch.html).

For this OpenPI environment, add `source examples/hanoi/env.sh` and `unset JAX_PLATFORMS` before `exec`, run from
`/scratch/cw5167/workspace/openpi`, and pass its `.venv/bin/python`. The environment includes
`NUMPY_MADVISE_HUGEPAGE=0`, our tested mitigation for very slow NumPy checkpoint-buffer allocation on Torch.
Use your own run/checkpoint directory. The existing `hanoi_20260914` experiment is owned by its active manager;
inspect `data/hanoi/runs/hanoi_20260914/manager.json` instead of starting another manager or writing that experiment.

## Monitor and handle waiting

Replace `JOBID` below with the exact ID returned by your submission:

```bash
squeue -j JOBID -o '%.18i %.30j %.10T %.10M %.50R'
squeue --start -j JOBID
scontrol show job JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

`PENDING` is waiting; `RUNNING` has an allocation. The scheduler reports pending reasons and, when available,
estimated starts. See the [squeue reference](https://slurm.schedmd.com/squeue.html).

- `Resources`/`Priority`: retain the job and inspect its estimate. `QOSGrpGRES` means partition GPU capacity is
  exhausted; it is not proof that your individual account is invalid. Physical idle counts and test-only estimates
  do not guarantee admission. [NYU explains these limits](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/).
- Check about every ten minutes. For a long wait, test compatible GPU alternatives with the same actual resource
  request. Only choose another GPU after checking memory capacity, software compatibility and restore support.
- Replace only your own still-pending job when the new estimate is materially better. Confirm its exact ID/name,
  use `scancel --ctld --state=PENDING JOBID`, and verify cancellation before submitting one replacement. Preserve
  running jobs and avoid duplicate submissions when a command's outcome is uncertain.
- For this Hanoi pipeline, the implemented policy is stricter: reassess after 30 minutes with an ETA over two hours
  away, or two hours without an ETA; probe alternatives at most hourly; require at least one hour improvement;
  at most two replacements per model and one per six hours. The manager applies it automatically.
- Diagnose OOM, invalid data, NaN, or cancellation before retrying. Keep completed checkpoints and restore the
  same experiment. Torch enforces GPU utilization, so do setup on CPU and fix stalls instead of leaving an idle
  GPU allocation. [Current utilization policy](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/).

## Optional interactive debugging

From a login shell, this requests a new one-hour allocation (unlike attaching to an existing GPU job):

```bash
srun --account=torch_pr_595_tandon_advanced \
  --nodes=1 --ntasks=1 --gres=gpu:1 --constraint=h100 \
  --cpus-per-task=8 --mem=128G --time=01:00:00 --pty bash
```

Once allocated, run `nvidia-smi` and your prepared workload; exit the shell when finished. Batch submission is more
convenient for an agent that needs to monitor a long job across sessions. Optional Torch preemption uses
`--comment="preemption=yes;requeue=true"`; enable it only when the batch script actually restores checkpoints on
restart, and track requeued IDs to avoid duplicate jobs. [NYU preemption instructions](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/#preemptible-jobs-on-torch).
