"""Bounded queue decisions and exact-job Slurm operations for the Hanoi pipeline."""

from collections.abc import Callable
import dataclasses
import datetime
import math
import os
import pathlib
import re
import subprocess


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    constraint: str
    gpus: int
    fsdp_devices: int


PROFILES = (
    Profile("h200_1", "h200", 1, 1),
    Profile("h100_1", "h100", 1, 1),
    Profile("h100_2", "h100", 2, 2),
    Profile("a100_2", "a100", 2, 2),
)


class ObservationError(RuntimeError):
    """A scheduler query failed without establishing a terminal job state."""


def parse_time(value: str) -> float | None:
    if value in ("Unknown", "N/A", "None", "(null)", "", "UNLIMITED"):
        return None
    parsed = datetime.datetime.fromisoformat(value)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.UTC)).timestamp()


def parse_job(text: str) -> dict[str, str]:
    # Job names and paths in this pipeline contain no whitespace. Slurm reasons may.
    return dict(re.findall(r"(?:^|\s)(\w+)=(.*?)(?=\s+[\w:]+=|$)", text.strip()))


@dataclasses.dataclass(frozen=True)
class Pending:
    state: str
    eligible_time: float | None
    estimated_start: float | None
    reason: str = "Priority"


def reassess(pending: Pending, now: float) -> bool:
    if pending.state != "PENDING" or pending.eligible_time is None:
        return False
    if pending.reason not in ("Priority", "Resources", "None", "QOSGrpGRES"):
        return False
    eligible_wait = now - pending.eligible_time
    if pending.estimated_start is None:
        return eligible_wait >= 2 * 3600
    return eligible_wait >= 30 * 60 and pending.estimated_start - now > 2 * 3600


def worthwhile_replacement(
    pending: Pending,
    *,
    now: float,
    alternative_start: float | None,
    remaining_seconds: float,
    alternative_remaining_seconds: float,
    replacement_times: list[float],
) -> bool:
    if not reassess(pending, now) or alternative_start is None or len(replacement_times) >= 2:
        return False
    if replacement_times and now - max(replacement_times) < 6 * 3600:
        return False
    if pending.estimated_start is None:
        return alternative_start <= now + 3600
    return pending.estimated_start + remaining_seconds - alternative_start - alternative_remaining_seconds >= 3600


def training_seconds(*, step_seconds: float, remaining_steps: int, save_seconds: float, save_interval: int) -> float:
    """Budget periodic saves conservatively, including when saving blocks training."""
    if save_interval < 1 or remaining_steps < 0:
        raise ValueError("Save interval must be positive and remaining steps nonnegative")
    return 1.25 * step_seconds * remaining_steps + math.ceil(remaining_steps / save_interval) * save_seconds


def walltime_minutes(
    *, startup: float, step_seconds: float, remaining_steps: int, save_seconds: float, save_interval: int
) -> int:
    seconds = (
        startup
        + training_seconds(
            step_seconds=step_seconds,
            remaining_steps=remaining_steps,
            save_seconds=save_seconds,
            save_interval=save_interval,
        )
        + 2 * save_seconds
    )
    return max(60, min(720, int((seconds + 899) // 900) * 15))


class Slurm:
    def __init__(
        self, root: pathlib.Path, account: str = "torch_pr_595_tandon_advanced", *, runner: Callable = subprocess.run
    ):
        self.root = root.resolve()
        self.account = account
        self.runner = runner

    def run(self, arguments: list[str]) -> subprocess.CompletedProcess:
        return self.runner(
            arguments, cwd=self.root, capture_output=True, text=True, timeout=45, env={**os.environ, "TZ": "UTC"}
        )

    def arguments(
        self, profile: Profile, *, job_name: str, minutes: int, script: str, script_args: list[str]
    ) -> list[str]:
        if not re.fullmatch(r"hanoi-[a-zA-Z0-9_-]+", job_name):
            raise ValueError("Managed job names must have a unique Hanoi prefix")
        logs = self.root / ".cache/hanoi/logs"
        logs.mkdir(parents=True, exist_ok=True)
        return [
            "sbatch",
            "--parsable",
            f"--account={self.account}",
            "--nodes=1",
            "--ntasks=1",
            f"--gres=gpu:{profile.gpus}",
            f"--constraint={profile.constraint}",
            "--cpus-per-task=8",
            "--mem=128G",
            f"--time={minutes}",
            f"--job-name={job_name}",
            f"--chdir={self.root}",
            f"--output={logs}/%x-%j.out",
            f"--error={logs}/%x-%j.err",
            script,
            *script_args,
        ]

    def estimate(self, arguments: list[str]) -> float | None:
        response = self.run([arguments[0], "--test-only", *arguments[1:]])
        if response.returncode:
            raise ValueError(f"Invalid scheduler request: {response.stderr.strip()}")
        match = re.search(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", response.stdout + response.stderr)
        return parse_time(match[1]) if match else None

    def get_job(self, job_id: str) -> dict[str, str]:
        if not job_id.isdigit():
            raise ValueError("An exact numeric job ID is required")
        response = self.run(["scontrol", "show", "job", job_id, "-o"])
        if response.returncode:
            # Accounting is authoritative after the controller forgets a completed job.
            response = self.run(
                [
                    "sacct",
                    "--jobs",
                    job_id,
                    "--noheader",
                    "--parsable2",
                    "--format=JobIDRaw,State,ExitCode,Account,JobName%100",
                ]
            )
            if response.returncode:
                raise ObservationError("Both controller and accounting queries failed; job state is unknown")
            for line in response.stdout.splitlines():
                fields = line.split("|")
                if len(fields) >= 5 and fields[0] == job_id:
                    return {
                        "JobId": job_id,
                        "JobState": fields[1].split()[0],
                        "ExitCode": fields[2],
                        "Account": fields[3],
                        "JobName": fields[4],
                    }
            raise ObservationError("Job state is not yet available; do not launch a duplicate")
        return parse_job(response.stdout)

    def get_jobs(self, job_ids: list[str]) -> dict[str, dict[str, str]]:
        """Batch the pending-start query, then retain authoritative per-job details."""
        if not job_ids:
            return {}
        if any(not job_id.isdigit() for job_id in job_ids):
            raise ValueError("Exact numeric job IDs are required")
        response = self.run(["squeue", "--start", "--noheader", "--jobs", ",".join(job_ids), "--format=%i|%S|%R"])
        # squeue can reject IDs that have already left the controller. get_job
        # falls back to accounting, so a finished job must not get stuck here.
        pending = {}
        for line in response.stdout.splitlines():
            parts = line.strip().split("|", 2)
            if len(parts) == 3 and parts[0] in job_ids:
                pending[parts[0]] = {"QueueStartTime": parts[1], "QueueReason": parts[2]}
        result = {job_id: {**self.get_job(job_id), **pending.get(job_id, {})} for job_id in job_ids}
        if response.returncode:
            for job in result.values():
                job["QueueObservationError"] = response.stderr.strip()
        return result

    def cancel_pending(self, job_id: str, *, expected_name: str) -> bool:
        """Return true only after this exact owned pending job is confirmed canceled."""
        job = self.get_job(job_id)
        if job.get("JobState") != "PENDING":
            return False
        if (
            job.get("JobName") != expected_name
            or job.get("Account") != self.account
            or job.get("WorkDir") != str(self.root)
            or not job.get("UserId", "").endswith(f"({os.getuid()})")
        ):
            raise ValueError("Refusing to cancel a job outside this managed run")
        response = self.run(["scancel", "--ctld", "--state=PENDING", job_id])
        if response.returncode:
            return False
        # The controller-side PENDING filter prevents canceling a job that started
        # between the ownership query and scancel. Ambiguity means no replacement.
        return self.get_job(job_id).get("JobState") == "CANCELLED"

    def find_submission(self, name: str, since: float) -> list[str]:
        """Reconcile a unique submission name in both the live queue and accounting."""
        start = datetime.datetime.fromtimestamp(since - 60, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")
        queries = [
            ["squeue", "--noheader", "--name", name, "--format=%i|%a|%j"],
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--name",
                name,
                "--starttime",
                start,
                "--format=JobIDRaw,Account,JobName%100",
            ],
        ]
        found = set()
        for query in queries:
            response = self.run(query)
            if response.returncode:
                raise ObservationError("Cannot reconcile submission while scheduler queries fail")
            for line in response.stdout.splitlines():
                fields = line.strip().split("|")
                if len(fields) >= 3 and fields[0].isdigit() and fields[1] == self.account and fields[2] == name:
                    found.add(fields[0])
        return sorted(found)

    def cancel_owned(self, job_id: str, *, expected_name: str) -> None:
        """Stop an owned job only for an exhausted failure budget, not queue optimization."""
        job = self.get_job(job_id)
        if (
            job.get("JobName") != expected_name
            or job.get("Account") != self.account
            or job.get("WorkDir") != str(self.root)
            or not job.get("UserId", "").endswith(f"({os.getuid()})")
        ):
            raise ValueError("Refusing to stop a job outside this managed run")
        response = self.run(["scancel", "--ctld", job_id])
        if response.returncode:
            raise ObservationError("Could not confirm cancellation of the owned job")
