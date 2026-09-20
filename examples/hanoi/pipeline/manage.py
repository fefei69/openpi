"""Durable, scoped orchestration for the three Hanoi fine-tunes.

The default performs one read-only scheduling pass. --submit enables the already
authorized pipeline actions; --watch polls every ten minutes while holding one manager
lock. Only job IDs submitted by this manifest (plus the explicitly adopted preparation
job) can be replaced or retried.
"""

import datetime
import json
import logging
import math
import os
import pathlib
import re
import shutil
import subprocess
import time
import uuid

import filelock
import tyro

from examples.hanoi.pipeline import scheduling
from examples.hanoi.pipeline import storage

TASKS = ("aaaa_to_cccc", "cccc_to_aaaa", "multitask")
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}
PROFILES = {profile.name: profile for profile in scheduling.PROFILES}


def write_state(path: pathlib.Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def complete_step(directory: pathlib.Path) -> int:
    if not directory.exists():
        return -1
    steps = []
    for path in directory.iterdir():
        if not path.name.isdigit() or not (path / "_CHECKPOINT_METADATA").is_file():
            continue
        try:
            metadata = json.loads((path / "_CHECKPOINT_METADATA").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("commit_timestamp_nsecs") and (path / "params").is_dir() and (path / "train_state").is_dir():
            steps.append(int(path.name))
    return max(steps, default=-1)


class Manager:
    def __init__(self, root: pathlib.Path, exp_name: str, *, submit: bool = False, slurm=None, now=time.time):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", exp_name):
            raise ValueError("Experiment name must be a simple directory/job component")
        self.root = root.resolve()
        self.exp_name = exp_name
        self.run_dir = self.root / "data/hanoi/runs" / exp_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "manager.json"
        self.slurm = slurm or scheduling.Slurm(self.root)
        self.submit_enabled = submit
        self.now = now
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["exp_name"] != exp_name or self.state["account"] != self.slurm.account:
                raise ValueError("Manager manifest identity changed")
        else:
            self.state = {
                "version": 1,
                "exp_name": exp_name,
                "account": self.slurm.account,
                "jobs": [],
                "qualified": {},
                "unsupported": [],
                "models": dict.fromkeys(TASKS, "planned"),
                "events": [],
                "attention": None,
                "complete": False,
                "last_alternative_probe": 0,
            }
        self.state.setdefault("deferred_qualifications", {})

    def persist(self):
        self.state["updated_utc"] = datetime.datetime.fromtimestamp(self.now(), datetime.UTC).isoformat()
        write_state(self.path, self.state)

    def event(self, message: str):
        self.state["events"].append({"time": self.now(), "message": message})
        logging.info(message)
        self.persist()

    def attention(self, message: str):
        self.state["attention"] = message
        self.event(message)

    def active(self) -> list[dict]:
        return [job for job in self.state["jobs"] if job["status"] not in TERMINAL and job["status"] != "REPLACED"]

    def progress(self, job: dict) -> int:
        if job["kind"] == "train":
            return complete_step(self.root / "checkpoints" / f"pi05_hanoi_{job['task']}" / self.exp_name)
        if job["kind"] == "evaluate":
            directory = self.run_dir / f"pi05_hanoi_{job['task']}"
            return len(list(directory.glob("validation_*.json"))) + int((directory / "test.json").exists())
        if job["kind"] == "prepare":
            path = self.root / "data/lerobot/local/hanoi_roundtrip_20260910/meta/info.json"
            return json.loads(path.read_text())["total_episodes"] if path.exists() else 0
        return -1

    def adopt_preparation(self):
        if any(job["kind"] == "prepare" for job in self.state["jobs"]):
            return
        path = self.run_dir / "preparation.json"
        if not path.exists():
            raise ValueError("An explicit preparation submission manifest is required")
        preparation = json.loads(path.read_text())
        observed = self.slurm.get_job(preparation["job_id"])
        if observed.get("JobName") != preparation["job_name"] or observed.get("Account") != self.slurm.account:
            raise ValueError("Preparation job identity does not match the submitted manifest")
        job = {
            "kind": "prepare",
            "task": "prepare",
            "profile": None,
            "gpus": 0,
            "job_id": preparation["job_id"],
            "name": preparation["job_name"],
            "arguments": preparation["arguments"],
            "created": scheduling.parse_time(preparation["submission_time"]),
            "status": observed["JobState"],
            "seen_restarts": 0,
            "failure_retries": 0,
            "no_progress": 0,
            "allocations": 1,
            "allocation_limit": 4,
            "replacement_times": [],
            "handled": False,
            "observed": observed,
        }
        job["start_progress"] = self.progress(job)
        self.state["jobs"].append(job)
        self.event(f"Adopted owned preparation job {job['job_id']}")

    def submit_job(
        self, kind: str, task: str, profile: str | None, *, previous: dict | None = None, preemptible: bool = False
    ) -> dict | None:
        gpus = PROFILES[profile].gpus if profile else 0
        if sum(job["gpus"] for job in self.active()) + gpus > 2:
            raise ValueError("Submission would exceed this pipeline's two-GPU budget")
        if any(job["task"] == task and job["kind"] == kind for job in self.active()):
            raise ValueError("A job for this stage/model is already active or awaiting reconciliation")
        if kind in ("train", "evaluate") and any(job["kind"] in ("train", "evaluate") for job in self.active()):
            raise ValueError("Storage qualification permits only one training/evaluation run at a time")
        if not self.submit_enabled:
            self.event(f"Dry run: would submit {kind} {task} on {profile}")
            return None
        token = uuid.uuid4().hex[:12]
        name = f"hanoi-{kind}-{task}-{token}"
        qualification = self.state["qualified"].get(profile)
        minutes = 60
        result_path = None
        if kind == "qualify":
            qualification_exp = f"qualification_{profile}_{token}"
            result_path = self.run_dir / f"qualification_{profile}_{token}.json"
            script_args = [
                "--profile",
                profile,
                "--exp-name",
                qualification_exp,
                "--output-path",
                str(result_path),
                "--fsdp-devices",
                str(PROFILES[profile].fsdp_devices),
            ]
        elif kind == "train":
            pilot = json.loads(pathlib.Path(qualification).read_text())
            step = complete_step(self.root / "checkpoints" / f"pi05_hanoi_{task}" / self.exp_name)
            minutes = scheduling.walltime_minutes(
                startup=pilot["startup_seconds"],
                step_seconds=pilot["step_seconds_p95"],
                remaining_steps=max(0, 29999 - step),
                save_seconds=pilot["checkpoint_save_seconds"],
                save_interval=pilot["save_interval"],
            )
            script_args = [
                "--config-name",
                f"pi05_hanoi_{task}",
                "--exp-name",
                self.exp_name,
                "--qualification-path",
                qualification,
            ]
        elif kind == "evaluate":
            minutes = 720
            script_args = ["--config-name", f"pi05_hanoi_{task}", "--exp-name", self.exp_name]
        elif kind == "prepare" and previous:
            args = [argument for argument in previous["arguments"] if not argument.startswith("--job-name=")]
            args.insert(2, f"--job-name={name}")
        else:
            raise ValueError(f"Unsupported stage {kind}")
        if kind != "prepare":
            args = self.slurm.arguments(
                PROFILES[profile],
                job_name=name,
                minutes=minutes,
                script=f"examples/hanoi/scripts/{'qualify' if kind == 'qualify' else kind}.sbatch",
                script_args=script_args,
            )
            if preemptible:
                args.insert(2, "--comment=preemption=yes;requeue=true")
        job = {
            "kind": kind,
            "task": task,
            "profile": profile,
            "gpus": gpus,
            "name": name,
            "arguments": args,
            "created": self.now(),
            "job_id": None,
            "status": "SUBMITTING",
            "seen_restarts": 0,
            "failure_retries": 0,
            "no_progress": 0,
            "allocations": 1,
            "allocation_limit": 4,
            "replacement_times": [],
            "handled": False,
            "preemptible": preemptible,
        }
        if previous:
            job["previous_name"] = previous["name"]
            for key in ("failure_retries", "no_progress", "allocation_limit", "replacement_times"):
                job[key] = previous[key]
            job["allocations"] = previous["allocations"] + 1
        elif kind == "train":
            seconds = scheduling.training_seconds(
                step_seconds=pilot["step_seconds_p95"],
                remaining_steps=30000,
                save_seconds=pilot["checkpoint_save_seconds"],
                save_interval=pilot["save_interval"],
            )
            usable = max(60, 12 * 3600 - pilot["startup_seconds"] - 2 * pilot["checkpoint_save_seconds"])
            job["allocation_limit"] = math.ceil(seconds / usable) + 1
        if kind == "qualify":
            job["result_path"] = str(result_path)
            job["qualification_exp"] = qualification_exp
        job["start_progress"] = self.progress(job)
        self.state["jobs"].append(job)
        self.persist()  # A submission intent is durable before invoking sbatch.
        try:
            response = self.slurm.run(args)
        except (OSError, subprocess.TimeoutExpired) as error:
            job["status"] = "AMBIGUOUS"
            self.event(f"Submission requires reconciliation: {name}: {error}")
            return job
        job["submission_stdout"], job["submission_stderr"] = response.stdout, response.stderr
        match = re.fullmatch(r"(\d+)(?:;[\w.-]+)?", response.stdout.strip())
        if response.returncode == 0 and match:
            job["job_id"], job["status"] = match[1], "PENDING"
            self.event(f"Submitted {kind} {task}: job {job['job_id']} on {profile}")
        else:
            job["status"] = "AMBIGUOUS"
            self.event(f"Submission response requires reconciliation for {name}: {response.stderr.strip()}")
        return job

    def refresh(self):
        for job in self.active():
            if job["job_id"] is None:
                matches = self.slurm.find_submission(job["name"], job["created"])
                if len(matches) > 1:
                    self.attention(f"Multiple jobs match unique submission {job['name']}; no further submissions")
                    return
                if not matches:
                    continue  # Unknown submission outcome never authorizes a duplicate.
                job["job_id"] = matches[0]
        known = [job for job in self.active() if job["job_id"] is not None]
        observations = self.slurm.get_jobs([job["job_id"] for job in known])
        for job in known:
            observed = observations[job["job_id"]]
            if observed.get("JobName") != job["name"] or observed.get("Account") != self.slurm.account:
                raise ValueError("Tracked job identity changed")
            job["observed"] = observed
            state = observed["JobState"]
            restarts = int(observed.get("Restarts", job["seen_restarts"]))
            if restarts > job["seen_restarts"]:
                increments = restarts - job["seen_restarts"]
                progress = self.progress(job)
                job["failure_retries"] += increments
                job["no_progress"] = job["no_progress"] + increments if progress <= job["start_progress"] else 0
                job["start_progress"] = progress
                job["seen_restarts"] = restarts
                if job["failure_retries"] > 3 or job["no_progress"] >= 2:
                    if self.submit_enabled:
                        self.slurm.cancel_owned(job["job_id"], expected_name=job["name"])
                    self.attention(f"Automatic requeue budget exhausted for job {job['job_id']}")
            if (
                state in ("PREEMPTED", "NODE_FAIL")
                and observed.get("Requeue", str(int(job.get("preemptible", False)))) == "1"
            ):
                if "preemption_seen" not in job:
                    job["preemption_seen"] = self.now()
                # A delay is not evidence that automatic requeue was abandoned.
                # Keep the existing ID and diagnose, rather than risk two writers.
                if self.now() - job["preemption_seen"] >= 3600:
                    self.attention(f"Job {job['job_id']} still awaits automatic requeue after one hour")
                state = "AWAITING_REQUEUE"
            else:
                job.pop("preemption_seen", None)
            job["status"] = state
        self.persist()

    def logs(self, job: dict) -> str:
        root = self.root / ".cache/hanoi/logs"
        text = ""
        for path in root.glob(f"*{job['job_id']}.*"):
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 100000))
                text += stream.read().decode(errors="replace")
        return text

    def handle_terminal(self, job: dict):
        if job.get("handled") or job["status"] not in TERMINAL:
            return
        # Reconcile a continuation whose submission intent was persisted immediately
        # before the old process stopped. Never create a second continuation.
        if any(candidate.get("previous_name") == job["name"] for candidate in self.state["jobs"]):
            job["handled"] = True
            self.persist()
            return
        if job["status"] == "COMPLETED":
            if job["kind"] == "qualify":
                pilot = json.loads(pathlib.Path(job["result_path"]).read_text())
                checks = (
                    "qualified",
                    "checkpoint_restore_passed",
                    "ema_inference_passed",
                    "ema_evaluation_forward_passed",
                    "ema_evaluation_sampling_passed",
                )
                if not all(pilot[key] for key in checks):
                    raise ValueError("Qualification job completed without passing all required checks")
                if (
                    pilot["profile"] != job["profile"]
                    or pilot["fsdp_devices"] != PROFILES[job["profile"]].fsdp_devices
                    or pilot["global_batch"] != 32
                    or pilot["horizon"] != 63
                ):
                    raise ValueError("Qualification result differs from the requested full-model profile")
                self.state["qualified"][job["profile"]] = job["result_path"]
                # Pilot weights are disposable once their restore/inference evidence is saved.
                path = self.root / "checkpoints/pi05_hanoi_multitask" / job["qualification_exp"]
                if self.submit_enabled and path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    job["pilot_weights_removed"] = True
            elif job["kind"] == "train":
                final = (
                    self.root
                    / "checkpoints"
                    / f"pi05_hanoi_{job['task']}"
                    / self.exp_name
                    / "exports/29999/export.json"
                )
                if not final.is_file():
                    raise ValueError("Training job completed without its final inference snapshot")
                self.state["models"][job["task"]] = "trained"
            elif job["kind"] == "evaluate":
                complete = self.run_dir / f"pi05_hanoi_{job['task']}" / "complete.json"
                result = json.loads(complete.read_text())
                from examples.hanoi.evaluation import metrics as evaluation

                evaluation.validate_export(pathlib.Path(result["selected_checkpoint"]))
                # This also finishes pruning if evaluation completed its durable
                # result immediately before interruption.
                if self.submit_enabled:
                    evaluation.compact_exports(
                        self.root / "checkpoints" / f"pi05_hanoi_{job['task']}" / self.exp_name / "exports",
                        pathlib.Path(result["selected_checkpoint"]),
                    )
                self.state["models"][job["task"]] = "completed"
            job["handled"] = True
            self.event(f"Completed {job['kind']} {job['task']} job {job['job_id']}")
            return
        if job["kind"] == "qualify" and (
            job["status"] == "OUT_OF_MEMORY"
            or re.search(r"RESOURCE_EXHAUSTED|out of memory", self.logs(job), re.IGNORECASE)
        ):
            self.state["unsupported"].append(job["profile"])
            # Only this manifest's disposable, terminal pilot directory is removed.
            path = self.root / "checkpoints/pi05_hanoi_multitask" / job["qualification_exp"]
            if self.submit_enabled and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            job["handled"] = True
            self.event(f"Profile {job['profile']} failed full-batch memory qualification")
            return
        infrastructure = job["status"] in ("NODE_FAIL", "BOOT_FAIL", "PREEMPTED")
        progressing_timeout = job["status"] == "TIMEOUT" and self.progress(job) > job["start_progress"]
        if infrastructure or progressing_timeout:
            if not job.get("retry_assessed"):
                progress = self.progress(job)
                job["no_progress"] = job["no_progress"] + 1 if progress <= job["start_progress"] else 0
                if infrastructure:
                    job["failure_retries"] += 1
                job["retry_assessed"] = True
                self.persist()
            if job["failure_retries"] > 3 or job["no_progress"] >= 2 or job["allocations"] >= job["allocation_limit"]:
                self.attention(f"Retry/allocation budget exhausted for job {job['job_id']}")
                return
            continuation = self.submit_job(
                job["kind"], job["task"], job["profile"], previous=job, preemptible=job.get("preemptible", False)
            )
            if continuation is not None:
                job["handled"] = True
                self.persist()
            return
        self.attention(
            f"Job {job['job_id']} ({job['kind']} {job['task']}) ended {job['status']}; diagnose before retry"
        )

    def probe(self, profile: str, *, kind: str, task: str, preemptible: bool = False) -> float | None:
        pilot_path = self.state["qualified"].get(profile)
        minutes = 60 if kind == "qualify" else 720
        if kind == "train":
            pilot = json.loads(pathlib.Path(pilot_path).read_text())
            remaining = 29999 - complete_step(self.root / "checkpoints" / f"pi05_hanoi_{task}" / self.exp_name)
            minutes = scheduling.walltime_minutes(
                startup=pilot["startup_seconds"],
                step_seconds=pilot["step_seconds_p95"],
                remaining_steps=remaining,
                save_seconds=pilot["checkpoint_save_seconds"],
                save_interval=pilot["save_interval"],
            )
        args = self.slurm.arguments(
            PROFILES[profile],
            job_name=f"hanoi-probe-{profile}",
            minutes=minutes,
            script="examples/hanoi/scripts/qualify.sbatch",
            script_args=[],
        )
        if preemptible:
            args.insert(2, "--comment=preemption=yes;requeue=true")
        return self.slurm.estimate(args)

    def remaining_seconds(self, profile: str, task: str) -> float:
        pilot = json.loads(pathlib.Path(self.state["qualified"][profile]).read_text())
        remaining = 29999 - complete_step(self.root / "checkpoints" / f"pi05_hanoi_{task}" / self.exp_name)
        return (
            pilot["startup_seconds"]
            + scheduling.training_seconds(
                step_seconds=pilot["step_seconds_p95"],
                remaining_steps=max(0, remaining),
                save_seconds=pilot["checkpoint_save_seconds"],
                save_interval=pilot["save_interval"],
            )
            + 2 * pilot["checkpoint_save_seconds"]
        )

    def replace_long_wait(self, job: dict):
        if job["kind"] != "train" or job["status"] != "PENDING":
            return
        observed = job["observed"]
        pending = scheduling.Pending(
            job["status"],
            scheduling.parse_time(observed.get("EligibleTime", "Unknown")),
            scheduling.parse_time(observed.get("QueueStartTime", observed.get("StartTime", "Unknown"))),
            observed.get("Reason", ""),
        )
        now = self.now()
        if not scheduling.reassess(pending, now) or now - self.state["last_alternative_probe"] < 3600:
            return
        self.state["last_alternative_probe"] = now
        options = []
        directory = self.root / "checkpoints" / f"pi05_hanoi_{job['task']}" / self.exp_name
        # A fresh queue entry has no optimizer state or placement to restore. Once a
        # launcher records its placement, keep the wrapper's cross-mesh resume gate.
        preserve_mesh = self.progress(job) >= 0 or (directory / "hanoi_placement.json").exists()
        for profile, path in self.state["qualified"].items():
            pilot = json.loads(pathlib.Path(path).read_text())
            previous_count = PROFILES[job["profile"]].fsdp_devices
            if (
                preserve_mesh
                and PROFILES[profile].fsdp_devices != previous_count
                and previous_count not in pilot.get("cross_mesh_restore_from", [])
            ):
                continue
            for preemptible in (False, True):
                if profile == job["profile"] and preemptible == job.get("preemptible", False):
                    continue
                estimate = self.probe(profile, kind="train", task=job["task"], preemptible=preemptible)
                remaining = self.remaining_seconds(profile, job["task"])
                if scheduling.worthwhile_replacement(
                    pending,
                    now=now,
                    alternative_start=estimate,
                    remaining_seconds=self.remaining_seconds(job["profile"], job["task"]),
                    alternative_remaining_seconds=remaining,
                    replacement_times=job["replacement_times"],
                ):
                    options.append((estimate + remaining, profile, preemptible))
        self.persist()
        if not options or not self.submit_enabled:
            return
        _, profile, preemptible = min(options)
        job["replacement_intent"] = {"profile": profile, "preemptible": preemptible, "time": now}
        self.event(f"Pending replacement intent for job {job['job_id']}: {profile}, preemptible={preemptible}")
        self.advance_replacement(job)

    def defer_optional_qualification(self, job: dict):
        if job["kind"] != "qualify" or job["status"] != "PENDING" or not self.state["qualified"]:
            return
        observed = job["observed"]
        if observed.get("Reason") not in ("Priority", "Resources", "None", "QOSGrpGRES"):
            return
        eligible = scheduling.parse_time(observed.get("EligibleTime", "Unknown"))
        if eligible is None:
            return
        now = self.now()
        estimate = scheduling.parse_time(observed.get("QueueStartTime", observed.get("StartTime", "Unknown")))
        waited = now - eligible
        if waited < 7200 and not (waited >= 1800 and estimate is not None and estimate - now > 7200):
            return
        if not self.submit_enabled:
            return
        job["qualification_deferral_intent"] = {"time": now, "eligible_time": eligible, "reason": observed["Reason"]}
        self.event(f"Deferring optional qualification {job['job_id']} after its eligible wait limit")
        self.advance_qualification_deferral(job)

    def advance_qualification_deferral(self, job: dict):
        intent = job.get("qualification_deferral_intent")
        if intent is None or not self.submit_enabled:
            return
        observed = self.slurm.get_job(job["job_id"])
        if observed.get("JobName") != job["name"] or observed.get("Account") != self.slurm.account:
            raise ValueError("Optional qualification job identity changed")
        state = observed["JobState"]
        if state == "PENDING":
            if not self.slurm.cancel_pending(job["job_id"], expected_name=job["name"]):
                self.event(f"Qualification {job['job_id']} not confirmed pending-canceled; preserving its intent")
                return
        elif state != "CANCELLED":
            job.pop("qualification_deferral_intent")
            job["status"] = state
            self.event(f"Qualification {job['job_id']} became {state}; withdrawing deferral")
            return
        self.state["deferred_qualifications"][job["profile"]] = {**intent, "job_id": job["job_id"]}
        job["status"], job["handled"] = "CANCELLED", True
        job["qualification_deferred"] = True
        job.pop("qualification_deferral_intent")
        self.event(f"Optional profile {job['profile']} deferred; continuing with qualified hardware")

    def storage_ready(self, profile: str) -> bool:
        try:
            storage.validate(
                pathlib.Path(self.state["qualified"][profile]),
                self.exp_name,
                output_path=self.run_dir / f"storage_preflight_{profile}.json",
            )
        except ValueError as error:
            # The validator records a failed forecast before raising this specific shortfall.
            # Parsing, identity, and other validation failures still require diagnosis.
            if not str(error).startswith("Scratch quota has "):
                raise
            self.event(f"Waiting for scratch headroom before training submission: {error}")
            return False
        return True

    def advance_replacement(self, job: dict):
        intent = job.get("replacement_intent")
        if intent is None or not self.submit_enabled:
            return
        if any(candidate.get("previous_name") == job["name"] for candidate in self.state["jobs"]):
            job.pop("replacement_intent")
            job["status"], job["handled"] = "REPLACED", True
            self.persist()
            return
        observed = self.slurm.get_job(job["job_id"])
        if observed.get("JobName") != job["name"] or observed.get("Account") != self.slurm.account:
            raise ValueError("Pending replacement job identity changed")
        state = observed["JobState"]
        if state == "PENDING":
            if not self.slurm.cancel_pending(job["job_id"], expected_name=job["name"]):
                self.event(f"Job {job['job_id']} not confirmed pending-canceled; replacement remains unsubmitted")
                return
        elif state != "CANCELLED":
            job.pop("replacement_intent")
            job["status"] = state
            self.event(f"Job {job['job_id']} became {state}; withdrawing queue replacement")
            return
        job["status"], job["handled"] = "REPLACED", True
        if intent["time"] not in job["replacement_times"]:
            job["replacement_times"].append(intent["time"])
        self.persist()
        # Queue replacement does not consume another runtime allocation.
        prior = {**job, "allocations": job["allocations"] - 1}
        self.submit_job("train", job["task"], intent["profile"], previous=prior, preemptible=intent["preemptible"])
        job.pop("replacement_intent")
        self.persist()

    def tick(self):
        self.adopt_preparation()
        self.refresh()
        if self.state["attention"]:
            return
        for job in list(self.state["jobs"]):
            self.advance_qualification_deferral(job)
            self.advance_replacement(job)
            self.handle_terminal(job)
            if self.state["attention"]:
                return
        if self.active():
            for job in list(self.active()):
                self.defer_optional_qualification(job)
                self.replace_long_wait(job)
        if self.active():
            self.persist()
            return
        from examples.hanoi.data import dataset
        from examples.hanoi.training import qualify

        qualify.require_validated_data()
        for path in self.state["qualified"].values():
            if json.loads(pathlib.Path(path).read_text())["training_code_sha256"] != dataset.training_code_identity():
                self.attention("Training code changed since qualification; requalify before continuing")
                return
        candidates = [
            profile
            for profile in PROFILES
            if profile not in self.state["qualified"]
            and profile not in self.state["unsupported"]
            and profile not in self.state["deferred_qualifications"]
        ]
        # Qualify up to two profiles if the additional candidate has a reasonable ETA.
        if candidates and len(self.state["qualified"]) < 2 and not self.state["deferred_qualifications"]:
            estimates = [(self.probe(profile, kind="qualify", task=profile), profile) for profile in candidates]
            known = sorted((eta, profile) for eta, profile in estimates if eta is not None)
            eta, profile = known[0] if known else (None, candidates[0])
            if not self.state["qualified"] or (eta is not None and eta <= self.now() + 7200):
                self.submit_job("qualify", profile, profile)
                return
        if not self.state["qualified"]:
            self.attention("No GPU profile passed full-model qualification")
            return
        for task in TASKS:
            status = self.state["models"][task]
            if status == "completed":
                continue
            if status == "trained":
                source_job = next(
                    job
                    for job in reversed(self.state["jobs"])
                    if job["kind"] == "train" and job["task"] == task and job["status"] == "COMPLETED"
                )
                self.submit_job("evaluate", task, source_job["profile"])
                return
            options = []
            for profile in self.state["qualified"]:
                estimate = self.probe(profile, kind="train", task=task)
                options.append(
                    (float("inf") if estimate is None else estimate + self.remaining_seconds(profile, task), profile)
                )
            profile = min(options)[1]
            if self.storage_ready(profile):
                self.submit_job("train", task, profile)
            return
        self.state["complete"] = True
        self.event("All three training and offline evaluation stages completed; final delivery audit remains")


def main(exp_name: str = "hanoi_20260914", *, submit: bool = False, watch: bool = False):
    logging.basicConfig(level=logging.INFO, force=True)
    root = pathlib.Path.cwd()
    manager = Manager(root, exp_name, submit=submit)
    with filelock.FileLock(str(manager.run_dir / "manager.lock"), timeout=0):
        while True:
            try:
                manager.tick()
            except (subprocess.TimeoutExpired, scheduling.ObservationError) as error:
                # Unknown scheduler observations are retried; they do not make a job terminal.
                manager.event(f"Observation/action error; preserving current jobs: {type(error).__name__}: {error}")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                manager.attention(f"Pipeline check failed: {type(error).__name__}: {error}")
            if manager.state["attention"]:
                raise RuntimeError(manager.state["attention"])
            if not watch or manager.state["complete"]:
                return
            for _ in range(10):
                time.sleep(60)


if __name__ == "__main__":
    tyro.cli(main)
