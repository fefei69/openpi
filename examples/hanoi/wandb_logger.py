"""Publish the two remaining Hanoi runs' recorded metrics from a CPU allocation.

The qualified trainer keeps writing its original Slurm logs and telemetry. This
independent logger adds online tracking without changing checkpoint identities.
"""

import dataclasses
import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import signal
import socket
import threading

import filelock
import tyro
import wandb

TASKS = ("cccc_to_aaaa", "multitask")


def write_json(path: pathlib.Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def complete_lines(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]


def training_events(path: pathlib.Path, job_id: str) -> list[dict]:
    events = []
    for line in complete_lines(path):
        match = re.fullmatch(r"Step (\d+): ([^\r\n]+)\n", line)
        if match is None:
            continue
        values = {key: float(value) for key, value in (field.strip().split("=") for field in match[2].split(","))}
        if set(values) != {"loss", "grad_norm", "param_norm"}:
            raise ValueError("Unexpected trainer metric fields")
        step = int(match[1])
        events.append(
            {
                "source_event": f"{job_id}:train:{step}",
                "slurm_job_id": job_id,
                "optimizer_step": step,
                **{f"train/{key}": value for key, value in values.items()},
            }
        )
    return events


def telemetry_events(path: pathlib.Path, job_ids: set[str]) -> list[dict]:
    events = []
    for line in complete_lines(path):
        row = json.loads(line)
        if row.get("job_id") not in job_ids:
            continue
        event = {
            "source_event": f"{row['job_id']}:telemetry:{row['time_utc']}",
            "slurm_job_id": row["job_id"],
            "wall_time_unix": datetime.datetime.fromisoformat(row["time_utc"]).timestamp(),
        }
        for key in ("checkpoint_step", "checkpoint_age_seconds", "observed_steps_per_second"):
            if row.get(key) is not None:
                event[f"telemetry/{key}"] = row[key]
        for index, gpu in enumerate(row.get("gpu_uuid_memory_mib_utilization_percent", "").splitlines()):
            _, memory, utilization = (field.strip() for field in gpu.split(","))
            event[f"gpu/{index}/memory_mib"] = float(memory.split()[0])
            event[f"gpu/{index}/utilization_percent"] = float(utilization.split()[0])
        events.append(event)
    return events


def selected_jobs(manager: dict, task: str) -> list[dict]:
    if task not in TASKS:
        raise ValueError("W&B tracking is enabled only for reverse and multitask runs")
    jobs = [job for job in manager["jobs"] if job["kind"] == "train" and job["task"] == task and job.get("job_id")]
    for job in jobs:
        if not job["job_id"].isdigit() or not re.fullmatch(rf"hanoi-train-{task}-[0-9a-f]{{12}}", job["name"]):
            raise ValueError("Unexpected managed training job identity")
    return jobs


def model_config(task: str) -> dict:
    from openpi.training import config as training_config

    config = training_config.get_config(f"pi05_hanoi_{task}")
    return {
        "config_name": config.name,
        "model": dataclasses.asdict(config.model),
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "seed": config.seed,
        "ema_decay": config.ema_decay,
        "optimizer": dataclasses.asdict(config.optimizer),
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "data_repo_id": config.data.repo_id,
        "contract": config.policy_metadata,
        "logging_source": "Slurm stdout and allocated-GPU telemetry",
    }


def initialize_run(root: pathlib.Path, exp_name: str, task: str, entry: dict):
    cache = root / ".cache/hanoi/wandb"
    cache.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=entry["entity"],
        project=entry["project"],
        id=entry["run_id"],
        resume="allow",
        name=f"{exp_name}-{task}",
        group=exp_name,
        job_type="train",
        tags=["hanoi", "pi05", "bc", task],
        config=model_config(task),
        mode="online",
        dir=str(cache),
        settings=wandb.Settings(init_timeout=45, console="off", disable_code=True, x_disable_stats=True),
    )
    # Custom x-axes preserve records when a resumed checkpoint rolls back a few
    # optimizer steps. W&B's internal history step remains monotonically increasing.
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("gpu/*", step_metric="wall_time_unix")
    run.define_metric("telemetry/*", step_metric="wall_time_unix")
    return run


def published_events(entry: dict) -> set[str]:
    # Read acknowledged remote history after restart instead of advancing a local
    # cursor ahead of the SDK's asynchronous upload. Missing events are replayed.
    path = f"{entry['entity']}/{entry['project']}/{entry['run_id']}"
    history = wandb.Api(timeout=30).run(path).scan_history(keys=["source_event"])
    return {row["source_event"] for row in history if row.get("source_event")}


def publish(run, events: list[dict], seen: set[str]) -> int:
    count = 0
    for event in events:
        key = event["source_event"]
        if key not in seen:
            run.log(event)
            seen.add(key)
            count += 1
    return count


def main(entity: str, *, project: str = "openpi", exp_name: str = "hanoi_20260914", poll_seconds: int = 60):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", exp_name) or poll_seconds < 10:
        raise ValueError("Use a simple experiment name and a polling interval of at least ten seconds")
    root = pathlib.Path.cwd().resolve()
    directory = root / "data/hanoi/runs" / exp_name
    if not (directory / "manager.json").is_file():
        raise ValueError("A managed Hanoi experiment is required")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    logging.basicConfig(level=logging.INFO)
    active_run = None
    with filelock.FileLock(str(directory / "wandb_logger.lock"), timeout=0):
        plan_path = directory / "wandb_logging.json"
        entries = {
            task: {
                "entity": entity,
                "project": project,
                "run_id": hashlib.sha256(f"{root}\0{exp_name}\0{task}".encode()).hexdigest()[:16],
            }
            for task in TASKS
        }
        plan = {"exp_name": exp_name, "runs": entries}
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("W&B destination or run identity changed; reconcile before restarting")
        write_json(plan_path, plan)
        status = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "process_start_ticks": pathlib.Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
            "slurm_allocation_id": os.environ.get("SLURM_JOB_ID"),
            "enabled_tasks": list(TASKS),
            "runs": {},
        }

        def heartbeat(task: str, state: str, error: str | None = None):
            status.update(
                task=task, state=state, error=error, updated_utc=datetime.datetime.now(datetime.UTC).isoformat()
            )
            write_json(directory / "wandb_logger.json", status)

        try:
            for task in TASKS:
                entry = entries[task]
                seen = None
                while not stop.is_set():
                    try:
                        manager = json.loads((directory / "manager.json").read_text())
                        jobs = selected_jobs(manager, task)
                        if not jobs:
                            heartbeat(task, "waiting_for_submission")
                            stop.wait(poll_seconds)
                            continue
                        if active_run is None:
                            active_run = initialize_run(root, exp_name, task, entry)
                            status["runs"][task] = {**entry, "url": active_run.url}
                        if seen is None:
                            seen = published_events(entry)
                        events = []
                        for job in jobs:
                            path = root / ".cache/hanoi/logs" / f"{job['name']}-{job['job_id']}.out"
                            events.extend(training_events(path, job["job_id"]))
                        task_dir = directory / f"pi05_hanoi_{task}"
                        events.extend(telemetry_events(task_dir / "telemetry.jsonl", {job["job_id"] for job in jobs}))
                        published = publish(active_run, events, seen)
                        identity = root / "checkpoints" / f"pi05_hanoi_{task}" / exp_name / "hanoi_identity.json"
                        if identity.exists():
                            active_run.config.update({"training_identity": json.loads(identity.read_text())})
                        state = manager["models"][task]
                        active_run.summary.update(
                            {
                                "pipeline_state": state,
                                "latest_job_id": jobs[-1]["job_id"],
                                "latest_job_state": jobs[-1]["status"],
                            }
                        )
                        heartbeat(task, "tracking")
                        if published:
                            logging.info("Published %d new events for %s", published, task)
                        if state in ("trained", "completed"):
                            active_run.summary["training_completed"] = True
                            active_run.finish()
                            active_run = None
                            heartbeat(task, "training_complete")
                            break
                    except (OSError, ValueError, wandb.errors.Error) as error:
                        logging.exception("W&B logger will retry; the independent training job is unaffected")
                        heartbeat(task, "retrying", f"{type(error).__name__}: {error}")
                    stop.wait(poll_seconds)
                if stop.is_set():
                    break
            if not stop.is_set():
                heartbeat("all", "complete")
        finally:
            if active_run is not None:
                active_run.finish(exit_code=1)


if __name__ == "__main__":
    tyro.cli(main)
