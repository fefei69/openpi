import json

import pytest

from examples.hanoi.pipeline import wandb_logger


def job(task, job_id="123"):
    return {
        "kind": "train",
        "task": task,
        "job_id": job_id,
        "name": f"hanoi-train-{task}-0123456789ab",
        "status": "COMPLETED",
    }


class FakeRun:
    def __init__(self):
        self.events = []
        self.metrics = []
        self.config = {}
        self.summary = {}
        self.url = "https://wandb.example/test"
        self.finished = False

    def log(self, event):
        self.events.append(event)

    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))

    def finish(self, **kwargs):
        self.finished = True


def test_only_requested_unsubmitted_models_are_selected():
    manager = {"jobs": [job("aaaa_to_cccc"), job("cccc_to_aaaa"), job("multitask")]}
    assert wandb_logger.selected_jobs(manager, "cccc_to_aaaa") == [job("cccc_to_aaaa")]
    with pytest.raises(ValueError, match="only for reverse"):
        wandb_logger.selected_jobs(manager, "aaaa_to_cccc")
    manager["jobs"][1]["name"] = "../../another-run"
    with pytest.raises(ValueError, match="identity"):
        wandb_logger.selected_jobs(manager, "cccc_to_aaaa")


def test_metric_reader_waits_for_complete_lines(tmp_path):
    path = tmp_path / "training.out"
    path.write_text("startup\nStep 100: grad_norm=0.1, loss=0.02, param_norm=1800\nStep 200: grad_norm=0.1, loss=")
    events = wandb_logger.training_events(path, "123")
    assert len(events) == 1
    assert events[0]["optimizer_step"] == 100
    assert events[0]["train/loss"] == 0.02
    assert events[0]["source_event"] == "123:train:100"


def test_telemetry_uses_only_owned_job_samples(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    row = {
        "job_id": "123",
        "time_utc": "2026-09-14T12:00:00+00:00",
        "checkpoint_step": 125,
        "checkpoint_age_seconds": 5,
        "observed_steps_per_second": 0.5,
        "gpu_uuid_memory_mib_utilization_percent": "GPU-owned, 73609 MiB, 100 %",
    }
    path.write_text(json.dumps({**row, "job_id": "other"}) + "\n" + json.dumps(row) + "\n" + '{"job_id":')
    events = wandb_logger.telemetry_events(path, {"123"})
    assert len(events) == 1
    assert events[0]["gpu/0/memory_mib"] == 73609
    assert events[0]["gpu/0/utilization_percent"] == 100
    assert events[0]["telemetry/checkpoint_step"] == 125


def test_resume_deduplicates_acknowledged_events_and_keeps_rollback_steps():
    run = FakeRun()
    seen = {"123:train:200"}
    events = [
        {"source_event": "123:train:200", "optimizer_step": 200},
        {"source_event": "124:train:100", "optimizer_step": 100},
    ]
    assert wandb_logger.publish(run, events, seen) == 1
    assert run.events == [events[1]]
    assert wandb_logger.publish(run, events, seen) == 0


def test_run_initialization_reuses_id_and_custom_optimizer_axis(tmp_path, monkeypatch):
    run = FakeRun()
    calls = []
    monkeypatch.setattr(wandb_logger, "model_config", lambda task: {"task": task})
    monkeypatch.setattr(wandb_logger.wandb, "init", lambda **kwargs: calls.append(kwargs) or run)
    entry = {"entity": "team", "project": "openpi", "run_id": "stable"}
    assert wandb_logger.initialize_run(tmp_path, "experiment", "multitask", entry) is run
    assert calls[0]["resume"] == "allow"
    assert calls[0]["id"] == "stable"
    assert calls[0]["mode"] == "online"
    assert ("train/*", {"step_metric": "optimizer_step"}) in run.metrics


def test_logger_tracks_both_requested_models_and_completes(tmp_path, monkeypatch):
    run_dir = tmp_path / "data/hanoi/runs/test"
    run_dir.mkdir(parents=True)
    jobs = [job(task, str(index + 1)) for index, task in enumerate(wandb_logger.TASKS)]
    (run_dir / "manager.json").write_text(
        json.dumps({"jobs": jobs, "models": dict.fromkeys(wandb_logger.TASKS, "trained")})
    )
    logs = tmp_path / ".cache/hanoi/logs"
    logs.mkdir(parents=True)
    for value in jobs:
        (logs / f"{value['name']}-{value['job_id']}.out").write_text(
            "Step 100: loss=0.1, grad_norm=0.2, param_norm=3\n"
        )
    created = []

    def initialize(*args):
        value = FakeRun()
        created.append(value)
        return value

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wandb_logger, "initialize_run", initialize)
    monkeypatch.setattr(wandb_logger, "published_events", lambda entry: set())
    monkeypatch.setattr(wandb_logger.signal, "signal", lambda *_: None)
    wandb_logger.main("team", exp_name="test", poll_seconds=10)
    assert len(created) == 2
    assert all(value.finished for value in created)
    assert all(len(value.events) == 1 for value in created)
    status = json.loads((run_dir / "wandb_logger.json").read_text())
    assert status["state"] == "complete"
    assert set(status["runs"]) == set(wandb_logger.TASKS)
