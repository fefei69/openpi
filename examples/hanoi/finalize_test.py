import hashlib
import json
import subprocess

import filelock
import pytest

from examples.hanoi import finalize
from examples.hanoi import manage


@pytest.fixture
def completed_manager(tmp_path):
    run = tmp_path / "data/hanoi/runs/unit"
    run.mkdir(parents=True)
    state = {
        "exp_name": "unit",
        "updated_utc": "2026-09-15T00:00:00+00:00",
        "attention": None,
        "complete": True,
        "models": dict.fromkeys(manage.TASKS, "completed"),
        "jobs": [{"handled": True}],
    }
    (run / "manager.json").write_text(json.dumps(state))
    return run, state


def write_delivery(run):
    destination = run / "delivery"
    destination.mkdir()
    (destination / "source.tar.gz").write_bytes(b"source archive")
    delivery = {
        "exp_name": "unit",
        "models": [{"config": f"pi05_hanoi_{task}", "completed_optimizer_steps": 30000} for task in manage.TASKS],
        "source_archive": {"sha256": hashlib.sha256(b"source archive").hexdigest()},
    }
    (destination / "delivery.json").write_text(json.dumps(delivery))
    for name in ("README.md", "source.sha256", "checkpoints.sha256"):
        (destination / name).write_text(name)
    return destination


def test_incomplete_models_and_manager_attention_wait_without_packaging(completed_manager, tmp_path):
    run, state = completed_manager
    for update in (
        {"complete": False, "models": dict.fromkeys(manage.TASKS, "planned")},
        {"complete": True, "attention": "Diagnose the failed job"},
    ):
        (run / "manager.json").write_text(json.dumps({**state, **update}))

        def unexpected_runner(*args, **kwargs):
            pytest.fail("Incomplete/held training must never start delivery")

        assert not finalize.tick(tmp_path, "unit", runner=unexpected_runner)
        assert json.loads((run / "finalizer.json").read_text())["state"] == "waiting_for_models"
    assert not (run / "delivery").exists()


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("missing_model", "three-model manager manifest"),
        ("untrained_model", "all three evaluated models"),
        ("unresolved_job", "unresolved job"),
        ("wrong_experiment", "three-model manager manifest"),
    ],
)
def test_completion_flag_cannot_bypass_readiness(completed_manager, invalid, message):
    _, state = completed_manager
    if invalid == "missing_model":
        del state["models"]["multitask"]
    elif invalid == "untrained_model":
        state["models"]["multitask"] = "trained"
    elif invalid == "unresolved_job":
        state["jobs"][0]["handled"] = False
    else:
        state["exp_name"] = "another"
    with pytest.raises(ValueError, match=message):
        finalize.ready(state, "unit")


def test_finalizer_uses_existing_audit_and_preserves_existing_delivery(completed_manager, tmp_path):
    run, _ = completed_manager
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        assert command[1:] == ["-m", "examples.hanoi.deliver", "--exp-name", "unit"]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
        assert kwargs["check"]
        write_delivery(run)
        return subprocess.CompletedProcess(command, 0, "delivery path\n", "")

    assert finalize.tick(tmp_path, "unit", runner=runner)
    assert json.loads((run / "finalizer.json").read_text())["state"] == "ready_for_review"
    before = (run / "delivery/delivery.json").read_bytes()
    with pytest.raises(FileExistsError, match="instead of overwriting"):
        finalize.tick(tmp_path, "unit", runner=runner)
    assert len(calls) == 1
    assert (run / "delivery/delivery.json").read_bytes() == before


def test_success_exit_without_valid_output_is_not_ready(completed_manager, tmp_path):
    run, _ = completed_manager

    def runner(command, **kwargs):
        destination = write_delivery(run)
        (destination / "source.tar.gz").write_bytes(b"corrupted")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(ValueError, match="recorded hash"):
        finalize.tick(tmp_path, "unit", runner=runner)
    assert json.loads((run / "finalizer.json").read_text())["state"] != "ready_for_review"


def test_delivery_failure_is_not_retried_in_the_same_tick(completed_manager, tmp_path):
    run, _ = completed_manager
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, "", "Audit rejected the selected checkpoint")

    with pytest.raises(subprocess.CalledProcessError):
        finalize.tick(tmp_path, "unit", runner=runner)
    assert len(calls) == 1
    assert json.loads((run / "finalizer.json").read_text())["state"] != "ready_for_review"


def test_timeout_records_captured_bytes_and_stops(completed_manager, tmp_path, monkeypatch):
    run, _ = completed_manager
    monkeypatch.chdir(tmp_path)

    def timeout(*args):
        raise subprocess.TimeoutExpired(
            ["python", "deliver"], 3600, output=b"partial output", stderr=b"slow filesystem"
        )

    monkeypatch.setattr(finalize, "tick", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        finalize.main("unit")
    result = json.loads((run / "finalizer.json").read_text())
    assert result["state"] == "attention"
    assert result["stdout"] == "partial output"
    assert result["stderr"] == "slow filesystem"


def test_second_finalizer_cannot_write_another_watchers_status(completed_manager, tmp_path, monkeypatch):
    run, _ = completed_manager
    monkeypatch.chdir(tmp_path)
    (run / "finalizer.json").write_text('{"existing_watcher": true}\n')
    with filelock.FileLock(str(run / "finalizer.lock"), timeout=0), pytest.raises(filelock.Timeout):
        finalize.main("unit")
    assert json.loads((run / "finalizer.json").read_text()) == {"existing_watcher": True}
