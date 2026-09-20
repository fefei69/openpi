"""Produce the audited delivery after all three managed models finish evaluation."""

import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import socket
import subprocess
import sys
import time

import filelock
import tyro

from examples.hanoi.pipeline import manage


def ready(manager: dict, exp_name: str) -> bool:
    if manager["exp_name"] != exp_name or set(manager["models"]) != set(manage.TASKS):
        raise ValueError("Finalization requires this experiment's three-model manager manifest")
    if manager["attention"] or not manager["complete"]:
        return False
    if any(manager["models"][task] != "completed" for task in manage.TASKS):
        raise ValueError("Manager completion does not cover all three evaluated models")
    if any(not job["handled"] for job in manager["jobs"]):
        raise ValueError("Manager completion still has an unresolved job")
    return True


def write_status(run: pathlib.Path, **values) -> None:
    manage.write_state(
        run / "finalizer.json",
        {
            "updated_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **values,
        },
    )


def tick(root: pathlib.Path, exp_name: str, *, runner=subprocess.run) -> bool:
    run = root / "data/hanoi/runs" / exp_name
    manager = json.loads((run / "manager.json").read_text())
    if not ready(manager, exp_name):
        write_status(
            run,
            state="waiting_for_models",
            models=manager["models"],
            manager_attention=manager["attention"],
            manager_updated_utc=manager["updated_utc"],
        )
        return False
    destination = run / "delivery"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Delivery already exists; inspect its evidence instead of overwriting: {destination}")
    command = [sys.executable, "-m", "examples.hanoi.pipeline.deliver", "--exp-name", exp_name]
    write_status(run, state="packaging", command=command)
    # Import the training stack only in the child, once the manager has finished.
    # The existing delivery command owns its locks and audits every model/result.
    completed = runner(
        command,
        cwd=root,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
        timeout=3600,
    )
    path = destination / "delivery.json"
    delivery = json.loads(path.read_text())
    expected = {f"pi05_hanoi_{task}" for task in manage.TASKS}
    if (
        delivery["exp_name"] != exp_name
        or len(delivery["models"]) != len(expected)
        or {model["config"] for model in delivery["models"]} != expected
        or any(model["completed_optimizer_steps"] != 30000 for model in delivery["models"])
    ):
        raise ValueError("Delivery output does not contain all three full training runs")
    archive_sha = hashlib.sha256((destination / "source.tar.gz").read_bytes()).hexdigest()
    if archive_sha != delivery["source_archive"]["sha256"]:
        raise ValueError("Delivery source archive differs from its recorded hash")
    for name in ("README.md", "source.sha256", "checkpoints.sha256"):
        if not (destination / name).is_file():
            raise ValueError(f"Delivery output is missing {name}")
    write_status(
        run,
        state="ready_for_review",
        delivery=str(destination),
        delivery_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    return True


def main(exp_name: str = "hanoi_20260914", *, watch: bool = False):
    logging.basicConfig(level=logging.INFO, force=True)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", exp_name):
        raise ValueError("Experiment name must be a simple directory component")
    root = pathlib.Path.cwd()
    run = root / "data/hanoi/runs" / exp_name
    with filelock.FileLock(str(run / "finalizer.lock"), timeout=0):
        while True:
            try:
                finished = tick(root, exp_name)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                details = {"state": "attention", "error": f"{type(error).__name__}: {error}"}
                if isinstance(error, subprocess.CalledProcessError | subprocess.TimeoutExpired):
                    for name in ("stdout", "stderr"):
                        output = getattr(error, name)
                        details[name] = output.decode(errors="replace") if isinstance(output, bytes) else output
                write_status(run, **details)
                raise
            if finished or not watch:
                return
            time.sleep(600)


if __name__ == "__main__":
    tyro.cli(main)
