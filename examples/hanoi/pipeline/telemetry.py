"""Observe only this process's allocated GPUs and this run's committed checkpoints."""

import contextlib
import datetime
import json
import os
import pathlib
import re
import subprocess
import threading
import time


def process_gpu_uuids(output: str, pid: int, expected_devices: int) -> list[str]:
    devices = set()
    for line in output.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            devices.add(parts[1])
    if len(devices) != expected_devices:
        raise ValueError(f"NVML found {len(devices)} GPUs for this process; JAX allocated {expected_devices}")
    return sorted(devices)


def allocated_devices() -> list[str]:
    import jax

    # Establish a context on each JAX-visible device before querying the process
    # table. CUDA_VISIBLE_DEVICES indices alone may be remapped by Slurm cgroups.
    for device in jax.local_devices():
        jax.device_put(0, device).block_until_ready()
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=30
    )
    return process_gpu_uuids(output, os.getpid(), jax.local_device_count())


def query(devices: list[str], fields: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", f"--id={','.join(devices)}", f"--query-gpu={fields}", "--format=csv,noheader"],
        text=True,
        timeout=30,
    ).strip()


def hardware_info(devices: list[str]) -> str:
    return query(devices, "name,uuid,memory.total,driver_version")


def checkpoint_progress(directory: pathlib.Path) -> dict:
    candidates = []
    for metadata in directory.glob("[0-9]*/_CHECKPOINT_METADATA"):
        if not metadata.parent.name.isdigit():
            continue
        try:
            committed = json.loads(metadata.read_text()).get("commit_timestamp_nsecs")
        except (OSError, json.JSONDecodeError):
            continue
        if committed:
            candidates.append((int(metadata.parent.name), committed / 1e9))
    if not candidates:
        return {"checkpoint_step": None, "checkpoint_age_seconds": None}
    step, committed = max(candidates)
    return {"checkpoint_step": step, "checkpoint_age_seconds": max(0, time.time() - committed)}


@contextlib.contextmanager
def monitor(devices: list[str], directory: pathlib.Path, output_path: pathlib.Path):
    """Record GPU use and available trainer progress once per minute without changing training."""
    stop = threading.Event()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    job_id, job_name = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_JOB_NAME")
    stdout = pathlib.Path(".cache/hanoi/logs") / f"{job_name}-{job_id}.out"
    prior = None

    def record():
        nonlocal prior
        value = {
            "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "job_id": job_id,
            "pid": os.getpid(),
        }
        try:
            value["gpu_uuid_memory_mib_utilization_percent"] = query(devices, "uuid,memory.used,utilization.gpu")
            value.update(checkpoint_progress(directory))
            if stdout.exists():
                with stdout.open("rb") as stream:
                    stream.seek(max(0, stdout.stat().st_size - 100000))
                    text = stream.read().decode(errors="replace")
                matches = re.findall(r"Step (\d+): ([^\n\r]+)", text)
                if matches:
                    step, metrics = matches[-1]
                    value["logged_step"], value["logged_metrics"] = int(step), metrics
                    now = time.monotonic()
                    if prior is not None:
                        value["observed_steps_per_second"] = (int(step) - prior[0]) / (now - prior[1])
                    prior = int(step), now
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            value["observation_error"] = str(error)
        with output_path.open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def worker():
        while not stop.is_set():
            record()
            stop.wait(60)

    thread = threading.Thread(target=worker, name="hanoi-telemetry", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=35)
