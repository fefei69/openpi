"""Quota-aware retention forecast for three serial train/evaluate runs."""

import datetime
import json
import pathlib
import re
import subprocess

import filelock
import tyro


def pipeline_lock(exp_name: str) -> filelock.FileLock:
    """Serialize the GPU stages assumed by the quota forecast, including direct launches."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", exp_name):
        raise ValueError("Experiment name must be a simple directory component")
    directory = pathlib.Path("data/hanoi/runs") / exp_name
    directory.mkdir(parents=True, exist_ok=True)
    return filelock.FileLock(str(directory / "gpu_stage.lock"), timeout=0)


def logical_bytes(path: pathlib.Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def unique_bytes(paths: list[pathlib.Path]) -> int:
    seen = set()
    total = 0
    for path in paths:
        if not path.exists():
            continue
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            stat = item.stat()
            identity = stat.st_dev, stat.st_ino
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
    return total


def scratch_headroom(output: str) -> int:
    """Conservatively interpret the rounded myquota display, never filesystem-wide free space."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    match = re.search(r"^/scratch\s+\S+\s+\S+\s+([\d.]+)([KMGT]B)/\S+\s+([\d.]+)([KMGT]B)\(", clean, re.MULTILINE)
    if match is None:
        raise ValueError("Could not parse scratch user quota; production must wait for an authoritative quota reading")
    factors = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    allocation = float(match[1]) * factors[match[2]]
    usage = float(match[3]) * factors[match[4]]
    # A full display increment covers rounding uncertainty. Decimal units also
    # understate positive headroom if the cluster display uses binary units.
    precision = len(match[3].partition(".")[2])
    rounding = factors[match[4]] * 10 ** (-precision)
    return max(0, int(allocation - usage - rounding))


def forecast(*, full_bytes: int, inference_bytes: int, occupied_bytes: int = 0) -> int:
    if full_bytes <= 0 or inference_bytes <= 0 or inference_bytes > full_bytes:
        raise ValueError("Pilot checkpoint sizes must be positive and internally consistent")
    # Serial training/evaluation: three final resumable states plus one in-flight
    # save; at most two selected earlier models and six current validation candidates.
    peak = 4 * full_bytes + 8 * inference_bytes
    reserve = 50 * 2**30  # compilation caches, logs, metadata, and unrelated quota drift
    return max(0, peak - occupied_bytes) + reserve


def validate(qualification_path: pathlib.Path, exp_name: str, *, output_path: pathlib.Path) -> dict:
    from examples.hanoi import dataset

    qualification = json.loads(qualification_path.read_text())
    if not qualification["qualified"]:
        raise ValueError("Storage sizing requires a completed real GPU pilot")
    reading = subprocess.run(["myquota"], capture_output=True, text=True, check=True, timeout=45).stdout
    available = scratch_headroom(reading)
    runs = [pathlib.Path("checkpoints") / f"pi05_hanoi_{task}" / exp_name for task in dataset.TASKS]
    occupied = unique_bytes(runs)
    required = forecast(
        full_bytes=qualification["full_checkpoint_bytes"],
        inference_bytes=qualification["inference_checkpoint_bytes"],
        occupied_bytes=occupied,
    )
    result = {
        "passed": available >= required,
        "checked_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "available_bytes_lower_bound": available,
        "additional_peak_bytes_with_reserve": required,
        "occupied_pipeline_bytes": occupied,
        "qualification_sha256": dataset.sha256(qualification_path),
        "exp_name": exp_name,
        "max_concurrent_training_runs": 1,
        "full_states_per_model": 1,
        "ema_export_interval": 5000,
        "compact_after_evaluation": True,
    }
    dataset.write_json(output_path, result)
    if not result["passed"]:
        raise ValueError(
            f"Scratch quota has {available / 1e9:.1f} GB available; forecast needs {required / 1e9:.1f} GB"
        )
    return result


if __name__ == "__main__":
    tyro.cli(validate)
