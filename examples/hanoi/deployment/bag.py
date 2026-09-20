"""Optional ROS 2 bag recording of the camera stream alongside a deployment run.

``ros2 bag record`` runs as a separate process for the whole run, from robot initialization to the
return home, so the bag holds the full-frame 30 Hz camera stream the model's 224 x 224 crops were
taken from. Raw 640 x 480 rgb8 at 30 Hz is about 28 MB/s (8 GB for a five-minute run); the storage
presets trade CPU for size. The client's control loop is not involved in the recording.
"""

import dataclasses
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

STORAGE_PRESETS = ("none", "fastwrite", "zstd_fast", "zstd_small")


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


@dataclasses.dataclass
class BagRecorder:
    process: subprocess.Popen
    directory: Path
    log: Path
    topics: tuple
    started_at: float

    def stop(self, timeout_s: float = 20.0) -> dict:
        """Interrupt the recorder so it closes the bag cleanly; escalate only if it does not exit."""
        outcome = "clean"
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                outcome = "terminated"
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    outcome = "killed"
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        else:
            outcome = f"exited early with code {self.process.returncode}"
        return {
            "directory": str(self.directory),
            "topics": list(self.topics),
            "duration_s": time.monotonic() - self.started_at,
            "size_bytes": directory_size(self.directory),
            "outcome": outcome,
            "returncode": self.process.returncode,
        }


def start_bag(directory: Path, topics, *, storage_preset: str = "none", settle_s: float = 1.5) -> BagRecorder:
    """Start ``ros2 bag record`` into ``directory`` (which must not exist yet) and confirm it is running."""
    if storage_preset not in STORAGE_PRESETS:
        raise ValueError(f"Storage preset must be one of {STORAGE_PRESETS}")
    topics = tuple(topics)
    if not topics:
        raise ValueError("At least one topic is required for bag recording")
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("ros2 is not on PATH; run through the launcher script so the ROS environment is sourced")
    if directory.exists():
        raise FileExistsError(f"Bag directory already exists: {directory}")
    command = [ros2, "bag", "record", "-o", str(directory), "-s", "mcap"]
    if storage_preset != "none":
        command += ["--storage-preset-profile", storage_preset]
    command += list(topics)
    log = directory.with_name(directory.name + ".log")
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    started = time.monotonic()
    time.sleep(settle_s)
    if process.poll() is not None:
        tail = log.read_text()[-800:]
        raise RuntimeError(f"ros2 bag record exited immediately with code {process.returncode}:\n{tail}")
    logging.info("Recording %s to %s (pid %d)", ", ".join(topics), directory, process.pid)
    return BagRecorder(process, directory, log, topics, started)
