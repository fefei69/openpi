"""Where experiment output lives: the external drive, with runs organised by policy family.

Run logs and camera bags are large, so every run goes to ``$HANOI_EXPERIMENT_ROOT/runs/<policy family>/<run>``
(default root ``/media/cpw/T9/chungpang/q-wm_exp``, the T9 drive). The scoreboard, the trial report and the chart
read every run under that folder, at any depth. ``published/`` on the drive holds copies of the two ``exp_vid``
folders. Nothing is created under the root while the drive is absent: a ``mkdir`` there would land on the system
disk and be hidden the next time the drive is mounted.
"""

import os
from pathlib import Path

EXPERIMENT_ROOT = Path(os.environ.get("HANOI_EXPERIMENT_ROOT", "/media/cpw/T9/chungpang/q-wm_exp"))
RUNS_DIR = EXPERIMENT_ROOT / "runs"


def require_available(path: Path, root: Path = EXPERIMENT_ROOT) -> Path:
    """``path``, after checking that the experiment drive is present when ``path`` lies under its root."""
    path = Path(path)
    resolved = path.resolve()
    if (resolved == root.resolve() or root.resolve() in resolved.parents) and not root.is_dir():
        raise FileNotFoundError(f"{root} is missing: mount the experiment drive, or pass --output to write somewhere else")
    return path


def run_dirs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Every run folder under ``runs_dir`` at any depth (a folder holding a summary.json), oldest name first."""
    return sorted(p.parent for p in Path(runs_dir).rglob("summary.json"))
