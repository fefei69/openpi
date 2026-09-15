"""Offline metrics for episode-bounded Hanoi reference chunks."""

import dataclasses
import json
import pathlib
import shutil

import numpy as np


def validate_export(path: pathlib.Path) -> dict:
    manifest = json.loads((path / "export.json").read_text())
    if manifest["step"] != int(path.name) or not manifest["files"]:
        raise ValueError(f"Invalid snapshot manifest: {path}")
    for name, size in manifest["files"].items():
        relative = pathlib.Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Snapshot files must stay inside their export directory")
        if not (path / relative).is_file() or (path / relative).stat().st_size != size:
            raise ValueError(f"Snapshot file missing or changed: {path / relative}")
    if not (path / "params").is_dir() or not list((path / "assets").rglob("norm_stats.json")):
        raise ValueError(f"Snapshot has no inference parameters/normalization assets: {path}")
    return manifest


def compact_exports(root: pathlib.Path, selected: pathlib.Path) -> None:
    """Idempotently finish cleanup, including after a completed-result write was interrupted."""
    if selected.resolve().parent != root.resolve() or selected.is_symlink():
        raise ValueError("Selected snapshot must belong to this experiment")
    validate_export(selected)
    for path in root.iterdir():
        if path.name.isdigit() and path.resolve() != selected.resolve():
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Refusing to compact an unexpected snapshot path")
            shutil.rmtree(path)


@dataclasses.dataclass
class Metrics:
    anchors: int = 0
    first_xyz_mm: float = 0.0
    mean_xyz_mm: float = 0.0
    last_xyz_mm: float = 0.0
    confusion: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((2, 2), dtype=np.int64))

    def update(self, predicted: np.ndarray, expected: np.ndarray, valid: np.ndarray) -> None:
        if predicted.shape != expected.shape or predicted.shape[-1] != 4 or valid.shape != expected.shape[:-1]:
            raise ValueError("Physical metrics require matching (batch, horizon, 4) chunks and validity masks")
        if not np.isfinite(predicted).all() or not valid[:, 0].all():
            raise ValueError("Predictions must be finite and every anchor must have a valid first target")
        errors = np.linalg.norm(predicted[..., :3] - expected[..., :3], axis=-1) * 1000
        lengths = valid.sum(axis=1)
        if not np.array_equal(valid, np.arange(valid.shape[1])[None] < lengths[:, None]):
            raise ValueError("Padding may occur only after the valid episode suffix")
        self.anchors += len(predicted)
        self.first_xyz_mm += float(errors[:, 0].sum())
        self.mean_xyz_mm += float(((errors * valid).sum(axis=1) / lengths).sum())
        self.last_xyz_mm += float(errors[np.arange(len(errors)), lengths - 1].sum())
        truth = expected[..., 3][valid].astype(np.int64)
        estimate = (predicted[..., 3][valid] >= 0.5).astype(np.int64)
        self.confusion += np.bincount(truth * 2 + estimate, minlength=4).reshape(2, 2)

    def result(self) -> dict:
        if self.anchors == 0:
            raise ValueError("No physical evaluation anchors were accumulated")
        support = self.confusion.sum(axis=1)
        recalls = np.divide(np.diag(self.confusion), support, out=np.zeros(2), where=support > 0)
        return {
            "anchors": self.anchors,
            "first_xyz_mm": self.first_xyz_mm / self.anchors,
            "mean_valid_xyz_mm": self.mean_xyz_mm / self.anchors,
            "last_valid_xyz_mm": self.last_xyz_mm / self.anchors,
            "jaw_confusion_true_rows_predicted_columns": self.confusion.tolist(),
            "jaw_balanced_accuracy": float(recalls[support > 0].mean()),
            "jaw_class_support": support.tolist(),
        }


def sampled_indices(indices: np.ndarray, episodes: list[dict], limit: int = 64) -> np.ndarray:
    selected = []
    for episode in episodes:
        start = episode["global_start"]
        candidates = indices[(indices >= start) & (indices < start + episode["length"])]
        if len(candidates):
            offsets = np.linspace(0, len(candidates) - 1, min(limit, len(candidates)), dtype=int)
            selected.append(candidates[offsets])
    return np.concatenate(selected) if selected else np.array([], dtype=np.int64)


def pad_batch(batch: dict, batch_size: int) -> tuple[dict, int]:
    import jax

    size = len(batch["state"])
    if size == 0 or size > batch_size:
        raise ValueError("Evaluation batch size is out of bounds")
    return jax.tree.map(
        lambda value: np.concatenate([value, np.repeat(value[-1:], batch_size - size, axis=0)]), batch
    ), size
