"""Read-only validation and episode indexing for the recorded Hanoi roundtrip."""

import contextlib
import hashlib
import itertools
import json
import pathlib

import h5py
import numpy as np

from openpi.policies import hanoi_policy

STEM = "hanoi_wm_roundtrip_20260910_195558"
DIRECTIONS = ("aaaa_to_cccc", "cccc_to_aaaa")
SPLITS = ("train", "val", "test")
TASKS = (*DIRECTIONS, "multitask")


def source_path(data_dir: pathlib.Path, direction: str) -> pathlib.Path:
    start, goal = direction.upper().split("_TO_")
    return data_dir / f"{STEM}_{start}_to_{goal}.h5"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def training_code_identity() -> str:
    """Hash the model/training implementation; documentation and monitoring edits are independent."""
    paths = [path for path in pathlib.Path("src/openpi").rglob("*.py") if not path.name.endswith("_test.py")]
    paths.extend(
        pathlib.Path(path)
        for path in (
            "scripts/train.py",
            "examples/hanoi/training/train.py",
            "examples/hanoi/training/qualify.py",
            "examples/hanoi/data/dataset.py",
            "examples/hanoi/data/compute_norm_stats.py",
            "examples/hanoi/deployment/execution.py",
            "examples/hanoi/pipeline/storage.py",
            "examples/hanoi/pipeline/telemetry.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def split_for_pair(pair: int) -> str:
    if not 0 <= pair < 50:
        raise ValueError(f"Unexpected episode pair {pair}; expected the 50-pair recording")
    return "train" if pair < 40 else "val" if pair < 45 else "test"


def eligible_anchors(handle: h5py.File, start: int, stop: int) -> np.ndarray:
    age_ns = handle["command_monotonic_ns"][start:stop] - handle["image_receipt_monotonic_ns"][start:stop]
    return (age_ns >= 0) & (age_ns <= 50_000_000)


def action_chunk(actions: np.ndarray, anchor: int, horizon: int = 63) -> np.ndarray:
    """Next-reference labels with native terminal hold padding, never a label shift."""
    return actions[np.minimum(np.arange(anchor, anchor + horizon), len(actions) - 1)].copy()


def legal_route(boards: np.ndarray) -> bool:
    for before, after in itertools.pairwise(boards):
        changed = np.flatnonzero(before != after)
        if len(changed) != 1:
            return False
        disk = changed[0]
        if np.any(before[:disk] == before[disk]) or np.any(before[:disk] == after[disk]):
            return False
    return bool(np.isin(boards, (0, 1, 2)).all())


def audit(data_dir: pathlib.Path, *, hash_files: bool = True) -> dict:
    """Fail on structural/label corruption; record and filter stale observation anchors."""
    episodes = []
    sources = []
    motion_seeds = set()
    max_endpoint_error = 0.0
    total_stale = 0
    global_start = 0
    with contextlib.ExitStack() as stack:
        handles = {}
        manifests = {}
        for direction in DIRECTIONS:
            path = source_path(data_dir, direction)
            handle = stack.enter_context(h5py.File(path, "r"))
            handles[direction] = handle
            manifests[direction] = json.loads(path.with_suffix(".json").read_text())
            if handle.attrs["action_abs_alignment"] != "post_action_reference":
                raise ValueError(f"Unsupported action alignment in {path}")
            if handle.attrs["action_profile"] != "segmented_quintic_velocity_noise_v1":
                raise ValueError(f"Unsupported action profile in {path}")
            if float(handle.attrs["rate_hz"]) != 30 or bool(handle.attrs["dry_run"]):
                raise ValueError(f"Expected a real 30 Hz recording: {path}")
            if handle["pixels"].shape != (360050, 224, 224, 3) or handle["pixels"].dtype != np.uint8:
                raise ValueError(f"Unexpected image dimensions/type in {path}")
            if not np.array_equal(handle["ep_len"][:], np.full(50, 7201)):
                raise ValueError(f"Expected 50 complete 7201-row episodes in {path}")
            if not np.array_equal(handle["ep_offset"][:], np.arange(50) * 7201):
                raise ValueError(f"Noncontiguous episode offsets in {path}")
            if len(manifests[direction]["episodes"]) != 50:
                raise ValueError(f"Manifest episode count disagrees in {path}")
            sources.append(
                {
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "sha256": sha256(path) if hash_files else None,
                    "json_sha256": sha256(path.with_suffix(".json")),
                }
            )
        for pair in range(50):
            for direction in DIRECTIONS:
                handle = handles[direction]
                start, stop = pair * 7201, (pair + 1) * 7201
                prefix = f"{direction} episode {pair}"
                arrays = {key: handle[key][start:stop] for key in ("action", "action_abs", "proprio", "state")}
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError(f"{prefix}: nonfinite numeric values")
                actions, reference, proprio = arrays["action"], arrays["action_abs"], arrays["proprio"]
                np.testing.assert_array_equal(arrays["state"][:, :3], proprio[:, :3], err_msg=prefix)
                np.testing.assert_allclose(
                    np.diff(reference[:, :3], axis=0),
                    actions[1:, :3],
                    atol=4e-8,
                    rtol=0,
                    err_msg=f"{prefix}: reference delta alignment",
                )
                if not np.isin(reference[:, 3], (0, 1)).all():
                    raise ValueError(f"{prefix}: invalid jaw intent")
                np.testing.assert_array_equal(actions[:, 3], reference[:, 3], err_msg=prefix)
                np.testing.assert_allclose(proprio[:, 7], 0.034 * reference[:, 3], atol=1e-8, rtol=0, err_msg=prefix)
                np.testing.assert_array_equal(handle["step_idx"][start:stop], np.arange(7201), err_msg=prefix)
                np.testing.assert_array_equal(handle["episode_idx"][start:stop], np.full(7201, pair), err_msg=prefix)
                if not np.all(handle["episode_success"][start:stop] == 1):
                    raise ValueError(f"{prefix}: unsuccessful episode")
                if np.any(np.diff(handle["command_monotonic_ns"][start:stop]) <= 0):
                    raise ValueError(f"{prefix}: command timestamps are not increasing")
                if np.any(np.diff(handle["image_receipt_monotonic_ns"][start:stop]) < 0):
                    raise ValueError(f"{prefix}: image receipt clock reversed")
                boards = handle["board"][start:stop]
                route = boards[np.r_[True, np.any(np.diff(boards, axis=0) != 0, axis=1)]]
                meta = manifests[direction]["episodes"][pair]
                route_text = ["".join("ABC"[int(peg)] for peg in board) for board in route]
                if len(route) != 16 or not legal_route(route) or route_text != meta["route"] or not meta["success"]:
                    raise ValueError(f"{prefix}: illegal or inconsistent route")
                expected_start, expected_goal = direction.upper().split("_TO_")
                if route_text[0] != expected_start or route_text[-1] != expected_goal:
                    raise ValueError(f"{prefix}: wrong direction")
                if meta["motion_attempt_seed"] in motion_seeds:
                    raise ValueError(f"{prefix}: reused motion seed")
                motion_seeds.add(meta["motion_attempt_seed"])
                bounds = np.array([93, 156, 261, 324, 387, 480])[None, :] + 480 * np.arange(15)[:, None]
                error = np.linalg.norm(proprio[bounds, :3] - reference[bounds - 1, :3], axis=-1).max()
                max_endpoint_error = max(max_endpoint_error, float(error))
                if error > 0.0012:
                    raise ValueError(f"{prefix}: endpoint tracking error {error * 1000:.3f} mm exceeds audited bound")
                eligible = int(eligible_anchors(handle, start, stop).sum())
                total_stale += 7201 - eligible
                episodes.append(
                    {
                        "episode_index": len(episodes),
                        "pair": pair,
                        "direction": direction,
                        "split": split_for_pair(pair),
                        "source_start": start,
                        "global_start": global_start,
                        "length": 7201,
                        "eligible": eligible,
                        "motion_seed": meta["motion_attempt_seed"],
                    }
                )
                global_start += 7201
    counts = {split: sum(ep["eligible"] for ep in episodes if ep["split"] == split) for split in SPLITS}
    if counts != {"train": 564714, "val": 70563, "test": 70615}:
        raise ValueError(f"Freshness selection differs from the read-only audit: {counts}")
    return {
        "contract": hanoi_policy.CONTRACT,
        "repo_id": hanoi_policy.REPO_ID,
        "sources": sources,
        "episodes": episodes,
        "rows": global_start,
        "eligible_counts": counts,
        "excluded_stale_anchors": total_stale,
        "max_endpoint_error_m": max_endpoint_error,
    }


def write_indices(manifest: dict, data_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    selections = {f"{task}_{split}": [] for task in TASKS for split in SPLITS}
    with contextlib.ExitStack() as stack:
        handles = {d: stack.enter_context(h5py.File(source_path(data_dir, d), "r")) for d in DIRECTIONS}
        for episode in manifest["episodes"]:
            start = episode["source_start"]
            mask = eligible_anchors(handles[episode["direction"]], start, start + episode["length"])
            indices = np.flatnonzero(mask) + episode["global_start"]
            for task in (episode["direction"], "multitask"):
                selections[f"{task}_{episode['split']}"].append(indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, chunks in selections.items():
        np.save(output_dir / f"{key}.npy", np.concatenate(chunks).astype(np.int64), allow_pickle=False)
