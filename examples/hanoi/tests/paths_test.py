from pathlib import Path

import pytest

from examples.hanoi.deployment import paths


def test_run_dirs_finds_runs_at_any_depth(tmp_path):
    for rel in ("cosmos_play/dense_live_1", "pi05_play/dense_live_2", "cosmos_waypoint/cosmos_live_3", "pi05_play/dense_replay_4"):
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "summary.json").write_text("{}")
    (tmp_path / "pi05_play/dense_live_5_no_summary").mkdir()
    assert [p.relative_to(tmp_path).as_posix() for p in paths.run_dirs(tmp_path)] == [
        "cosmos_play/dense_live_1", "cosmos_waypoint/cosmos_live_3", "pi05_play/dense_live_2", "pi05_play/dense_replay_4"]


def test_require_available_refuses_the_root_while_the_drive_is_absent(tmp_path):
    root = tmp_path / "drive" / "q-wm_exp"
    with pytest.raises(FileNotFoundError, match="mount the experiment drive"):
        paths.require_available(root / "runs", root)
    assert paths.require_available(tmp_path / "elsewhere", root) == tmp_path / "elsewhere"  # not under the root: no check
    root.mkdir(parents=True)
    assert paths.require_available(root / "runs", root) == root / "runs"
    assert paths.RUNS_DIR == paths.EXPERIMENT_ROOT / "runs" and paths.EXPERIMENT_ROOT.name == "q-wm_exp"
