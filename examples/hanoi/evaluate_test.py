import dataclasses
import json
import pathlib

import pytest

from examples.hanoi import evaluate
from examples.hanoi import evaluation
from openpi.training import config as training_config


@dataclasses.dataclass
class FakeDataConfig:
    repo_id: str = "local/hanoi"


@dataclasses.dataclass
class FakeConfig:
    checkpoint_dir: pathlib.Path
    name: str = "pi05_hanoi_multitask"
    exp_name: str = "unit"
    num_train_steps: int = 30000
    data: FakeDataConfig = dataclasses.field(default_factory=FakeDataConfig)


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = FakeConfig(tmp_path / "checkpoints/pi05_hanoi_multitask/unit")
    norm_bytes = '{"state": {"mean": [0], "std": [1]}}\n'
    for step in (5000, 10000, 29999):
        snapshot = config.checkpoint_dir / "exports" / str(step)
        (snapshot / "params").mkdir(parents=True)
        (snapshot / "params/weights").write_bytes(b"parameters")
        norm = snapshot / "assets/local/hanoi/norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text(norm_bytes)
        files = {str(path.relative_to(snapshot)): path.stat().st_size for path in snapshot.rglob("*") if path.is_file()}
        (snapshot / "export.json").write_text(json.dumps({"step": step, "files": files}))
    conversion = tmp_path / "data/hanoi/conversion.json"
    conversion.parent.mkdir(parents=True)
    conversion.write_text("{}")
    (tmp_path / "uv.lock").write_text("locked")
    identity = {
        "config": config.name,
        "exp_name": config.exp_name,
        "training_code_sha256": "unchanged",
        "conversion_sha256": evaluate.dataset.sha256(conversion),
        "lockfile_sha256": evaluate.dataset.sha256(tmp_path / "uv.lock"),
        "norm_stats_sha256": evaluate.dataset.sha256(norm),
    }
    (config.checkpoint_dir / "hanoi_identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(evaluate.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(evaluate.qualify, "require_validated_data", dict)
    monkeypatch.setattr(training_config, "get_config", lambda _: config)
    monkeypatch.setattr(evaluate.dataset, "training_code_identity", lambda: "unchanged")
    calls = []

    def metrics(config, snapshot, split, manifest):
        calls.append((int(snapshot.name), split))
        losses = {5000: 0.8, 10000: 0.4, 29999: 0.7}
        return {"flow_loss": losses[int(snapshot.name)], "physical": {"all": {"first_xyz_mm": 2}}}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", metrics)

    def serving_check(config, snapshot, manifest, output_path):
        result = {"passed": True}
        evaluate.dataset.write_json(output_path, result)
        return result

    monkeypatch.setattr(evaluate.verify_serving, "verify", serving_check)
    return config, calls


def test_selection_uses_validation_then_test_once_and_recovers_interrupted_cleanup(prepared_run, monkeypatch):
    config, calls = prepared_run
    original = evaluation.compact_exports

    def interrupted(*args):
        raise OSError("simulated interruption after the completed result was committed")

    monkeypatch.setattr(evaluation, "compact_exports", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        evaluate.main(config.name, config.exp_name)
    assert calls == [(5000, "val"), (10000, "val"), (29999, "val"), (10000, "test")]
    monkeypatch.setattr(evaluation, "compact_exports", original)
    evaluate.main(config.name, config.exp_name)
    assert len(calls) == 4
    assert [path.name for path in (config.checkpoint_dir / "exports").iterdir()] == ["10000"]
    evaluate.main(config.name, config.exp_name)
    assert len(calls) == 4


def test_completed_result_tampering_is_detected_before_any_test_rerun(prepared_run):
    config, calls = prepared_run
    evaluate.main(config.name, config.exp_name)
    path = pathlib.Path("data/hanoi/runs/unit/pi05_hanoi_multitask/test.json")
    path.write_text("{}")
    with pytest.raises(ValueError, match="completed evaluation result changed"):
        evaluate.main(config.name, config.exp_name)
    assert len(calls) == 4


def test_missing_snapshot_shard_prevents_checkpoint_selection(prepared_run):
    config, calls = prepared_run
    (config.checkpoint_dir / "exports/5000/params/weights").unlink()
    with pytest.raises(ValueError, match="Snapshot file missing"):
        evaluate.main(config.name, config.exp_name)
    assert not calls


def test_failed_serving_parity_prevents_completion_and_preserves_candidates(prepared_run, monkeypatch):
    config, _ = prepared_run
    monkeypatch.setattr(evaluate.verify_serving, "verify", lambda *args: {"passed": False})
    with pytest.raises(ValueError, match="failed serving parity"):
        evaluate.main(config.name, config.exp_name)
    assert not pathlib.Path("data/hanoi/runs/unit/pi05_hanoi_multitask/complete.json").exists()
    assert len(list((config.checkpoint_dir / "exports").iterdir())) == 3
