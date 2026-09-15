import dataclasses
import json
import pathlib
import shlex
import subprocess
import tarfile

import pytest

from examples.hanoi import dataset
from examples.hanoi import deliver
from examples.hanoi import evaluate
from examples.hanoi import evaluation
from examples.hanoi import verify_serving
from openpi.policies import hanoi_policy
from openpi.training import config as training_config


def physical(anchors):
    return {
        "anchors": anchors,
        "first_xyz_mm": 0.4,
        "mean_valid_xyz_mm": 0.5,
        "last_valid_xyz_mm": 0.6,
        "jaw_balanced_accuracy": 1.0,
        "jaw_confusion_true_rows_predicted_columns": [[anchors, 0], [0, anchors]],
        "jaw_class_support": [anchors, anchors],
    }


def metrics(episodes, directions, split, loss):
    selected = [ep for ep in episodes if ep["direction"] in directions and ep["split"] == split]
    anchors = sum(ep["eligible"] for ep in selected)
    count_by_direction = {
        direction: sum(ep["eligible"] for ep in selected if ep["direction"] == direction) for direction in directions
    }
    result = {
        "split": split,
        "flow_loss": loss,
        "flow_anchors": anchors,
        "flow_by_episode": {str(ep["episode_index"]): {"loss": loss, "anchors": ep["eligible"]} for ep in selected},
        "flow_by_direction": {
            direction: {"loss": loss, "anchors": count} for direction, count in count_by_direction.items()
        },
        "physical": {
            "all": physical(anchors),
            **{direction: physical(count) for direction, count in count_by_direction.items()},
            **{f"episode_{ep['episode_index']}": physical(ep["eligible"]) for ep in selected},
        },
    }
    if len(directions) == 2:
        result["prompt_swap_diagnostic"] = physical(anchors)
    return result


@pytest.fixture(params=dataset.TASKS)
def completed_model(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset, "training_code_identity", lambda: "fixed-code")
    monkeypatch.setattr(deliver, "optimizer_step", lambda _: 30000)
    return make_completed_model(tmp_path, request.param)


def make_completed_model(tmp_path, task):
    config = dataclasses.replace(training_config.get_config(f"pi05_hanoi_{task}"), exp_name="unit")
    directions = dataset.DIRECTIONS if task == "multitask" else (task,)
    episodes = [
        {"episode_index": i * 2 + j, "direction": direction, "split": split, "eligible": 2}
        for i, split in enumerate(["val"] * 5 + ["test"] * 5)
        for j, direction in enumerate(dataset.DIRECTIONS)
    ]
    manifest = {"episodes": episodes}
    dataset.write_json(pathlib.Path("data/hanoi/conversion.json"), manifest)
    pathlib.Path("uv.lock").write_text("fixed lock")
    for name in ("params", "train_state"):
        (config.checkpoint_dir / "29999" / name).mkdir(parents=True)
    dataset.write_json(config.checkpoint_dir / "29999/_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 123})
    snapshot = config.checkpoint_dir / "exports/10000"
    (snapshot / "params").mkdir(parents=True)
    (snapshot / "params/weights").write_bytes(b"small test parameters")
    norm = snapshot / "assets" / config.data.repo_id / "norm_stats.json"
    dataset.write_json(norm, {"norm": 1})
    dataset.write_json(
        snapshot / "export.json",
        {
            "step": 10000,
            "files": {
                str(path.relative_to(snapshot)): path.stat().st_size for path in snapshot.rglob("*") if path.is_file()
            },
        },
    )
    training_path = config.checkpoint_dir / "hanoi_identity.json"
    dataset.write_json(config.checkpoint_dir / "hanoi_placement.json", {"fsdp_devices": 1})
    dataset.write_json(
        training_path,
        {
            "config": config.name,
            "exp_name": config.exp_name,
            "contract": hanoi_policy.CONTRACT,
            "training_code_sha256": "fixed-code",
            "lockfile_sha256": dataset.sha256(pathlib.Path("uv.lock")),
            "conversion_sha256": dataset.sha256(pathlib.Path("data/hanoi/conversion.json")),
            "norm_stats_sha256": dataset.sha256(norm),
            "num_train_steps": 30000,
            "batch_size": 32,
            "seed": 42,
        },
    )
    identity = {
        "config": config.name,
        "exp_name": config.exp_name,
        "conversion_sha256": dataset.sha256(pathlib.Path("data/hanoi/conversion.json")),
        "evaluation_code_sha256": dataset.sha256(pathlib.Path(evaluate.__file__)),
        "metrics_code_sha256": dataset.sha256(pathlib.Path(evaluation.__file__)),
        "serving_code_sha256": dataset.sha256(pathlib.Path(verify_serving.__file__)),
        "training_identity_sha256": dataset.sha256(training_path),
    }
    export_hash = dataset.sha256(snapshot / "export.json")
    selected_identity = {**identity, "export_sha256": export_hash}
    results = tmp_path / "data/hanoi/runs/unit" / config.name
    for step in (5000, 10000, 15000, 20000, 25000, 29999):
        result = {
            "step": step,
            "identity": selected_identity,
            **metrics(episodes, directions, "val", 0.4 if step == 10000 else 0.8),
        }
        dataset.write_json(results / f"validation_{step}.json", result)
    test = {"step": 10000, "identity": selected_identity, **metrics(episodes, directions, "test", 0.5)}
    dataset.write_json(results / "test.json", test)
    dataset.write_json(
        results / "selection.json",
        {
            "identity": identity,
            "step": 10000,
            "criterion": "validation_flow_loss",
            "validation_flow_loss": 0.4,
            "checkpoint": str(snapshot.resolve()),
        },
    )
    dataset.write_json(
        results / "serving_validation.json",
        {
            "identity": {
                "config": config.name,
                "exp_name": config.exp_name,
                "export_sha256": export_hash,
                "conversion_sha256": identity["conversion_sha256"],
                "training_identity_sha256": identity["training_identity_sha256"],
                "code_sha256": identity["serving_code_sha256"],
            },
            "passed": True,
            "hardware_executed": False,
            "probes": [{"direction": direction, "jaw_decisions_equal": True} for direction in directions],
        },
    )
    dataset.write_json(
        results / "complete.json",
        {
            "identity": identity,
            "selected_step": 10000,
            "selected_checkpoint": str(snapshot.resolve()),
            "test_flow_loss": test["flow_loss"],
            "test_physical": test["physical"],
            "serving_parity_passed": True,
            "hardware_success_measured": False,
            "export_sha256": export_hash,
            "result_sha256": {path.name: dataset.sha256(path) for path in results.glob("*.json")},
        },
    )
    return config, manifest, results


def test_delivery_audits_each_model_and_produces_exact_serving_arguments(completed_model):
    config, manifest, results = completed_model
    result = deliver.audit_model(config, manifest, results)
    assert result["selected_step"] == 10000
    assert result["completed_optimizer_steps"] == 30000
    assert shlex.split(result["serve_command"]) == result["serve_arguments"]
    assert ("--default-prompt" in result["serve_arguments"]) == (config.name != "pi05_hanoi_multitask")
    for name, metadata in result["checkpoint_files"].items():
        assert dataset.sha256(pathlib.Path(result["checkpoint"]) / name) == metadata["sha256"]


def test_delivery_refuses_to_treat_a_folder_label_as_completed_training(completed_model, monkeypatch):
    config, manifest, results = completed_model
    monkeypatch.setattr(deliver, "optimizer_step", lambda _: 20000)
    with pytest.raises(ValueError, match="all requested optimizer steps"):
        deliver.audit_model(config, manifest, results)


def test_delivery_recomputes_validation_selection_instead_of_trusting_completion_flag(completed_model):
    config, manifest, results = completed_model
    selection_path = results / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["step"] = 29999
    dataset.write_json(selection_path, selection)
    complete = json.loads((results / "complete.json").read_text())
    complete["result_sha256"]["selection.json"] = dataset.sha256(selection_path)
    dataset.write_json(results / "complete.json", complete)
    with pytest.raises(ValueError, match="validation-only optimum"):
        deliver.audit_model(config, manifest, results)


def test_source_archive_is_reproducible_and_excludes_recordings(tmp_path):
    for path in deliver.source_files(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    source = tmp_path / "examples/hanoi/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('source')\n")
    (source.parent / "recording.h5").write_bytes(b"raw dataset")
    first = deliver.archive_source(tmp_path, tmp_path / "first.tar.gz")
    second = deliver.archive_source(tmp_path, tmp_path / "second.tar.gz")
    assert first == second
    with tarfile.open(tmp_path / "first.tar.gz") as archive:
        assert "examples/hanoi/example.py" in archive.getnames()
        assert "examples/hanoi/recording.h5" not in archive.getnames()
        assert archive.extractfile("examples/hanoi/example.py").read() == source.read_bytes()


def test_delivery_requires_all_three_models_even_if_manager_claims_complete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(deliver.qualify, "require_validated_data", dict)
    run = tmp_path / "data/hanoi/runs/unit"
    dataset.write_json(
        run / "manager.json",
        {
            "complete": True,
            "models": {"aaaa_to_cccc": "completed", "cccc_to_aaaa": "completed", "multitask": "planned"},
        },
    )
    with pytest.raises(ValueError, match="All three training/evaluation stages"):
        deliver.main("unit")
    assert not (run / "delivery").exists()


def test_complete_delivery_produces_portable_checksum_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset, "training_code_identity", lambda: "fixed-code")
    monkeypatch.setattr(deliver, "optimizer_step", lambda _: 30000)
    for task in dataset.TASKS:
        _, manifest, _ = make_completed_model(tmp_path, task)
    monkeypatch.setattr(deliver.qualify, "require_validated_data", lambda: manifest)
    for name in ("audit.json", "data_validation.json", "execution_validation.json", "visual_review.json"):
        dataset.write_json(tmp_path / "data/hanoi" / name, {})
    run = tmp_path / "data/hanoi/runs/unit"
    dataset.write_json(run / "manager.json", {"complete": True, "models": dict.fromkeys(dataset.TASKS, "completed")})
    for path in deliver.source_files(tmp_path):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name)
    deliver.main("unit")
    output = run / "delivery"
    result = json.loads((output / "delivery.json").read_text())
    assert {model["config"] for model in result["models"]} == {f"pi05_hanoi_{task}" for task in dataset.TASKS}
    for model in result["models"]:
        arguments = shlex.split(model["serve_command"])
        assert not pathlib.Path(arguments[arguments.index("--policy.dir") + 1]).is_absolute()
    subprocess.run(["sha256sum", "--check", "source.sha256"], cwd=output, capture_output=True, check=True)
    subprocess.run(["sha256sum", "--check", str(output / "checkpoints.sha256")], capture_output=True, check=True)
    assert not list(run.glob(".delivery-*.partial"))
