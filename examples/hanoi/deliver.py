"""Audit and package the three selected policies without duplicating their weights."""

import dataclasses
import datetime
import gzip
import json
import math
import pathlib
import shlex
import shutil
import tarfile
import uuid

import filelock
import numpy as np
import orbax.checkpoint as ocp
import tyro

from examples.hanoi import dataset
from examples.hanoi import evaluate
from examples.hanoi import evaluation
from examples.hanoi import manage
from examples.hanoi import qualify
from examples.hanoi import storage
from examples.hanoi import verify_serving
from openpi.policies import hanoi_policy
from openpi.training import config as _config


def optimizer_step(checkpoint: pathlib.Path) -> int:
    """Read only the step scalar; omitted parameters/optimizer arrays are not restored."""
    with ocp.PyTreeCheckpointer() as checkpointer:
        value = checkpointer.restore(
            checkpoint / "train_state",
            args=ocp.args.PyTreeRestore(
                item={"step": np.array(0, dtype=np.int32)},
                restore_args={"step": ocp.ArrayRestoreArgs(restore_type=np.ndarray)},
                transforms={},
            ),
        )["step"]
    value = np.asarray(value)
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("The saved optimizer step must be an integer scalar")
    return int(value)


def require_metrics(result: dict, manifest: dict, directions: tuple[str, ...], split: str) -> None:
    episodes = [ep for ep in manifest["episodes"] if ep["direction"] in directions and ep["split"] == split]
    if result["split"] != split or result["flow_anchors"] != sum(ep["eligible"] for ep in episodes):
        raise ValueError("Evaluation does not cover every eligible anchor in its declared split")
    if not math.isfinite(result["flow_loss"]) or result["flow_loss"] < 0:
        raise ValueError("Evaluation flow loss must be finite and nonnegative")
    for field, expected in (
        ("flow_by_episode", {str(ep["episode_index"]): ep["eligible"] for ep in episodes}),
        (
            "flow_by_direction",
            {
                direction: sum(ep["eligible"] for ep in episodes if ep["direction"] == direction)
                for direction in directions
            },
        ),
    ):
        breakdown = result[field]
        if set(breakdown) != set(expected):
            raise ValueError("Flow metrics lack the required episode/direction breakdown")
        for key, count in expected.items():
            if (
                breakdown[key]["anchors"] != count
                or not math.isfinite(breakdown[key]["loss"])
                or breakdown[key]["loss"] < 0
            ):
                raise ValueError("A flow breakdown has an incorrect count or nonfinite loss")
        weighted = sum(value["anchors"] * value["loss"] for value in breakdown.values()) / result["flow_anchors"]
        if not np.isclose(weighted, result["flow_loss"], rtol=1e-7, atol=1e-9):
            raise ValueError("Flow breakdowns disagree with the aggregate loss")
    physical = result["physical"]
    expected_keys = {"all", *directions, *(f"episode_{ep['episode_index']}" for ep in episodes)}
    if set(physical) != expected_keys:
        raise ValueError("Physical metrics lack the required episode/direction breakdown")
    if physical["all"]["anchors"] != sum(min(64, ep["eligible"]) for ep in episodes):
        raise ValueError("Physical metrics do not cover the required sampled anchors")
    if len(directions) == 2 and "prompt_swap_diagnostic" not in result:
        raise ValueError("Multitask evaluation requires the prompt-swap diagnostic")
    metric_values = list(physical.values())
    if len(directions) == 2:
        swapped = result["prompt_swap_diagnostic"]
        if swapped["anchors"] != physical["all"]["anchors"]:
            raise ValueError("Prompt-swap diagnostic used different anchors")
        metric_values.append(swapped)
    for metrics in metric_values:
        for key in ("first_xyz_mm", "mean_valid_xyz_mm", "last_valid_xyz_mm"):
            if not math.isfinite(metrics[key]) or metrics[key] < 0:
                raise ValueError("Physical errors must be finite and nonnegative")
        if not 0 <= metrics["jaw_balanced_accuracy"] <= 1:
            raise ValueError("Jaw balanced accuracy must be between zero and one")
        confusion = np.asarray(metrics["jaw_confusion_true_rows_predicted_columns"])
        if confusion.shape != (2, 2) or not np.issubdtype(confusion.dtype, np.integer) or np.any(confusion < 0):
            raise ValueError("Invalid jaw confusion matrix")
        if not np.array_equal(confusion.sum(axis=1), metrics["jaw_class_support"]):
            raise ValueError("Jaw confusion matrix disagrees with its class support")


def audit_model(config: _config.TrainConfig, manifest: dict, results: pathlib.Path) -> dict:
    task = config.name.removeprefix("pi05_hanoi_")
    directions = dataset.DIRECTIONS if task == "multitask" else (task,)
    complete = json.loads((results / "complete.json").read_text())
    identity = complete["identity"]
    training_path = config.checkpoint_dir / "hanoi_identity.json"
    training = json.loads(training_path.read_text())
    expected_identity = {
        "config": config.name,
        "exp_name": config.exp_name,
        "conversion_sha256": dataset.sha256(pathlib.Path("data/hanoi/conversion.json")),
        "evaluation_code_sha256": dataset.sha256(pathlib.Path(evaluate.__file__)),
        "metrics_code_sha256": dataset.sha256(pathlib.Path(evaluation.__file__)),
        "serving_code_sha256": dataset.sha256(pathlib.Path(verify_serving.__file__)),
        "training_identity_sha256": dataset.sha256(training_path),
    }
    if identity != expected_identity or training["training_code_sha256"] != dataset.training_code_identity():
        raise ValueError("Delivery code/data differs from the training and evaluation identities")
    if training["lockfile_sha256"] != dataset.sha256(pathlib.Path("uv.lock")):
        raise ValueError("Delivery dependencies differ from the trained environment")
    if any(
        training[key] != value
        for key, value in {
            "config": config.name,
            "exp_name": config.exp_name,
            "contract": hanoi_policy.CONTRACT,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "num_train_steps": config.num_train_steps,
            "conversion_sha256": identity["conversion_sha256"],
        }.items()
    ):
        raise ValueError("Training identity differs from the requested run")
    latest = manage.complete_step(config.checkpoint_dir)
    if (
        latest != config.num_train_steps - 1
        or optimizer_step(config.checkpoint_dir / str(latest)) != config.num_train_steps
    ):
        raise ValueError("The model has not completed all requested optimizer steps")
    expected_steps = [
        *range(config.export_params_interval, config.num_train_steps, config.export_params_interval),
        latest,
    ]
    required_results = {
        "selection.json",
        "test.json",
        "serving_validation.json",
        *(f"validation_{step}.json" for step in expected_steps),
    }
    if set(complete["result_sha256"]) != required_results:
        raise ValueError("Delivery is missing a required validation/test/serving result")
    for name, digest in complete["result_sha256"].items():
        if dataset.sha256(results / name) != digest:
            raise ValueError(f"A completed result changed: {name}")
    validation = [json.loads((results / f"validation_{step}.json").read_text()) for step in expected_steps]
    for step, result in zip(expected_steps, validation, strict=True):
        if result["step"] != step or {key: result["identity"][key] for key in identity} != identity:
            raise ValueError("A validation result belongs to a different checkpoint/run")
        require_metrics(result, manifest, directions, "val")
    best = min(validation, key=lambda result: (result["flow_loss"], result["step"]))
    selection = json.loads((results / "selection.json").read_text())
    selected = config.checkpoint_dir / "exports" / str(best["step"])
    if (
        selection["identity"] != identity
        or selection["criterion"] != "validation_flow_loss"
        or selection["step"] != best["step"]
        or selection["validation_flow_loss"] != best["flow_loss"]
        or pathlib.Path(selection["checkpoint"]).resolve() != selected.resolve()
        or pathlib.Path(complete["selected_checkpoint"]).resolve() != selected.resolve()
        or complete["selected_step"] != best["step"]
    ):
        raise ValueError("The selected checkpoint is not the validation-only optimum")
    export = evaluation.validate_export(selected)
    export_hash = dataset.sha256(selected / "export.json")
    if export_hash != complete["export_sha256"] or export_hash != best["identity"]["export_sha256"]:
        raise ValueError("The selected checkpoint differs from its evaluated snapshot")
    norm_path = selected / "assets" / config.data.repo_id / "norm_stats.json"
    if dataset.sha256(norm_path) != training["norm_stats_sha256"]:
        raise ValueError("Selected normalization differs from training")
    test = json.loads((results / "test.json").read_text())
    if test["identity"] != best["identity"] or test["step"] != best["step"]:
        raise ValueError("Test evaluated a different snapshot than validation selected")
    require_metrics(test, manifest, directions, "test")
    if complete["test_flow_loss"] != test["flow_loss"] or complete["test_physical"] != test["physical"]:
        raise ValueError("Completion summary disagrees with the saved test metrics")
    serving = json.loads((results / "serving_validation.json").read_text())
    serving_identity = {
        "config": config.name,
        "exp_name": config.exp_name,
        "export_sha256": export_hash,
        "training_identity_sha256": identity["training_identity_sha256"],
        "conversion_sha256": identity["conversion_sha256"],
        "code_sha256": identity["serving_code_sha256"],
    }
    if (
        not complete["serving_parity_passed"]
        or not serving["passed"]
        or serving["identity"] != serving_identity
        or {probe["direction"] for probe in serving["probes"]} != set(directions)
        or not all(probe["jaw_decisions_equal"] for probe in serving["probes"])
    ):
        raise ValueError("The selected checkpoint lacks matching native-serving evidence")
    if complete["hardware_success_measured"] or serving["hardware_executed"]:
        raise ValueError("This delivery covers offline validation; hardware trials remain deferred")
    files = {
        name: {"bytes": (selected / name).stat().st_size, "sha256": dataset.sha256(selected / name)}
        for name in sorted([*export["files"], "export.json"])
    }
    command = [".venv/bin/python", "scripts/serve_policy.py", "--port", "8000"]
    if len(directions) == 1:
        command += ["--default-prompt", hanoi_policy.PROMPTS[task]]
    relative = selected.resolve().relative_to(pathlib.Path.cwd())
    command += ["policy:checkpoint", "--policy.config", config.name, "--policy.dir", str(relative)]
    return {
        "config": config.name,
        "selected_step": best["step"],
        "completed_optimizer_steps": config.num_train_steps,
        "checkpoint": str(selected.resolve()),
        "checkpoint_relative": str(relative),
        "normalization": str(norm_path.resolve()),
        "validation_flow_loss": best["flow_loss"],
        "test_flow_loss": test["flow_loss"],
        "test_physical": test["physical"],
        "serving_validation": serving,
        "checkpoint_files": files,
        "serve_arguments": command,
        "serve_command": shlex.join(command),
        "model": dataclasses.asdict(config.model),
        "seed": config.seed,
        "batch_size": config.batch_size,
        "ema_decay": config.ema_decay,
        "learning_rate": dataclasses.asdict(config.lr_schedule),
        "optimizer": dataclasses.asdict(config.optimizer),
    }


def source_files(root: pathlib.Path) -> list[pathlib.Path]:
    paths = set()
    for directory in ("src/openpi", "packages/openpi-client/src", "examples/hanoi"):
        paths.update(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".sh", ".sbatch"}
        )
    paths.update(
        root / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "LICENSE",
            "LICENSE_GEMMA.txt",
            "README.md",
            "packages/openpi-client/pyproject.toml",
            "scripts/train.py",
            "scripts/serve_policy.py",
            "docs/hanoi_training_plan.md",
        )
    )
    return sorted(paths)


def archive_source(root: pathlib.Path, destination: pathlib.Path) -> dict:
    files = source_files(root)
    hashes = {str(path.relative_to(root)): dataset.sha256(path) for path in files}
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in files:
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.pax_headers = {}
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    if any(dataset.sha256(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("Source changed while the reproducible archive was being built")
    return {"sha256": dataset.sha256(destination), "files": hashes}


def main(exp_name: str = "hanoi_20260914", remote_host: str = "cw5167@login.torch.hpc.nyu.edu"):
    root = pathlib.Path.cwd()
    run = root / "data/hanoi/runs" / exp_name
    destination = run / "delivery"
    with storage.pipeline_lock(exp_name), filelock.FileLock(str(run / "delivery.lock"), timeout=0):
        manifest = qualify.require_validated_data()
        manager = json.loads((run / "manager.json").read_text())
        if not manager["complete"] or any(manager["models"][task] != "completed" for task in dataset.TASKS):
            raise ValueError("All three training/evaluation stages must finish before final delivery")
        if destination.exists():
            raise FileExistsError(
                f"Delivery already exists; verify its recorded hashes instead of overwriting: {destination}"
            )
        models = []
        for task in dataset.TASKS:
            config = dataclasses.replace(_config.get_config(f"pi05_hanoi_{task}"), exp_name=exp_name)
            models.append(audit_model(config, manifest, run / config.name))
        temporary = run / f".delivery-{uuid.uuid4().hex}.partial"
        temporary.mkdir()
        try:
            source = archive_source(root, temporary / "source.tar.gz")
            evidence = temporary / "evidence"
            evidence.mkdir()
            for name in (
                "audit.json",
                "conversion.json",
                "data_validation.json",
                "execution_validation.json",
                "visual_review.json",
            ):
                shutil.copy2(root / "data/hanoi" / name, evidence / name)
            for path in run.glob("*.json"):
                shutil.copy2(path, evidence / path.name)
            for model in models:
                target = evidence / model["config"]
                target.mkdir()
                for path in (run / model["config"]).glob("*.json"):
                    shutil.copy2(path, target / path.name)
                checkpoint = pathlib.Path(model["checkpoint"]).parent.parent
                for name in ("hanoi_identity.json", "hanoi_placement.json"):
                    shutil.copy2(checkpoint / name, target / name)
            result = {
                "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "exp_name": exp_name,
                "checkpoint_source_host": remote_host,
                "contract": hanoi_policy.CONTRACT,
                "prompts": hanoi_policy.PROMPTS,
                "models": models,
                "source_archive": source,
                "evidence_sha256": {
                    str(path.relative_to(temporary)): dataset.sha256(path)
                    for path in evidence.rglob("*")
                    if path.is_file()
                },
                "hardware_trials_deferred": True,
            }
            dataset.write_json(temporary / "delivery.json", result)
            (temporary / "source.sha256").write_text(f"{source['sha256']}  source.tar.gz\n")
            checksum_lines = [
                f"{metadata['sha256']}  {model['checkpoint_relative']}/{name}"
                for model in models
                for name, metadata in model["checkpoint_files"].items()
            ]
            (temporary / "checkpoints.sha256").write_text("\n".join(checksum_lines) + "\n")
            lines = [
                "# Hanoi policy delivery",
                "",
                "Three independent full pi0.5 fine-tunes, each with 30,000 optimizer steps.",
                "Checkpoints include their own normalization assets. Hardware task success has not been measured.",
                "",
                "| Model | Selected step | Validation flow loss | Test flow loss | Test mean XYZ (mm) | Jaw balanced accuracy |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for model in models:
                metric = model["test_physical"]["all"]
                lines.append(
                    f"| {model['config']} | {model['selected_step']} | {model['validation_flow_loss']:.6g} | "
                    f"{model['test_flow_loss']:.6g} | {metric['mean_valid_xyz_mm']:.4g} | {metric['jaw_balanced_accuracy']:.4g} |"
                )
            lines += [
                "",
                "## Reproduce the environment",
                "",
                "From the copied delivery directory:",
                "",
                "```bash",
                "sha256sum --check source.sha256",
                "mkdir source",
                "tar -xzf source.tar.gz -C source",
                "cd source",
                "GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group hanoi",
                "source examples/hanoi/env.sh",
                "```",
                "",
                "## Download checkpoints",
                "",
                "Run the download commands on the destination machine from the extracted source directory. "
                "Each complete export includes its normalization assets. Run serving in an appropriate GPU allocation.",
                "The multitask client supplies its direction's prompt on every request. See `examples/hanoi/README.md` "
                "for the canonical ROS image/state and command contract.",
            ]
            for model in models:
                fetch = shlex.join(
                    [
                        "rsync",
                        "-avhP",
                        "--append-verify",
                        f"{remote_host}:{model['checkpoint']}/",
                        f"{model['checkpoint_relative']}/",
                    ]
                )
                lines += [
                    "",
                    f"### {model['config']}",
                    "",
                    f"Checkpoint: `{model['checkpoint']}`",
                    "",
                    "```bash",
                    shlex.join(["mkdir", "-p", model["checkpoint_relative"]]),
                    fetch,
                    "```",
                ]
            lines += [
                "",
                "Verify all downloaded checkpoint files from the extracted source directory:",
                "",
                "```bash",
                "sha256sum --check ../checkpoints.sha256",
                "```",
                "",
                "## Serve a selected model",
                "",
                "Choose one model and run its command on a GPU. Each example uses port 8000.",
            ]
            for model in models:
                lines += ["", f"### {model['config']}", "", "```bash", model["serve_command"], "```"]
            lines += [
                "",
                "`delivery.json` contains file sizes and SHA-256 hashes for all selected checkpoint files, "
                "source files, and copied evidence. Checkpoint weights stay in their audited directories.",
            ]
            (temporary / "README.md").write_text("\n".join(lines) + "\n")
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    print(destination)


if __name__ == "__main__":
    tyro.cli(main)
