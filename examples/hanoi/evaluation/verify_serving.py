"""Compare a selected checkpoint's native serving output with the evaluation path."""

import dataclasses
import gc
import json
import pathlib

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from examples.hanoi.data import dataset
from openpi import transforms
from openpi.models import model as _model
from openpi.policies import hanoi_policy
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as _config


def verify(config: _config.TrainConfig, snapshot: pathlib.Path, manifest: dict, output_path: pathlib.Path) -> dict:
    identity = {
        "config": config.name,
        "exp_name": config.exp_name,
        "export_sha256": dataset.sha256(snapshot / "export.json"),
        "conversion_sha256": dataset.sha256(pathlib.Path("data/hanoi/conversion.json")),
        "code_sha256": dataset.sha256(pathlib.Path(__file__)),
        "training_identity_sha256": dataset.sha256(config.checkpoint_dir / "hanoi_identity.json"),
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing["identity"] != identity or not existing["passed"]:
            raise ValueError("Existing serving validation belongs to a different checkpoint/contract")
        return existing
    task = config.name.removeprefix("pi05_hanoi_")
    directions = dataset.DIRECTIONS if task == "multitask" else (task,)
    indices = np.load(config.data.frame_indices_path, allow_pickle=False)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(
        data_config, norm_stats=checkpoints.load_norm_stats(snapshot / "assets", data_config.asset_id)
    )
    native = LeRobotDataset(manifest["repo_id"], delta_timestamps={"actions": [i / 30 for i in range(63)]})
    train_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=True),
            *data_config.data_transforms.outputs,
        ]
    )
    requests = []
    for direction in directions:
        episode = next(ep for ep in manifest["episodes"] if ep["direction"] == direction and ep["split"] == "train")
        index = int(indices[np.searchsorted(indices, episode["global_start"])])
        if index >= episode["global_start"] + episode["length"]:
            raise ValueError("Serving probe must use an eligible training anchor from the requested direction")
        row = native[index]
        row["prompt"] = hanoi_policy.PROMPTS[direction]
        requests.append(
            {
                "index": index,
                "direction": direction,
                "inputs": train_transform(dict(row)),
                "canonical": {
                    "observation/image": hanoi_policy.parse_image(row["image"]),
                    "observation/state": np.asarray(row["state"]),
                    "prompt": row["prompt"],
                },
            }
        )
    noise = np.random.default_rng(44).standard_normal((63, 32)).astype(np.float32)
    model = config.model.load(_model.restore_params(snapshot / "params", dtype=jnp.bfloat16))
    model.eval()
    sample = nnx_utils.module_jit(model.sample_actions)
    for request in requests:
        inputs = {key: value for key, value in request["inputs"].items() if key != "actions"}
        observation = _model.Observation.from_dict(jax.tree.map(lambda value: jnp.asarray(value)[None], inputs))
        predicted = np.asarray(sample(jax.random.key(0), observation, num_steps=10, noise=jnp.asarray(noise)[None]))[0]
        request["reference"] = output_transform({"state": inputs["state"].copy(), "actions": predicted.copy()})[
            "actions"
        ]
    del model, sample, native
    jax.clear_caches()
    gc.collect()
    # Reopen through the same public factory used by scripts/serve_policy.py.
    policy = policy_config.create_trained_policy(config, snapshot, sample_kwargs={"num_steps": 10})
    probes = []
    for request in requests:
        actual = policy.infer(request["canonical"], noise=noise)["actions"]
        reference = request["reference"]
        if actual.shape != (63, 4) or not np.isfinite(actual).all():
            raise ValueError("The selected checkpoint cannot serve finite Cartesian/jaw chunks")
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(actual[:, 3] >= 0.5, reference[:, 3] >= 0.5)
        probes.append(
            {
                "direction": request["direction"],
                "training_anchor": request["index"],
                "max_xyz_difference_m": float(np.linalg.norm(actual[:, :3] - reference[:, :3], axis=-1).max()),
                "max_jaw_difference": float(np.abs(actual[:, 3] - reference[:, 3]).max()),
                "jaw_decisions_equal": True,
            }
        )
    result = {
        "identity": identity,
        "passed": True,
        "probes": probes,
        "action_shape": [63, 4],
        "sampling_steps": 10,
        "numpy_noise_seed": 44,
        "atol": 1e-6,
        "rtol": 1e-5,
        "hardware_executed": False,
    }
    dataset.write_json(output_path, result)
    del policy
    jax.clear_caches()
    gc.collect()
    return result
