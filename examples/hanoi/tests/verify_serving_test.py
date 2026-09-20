import dataclasses
import pathlib

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from examples.hanoi.data import dataset
from examples.hanoi.evaluation import verify_serving
from openpi import transforms
from openpi.models import model as model_module
from openpi.policies import hanoi_policy
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as training_config


class SmallModel(nnx.Module):
    def __init__(self):
        self.bias = nnx.Param(jnp.arange(32, dtype=jnp.float32) / 100)

    def sample_actions(self, rng, observation, *, num_steps, noise):
        del rng
        positions = jnp.arange(1, observation.tokenized_prompt.shape[-1] + 1)
        prompt = (observation.tokenized_prompt * positions).sum(axis=-1) / 1_000_000
        pixels = observation.images["base_0_rgb"].mean(axis=(1, 2, 3))
        base = observation.state * 0.01 + self.bias.value + (prompt + pixels)[:, None]
        return base[:, None, :] + noise * (0.1 / num_steps)


@dataclasses.dataclass(frozen=True)
class SmallModelInputs(transforms.DataTransformFn):
    def __call__(self, data):
        prompt = data.pop("prompt")
        data["tokenized_prompt"] = np.array([ord(value) for value in prompt], dtype=np.int32)
        data["tokenized_prompt_mask"] = np.ones(len(prompt), dtype=bool)
        return transforms.PadStatesAndActions(32)(data)


@pytest.mark.parametrize("mismatch", [None, "actions", "prompt"])
def test_native_serving_matches_evaluation_and_detects_changed_outputs(tmp_path, monkeypatch, mismatch):
    monkeypatch.chdir(tmp_path)
    base = training_config.get_config("pi05_hanoi_multitask")
    indices_path = tmp_path / "indices.npy"
    np.save(indices_path, [0, 1])
    config = dataclasses.replace(
        base, exp_name="unit", data=dataclasses.replace(base.data, frame_indices_path=str(indices_path))
    )
    snapshot = config.checkpoint_dir / "exports/10000"
    snapshot.mkdir(parents=True)
    dataset.write_json(snapshot / "export.json", {"step": 10000})
    dataset.write_json(config.checkpoint_dir / "hanoi_identity.json", {"config": config.name})
    dataset.write_json(pathlib.Path("data/hanoi/conversion.json"), {})
    stats = {
        key: normalize.NormStats(mean=np.zeros(size), std=np.ones(size), q01=-np.ones(size), q99=np.ones(size))
        for key, size in (("state", 7), ("actions", 4))
    }
    normalize.save(config.assets_dirs / config.data.repo_id, stats)
    normalize.save(snapshot / "assets" / config.data.repo_id, stats)
    monkeypatch.setattr(
        training_config.ModelTransformFactory, "__call__", lambda *args: transforms.Group(inputs=[SmallModelInputs()])
    )
    monkeypatch.setattr(type(config.model), "load", lambda *args: SmallModel())
    monkeypatch.setattr(model_module, "restore_params", lambda *args, **kwargs: {})

    class Native:
        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            return {
                "image": np.full((3, 224, 224), (100 + index) / 255, dtype=np.float32),
                "state": np.linspace(0.1, 0.7, 7, dtype=np.float32),
                "actions": np.zeros((63, 4), dtype=np.float32),
            }

    monkeypatch.setattr(verify_serving, "LeRobotDataset", Native)
    factory = policy_config.create_trained_policy

    def changed_factory(*args, **kwargs):
        policy = factory(*args, **kwargs)
        original = policy.infer

        def infer(obs, **kwargs):
            if mismatch == "prompt":
                other = next(prompt for prompt in hanoi_policy.PROMPTS.values() if prompt != obs["prompt"])
                obs = {**obs, "prompt": other}
            result = original(obs, **kwargs)
            if mismatch == "actions":
                result["actions"][:, 0] += 0.01
            return result

        policy.infer = infer
        return policy

    if mismatch:
        monkeypatch.setattr(policy_config, "create_trained_policy", changed_factory)
    manifest = {
        "repo_id": hanoi_policy.REPO_ID,
        "episodes": [
            {"direction": direction, "split": "train", "global_start": i, "length": 1}
            for i, direction in enumerate(dataset.DIRECTIONS)
        ],
    }
    output_path = tmp_path / "serving_validation.json"
    if mismatch:
        with pytest.raises(AssertionError):
            verify_serving.verify(config, snapshot, manifest, output_path)
        assert not output_path.exists()
    else:
        result = verify_serving.verify(config, snapshot, manifest, output_path)
        assert result["passed"]
        assert len(result["probes"]) == 2
        assert all(probe["max_xyz_difference_m"] < 1e-6 for probe in result["probes"])
        assert verify_serving.verify(config, snapshot, manifest, output_path) == result
