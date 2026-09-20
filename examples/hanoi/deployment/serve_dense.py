"""Serve the pi0.5 dense contract-five checkpoint for dense_client.py.

Same WebSocket protocol as the other Hanoi servers. The checkpoint's own output transform
returns 30 absolute reference poses at 10 Hz with the jaw intent thresholded, plus the rate and
execution prefix; ``policy_config`` attaches the ``hanoi_dense`` identity to the metadata. This
wrapper validates every reply, warms the model up before listening, and serves on port 8000.

    ./run_dense_server.sh                                                   # 30-step variant
    ./run_dense_server.sh --config-name pi05_hanoi_dense_h16_aaaa_to_cccc   # 16-step variant
    ./run_dense_client.sh --mode live --duration-s 180
"""

import dataclasses
import logging
from pathlib import Path
import time

import numpy as np

DEFAULT_EXPORTS = {
    "pi05_hanoi_dense_aaaa_to_cccc": Path("checkpoints/pi05_hanoi_dense_aaaa_to_cccc/hanoi_dense_20260919/exports/29999"),
    "pi05_hanoi_dense_h16_aaaa_to_cccc": Path("checkpoints/pi05_hanoi_dense_h16_aaaa_to_cccc/hanoi_dense_h16_20260920/exports/29999"),
}


class DensePolicy:
    def __init__(self, policy, horizon: int):
        self._policy, self.horizon = policy, horizon

    def infer(self, obs: dict) -> dict:
        result = self._policy.infer(obs)
        actions = np.array(result["actions"], dtype=np.float32)
        if actions.shape != (self.horizon, 4) or not np.isfinite(actions).all():
            raise ValueError(f"Policy must return {self.horizon} finite XYZ/jaw references")
        if not np.isin(actions[:, 3], (0, 1)).all():
            raise ValueError("Jaw intent must be thresholded to 0/1 by the checkpoint's output transform")
        return {
            "actions": actions,
            "reference_rate_hz": int(result.get("reference_rate_hz", 10)),
            "execution_prefix": int(result.get("execution_prefix", 3)),
            "policy_timing": dict(result.get("policy_timing", {})),
        }


def warm_up(policy, prompt: str) -> float:
    observation = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(7, np.float32),
        "prompt": prompt,
    }
    started = time.monotonic()
    policy.infer(observation)
    return time.monotonic() - started


@dataclasses.dataclass
class Config:
    # Trained variant: pi05_hanoi_dense_aaaa_to_cccc (30-step chunk) or pi05_hanoi_dense_h16_aaaa_to_cccc (16).
    config_name: str = "pi05_hanoi_dense_aaaa_to_cccc"
    checkpoint_dir: Path | None = None  # default: the variant's selected export
    host: str = "127.0.0.1"
    port: int = 8000
    num_steps: int = 10


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import jax

    from openpi.policies import hanoi_dense_policy
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as training_config

    if config.config_name not in hanoi_dense_policy.VARIANTS:
        raise ValueError(f"Unknown dense variant {config.config_name!r}; choose from {sorted(hanoi_dense_policy.VARIANTS)}")
    checkpoint_dir = config.checkpoint_dir or DEFAULT_EXPORTS[config.config_name]
    logging.info("Loading %s from %s", config.config_name, checkpoint_dir)
    trained = policy_config.create_trained_policy(
        training_config.get_config(config.config_name), checkpoint_dir,
        sample_kwargs={"num_steps": config.num_steps},
    )
    metadata = dict(trained.metadata)
    identity = metadata.get("hanoi_dense")
    if not isinstance(identity, dict) or identity.get("export_sha256") is None:
        raise RuntimeError("The checkpoint did not publish its hanoi_dense identity; check policy_config")
    identity = {**identity, "num_steps": config.num_steps, "gpu": jax.devices()[0].device_kind, "model": "pi05_dense"}
    metadata["hanoi_dense"] = identity
    policy = DensePolicy(trained, int(identity["contract"]["action_horizon"]))
    logging.info("Export SHA-256 %s (step %s), %d-step chunks", identity["export_sha256"], identity.get("export_step"), policy.horizon)
    logging.info("Warm-up inference took %.1f s", warm_up(policy, identity["prompt"]))
    logging.info("Serving pi0.5 dense Hanoi policy on ws://%s:%d", config.host, config.port)
    WebsocketPolicyServer(policy, host=config.host, port=config.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
