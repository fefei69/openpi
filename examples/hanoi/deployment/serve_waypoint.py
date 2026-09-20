"""Serve the pi0.5 waypoint_v4 checkpoint for the waypoint client (cosmos_client.py).

Same WebSocket protocol and identity handshake as the Cosmos waypoint server, so the client
and everything around it are shared: the v4 episode start pose, the joint-space start check,
one committed destination per observation, workspace bounds, watchdogs and recovery. The
metadata carries ``hanoi_waypoint`` with the contract, prompt, export and normalization hashes.

Run from the repository root in the model ``.venv``::

    ./run_waypoint_server.sh            # port 8000
    ./run_cosmos_client.sh --server ws://127.0.0.1:8000 --mode live --duration-s 150

Each reply is ``{"actions": (8, 4) float32 absolute XYZ + binary jaw intent, "commit_count": 1}``.
The policy's jaw output is thresholded at 0.5 as the contract states.
"""

import dataclasses
import hashlib
import logging
from pathlib import Path
import time

import numpy as np

DEFAULT_EXPORT = Path("checkpoints/pi05_hanoi_waypoint_aaaa_to_cccc/hanoi_waypoint_full_20260917/exports/29999")
HORIZON = 8


class WaypointPolicy:
    """Binary jaw intent and one committed destination per reply, on top of the trained policy."""

    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs: dict) -> dict:
        result = self._policy.infer(obs)
        actions = np.array(result["actions"], dtype=np.float32)
        if actions.shape != (HORIZON, 4) or not np.isfinite(actions).all():
            raise ValueError("Policy must return eight finite XYZ/jaw destinations")
        actions[:, 3] = (actions[:, 3] >= 0.5).astype(np.float32)
        return {"actions": actions, "commit_count": 1, "policy_timing": dict(result.get("policy_timing", {}))}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_identity(checkpoint_dir: Path, *, num_steps: int, gpu: str) -> dict:
    from openpi.policies import hanoi_policy
    from openpi.policies import hanoi_waypoint_policy

    norm_path = checkpoint_dir / f"assets/{hanoi_waypoint_policy.ASSET_ID}/norm_stats.json"
    contract = {k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v) for k, v in hanoi_waypoint_policy.CONTRACT.items()}
    return {
        "model": "pi05",
        "config_name": hanoi_waypoint_policy.CONFIG_NAME,
        "checkpoint": str(checkpoint_dir.resolve()),
        "contract": contract,
        "prompt": hanoi_policy.PROMPTS["aaaa_to_cccc"],
        # Manifest and normalization hashes identify the export; they are not a weights checksum.
        "export_sha256": sha256_file(checkpoint_dir / "export.json"),
        "normalization_sha256": sha256_file(norm_path),
        "num_steps": num_steps,
        "sampling": "jax.random.key(0), advanced once per request",
        "commit_count": 1,
        "gpu": gpu,
    }


def warm_up(policy, prompt: str) -> float:
    """One discarded inference so the first robot request does not pay JAX compilation."""
    observation = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(7, np.float32),
        "observation/cartesian_position": np.zeros(3, np.float32),
        "prompt": prompt,
    }
    started = time.monotonic()
    policy.infer(observation)
    return time.monotonic() - started


@dataclasses.dataclass
class Config:
    checkpoint_dir: Path = DEFAULT_EXPORT
    host: str = "127.0.0.1"
    port: int = 8000
    num_steps: int = 10


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import jax

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as training_config

    gpu = jax.devices()[0].device_kind
    identity = build_identity(config.checkpoint_dir, num_steps=config.num_steps, gpu=gpu)
    logging.info("Loading %s from %s", identity["config_name"], config.checkpoint_dir)
    trained = policy_config.create_trained_policy(
        training_config.get_config(identity["config_name"]), config.checkpoint_dir,
        sample_kwargs={"num_steps": config.num_steps},
    )
    policy = WaypointPolicy(trained)
    logging.info("Export manifest SHA-256 %s", identity["export_sha256"])
    logging.info("Warm-up inference took %.1f s", warm_up(policy, identity["prompt"]))
    metadata = {**trained.metadata, "hanoi_waypoint": identity}
    logging.info("Serving pi0.5 Hanoi waypoint policy on ws://%s:%d", config.host, config.port)
    WebsocketPolicyServer(policy, host=config.host, port=config.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
