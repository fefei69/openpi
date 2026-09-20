"""Serve the audited forward Hanoi checkpoint with deployment identity metadata."""

import hashlib
import logging
from pathlib import Path

import tyro

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


def main(
    checkpoint_dir: Path = Path("checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999"),
    host: str = "127.0.0.1",
    port: int = 8000,
):
    logging.basicConfig(level=logging.INFO)
    norm_path = checkpoint_dir / "assets/local/hanoi_roundtrip_20260910/norm_stats.json"
    identity = {
        "config_name": "pi05_hanoi_aaaa_to_cccc",
        "export_sha256": hashlib.sha256((checkpoint_dir / "export.json").read_bytes()).hexdigest(),
        "normalization_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "num_steps": 10,
    }
    policy = policy_config.create_trained_policy(
        config.get_config(identity["config_name"]), checkpoint_dir, sample_kwargs={"num_steps": 10}
    )
    # These hashes identify manifests/stats; they don't replace a weights checksum audit.
    metadata = {**policy.metadata, "hanoi_deployment": identity}
    WebsocketPolicyServer(policy, host=host, port=port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
