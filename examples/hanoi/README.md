# Single-arm Hanoi pi0.5

Trossen WXAI, one external ROS RGB camera, and 30 Hz Cartesian references. Start with the
[deployment guide](deployment/README.md) for the robot client, or the
[training and pipeline guide](training/README.md) for dataset preparation through checkpoint delivery.
The full contract is in the [training plan](../../docs/hanoi_training_plan.md).

## Directory guide

| Directory | Purpose |
| --- | --- |
| [deployment/](deployment/README.md) | Async client, ROS/Trossen adapters, policy server, and reference execution. |
| [deployment/validation/](deployment/validation/README.md) | Local checkpoint smoke checks and recorded-episode accuracy. |
| [data/](data/) | Dataset indexing, conversion, normalization, and verification. |
| [training/](training/README.md) | Training/qualification entrypoints and the pipeline operations guide. |
| [evaluation/](evaluation/) | Offline metrics, checkpoint selection, serving parity, and teacher execution checks. |
| [pipeline/](pipeline/) | Job management, scheduling, storage, telemetry, W&B logging, and delivery. |
| [scripts/](scripts/) | Environment setup and Slurm launchers. |
| [tests/](tests/) | All Hanoi example tests, including deployment tests. |

## Key entrypoints

Run from the repository root in the environment described by the corresponding guide.

| Task | Entrypoint |
| --- | --- |
| Serve the selected policy | `./run_policy_server.sh` |
| Move to initial pose and check readback | `./test_robot_initial_pose.sh` |
| Run the live robot client for 30 seconds | `./run_policy_client.sh` |
| Prepare the dataset | `.venv/bin/python -m examples.hanoi.data.convert_hanoi_data_to_lerobot` |
| Inspect the training pipeline | `.venv/bin/python -m examples.hanoi.pipeline.manage` |
| Evaluate production checkpoints | `.venv/bin/python -m examples.hanoi.evaluation.evaluate` |
| Run Hanoi example tests | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest examples/hanoi/tests -q` |

The target adapter passes saved-prediction and simulated-hardware checks. Physical tracking and task success
remain unverified; see the deployment guide for the current execution behavior. Historical training progress and detailed launch arguments are
preserved in the training guide and `data/hanoi/PROGRESS.md` on HPC.
