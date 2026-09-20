# Local deployment validation

These tools load the selected checkpoint or evaluate the transferred training episode before hardware trials.
Run commands from the repository root in the existing model environment.
For async replay, ROS camera input, and robot launch instructions, see the [deployment guide](../README.md).

## Local checkpoint smoke test

After copying the selected forward export, run this from the repository root using the existing local environment:

```bash
OPENPI_DATA_HOME="$PWD/.cache/openpi" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  JAX_PLATFORMS=cuda .venv/bin/python -m examples.hanoi.deployment.validation.smoke_inference
```

The default loads `checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999` through the public policy factory,
uses the export's normalization assets, and runs three requests with a synthetic RGB image and seven-value state.
It checks finite `(63, 4)` outputs, contract metadata, and repeatability with fixed noise and ten sampling steps.
The GPU settings above passed on the local RTX 5080; no environment rebuild or robot connection is needed.
The first request includes compilation. Reported request times include preprocessing, inference, and output conversion.

To also query the transferred training episode, append:

```bash
--episode-path data/hanoi/deployment_debug/aaaa_to_cccc_episode_000/episode.h5
```

The script saves `data/hanoi/local_smoke/inference_report.json` and `inference_samples.npz`, including versions,
checkpoint/normalization hashes, input/output arrays, fixed noise, and timings. The episode check uses the first fresh
observation and reports reference error for debugging; it is not a held-out evaluation or a local/HPC parity check.
Inference runs in process and sends no robot commands. This command does not start the WebSocket server.

## Accuracy on the transferred episode

To compare the selected forward policy against every fresh observation in the transferred training episode:

```bash
OPENPI_DATA_HOME="$PWD/.cache/openpi" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 JAX_PLATFORMS=cuda \
  .venv/bin/python -m examples.hanoi.deployment.validation.evaluate_episode
.venv/bin/python -m examples.hanoi.deployment.validation.plot_episode_accuracy
```

The default evaluates all 7,005 eligible anchors in episode 0, using recorded RGB, measured `proprio[:7]`, and
`action_abs[t:t+63]` targets. It reports next-reference, nine-reference prefix, and full-horizon XYZ/jaw metrics,
excludes terminal padding, and compares XYZ error with simply holding the observed position. Each anchor uses
one reproducible noise draw and ten sampling steps. Predictions never replace recorded observations.
For a smaller evenly spaced sample, pass `--max-anchors N` and a separate `--output-dir`.

Results, per-move/horizon breakdowns, saved predictions/noise, and plots are written to `data/hanoi/episode_accuracy/`.
This measures accuracy on a training demonstration. Closed-loop task success and cross-runtime parity require
separate checks.

