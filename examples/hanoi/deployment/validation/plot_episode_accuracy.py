"""Plot saved episode predictions and write a compact accuracy report without loading the model."""

import json
import pathlib

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(output_dir: pathlib.Path = pathlib.Path("data/hanoi/episode_accuracy")):
    report = json.loads((output_dir / "report.json").read_text())
    with np.load(output_dir / "predictions.npz") as saved:
        indices = saved["indices"]
        predicted, target, valid = saved["predictions"], saved["targets"], saved["valid"]
        moves = saved["moves"]
    error = np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1) * 1000
    chunk_means = {
        length: (error[:, :length] * valid[:, :length]).sum(axis=1) / valid[:, :length].sum(axis=1)
        for length in (1, 9, 63)
    }
    colors = {1: "#287b8e", 9: "#d87523", 63: "#6155a6"}
    names = {1: "Next reference", 9: "First 9 (0.3 s)", 63: "Full 63 (2.1 s)"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for length in (63, 9, 1):
        axes[0, 0].plot(indices / 30, chunk_means[length], color=colors[length], lw=0.6, alpha=0.8, label=names[length])
    axes[0, 0].set(
        title="Prediction error throughout the demonstration", xlabel="Episode time (s)", ylabel="Mean XYZ error (mm)"
    )
    axes[0, 0].legend()
    horizons = np.arange(1, 64) / 30
    for key, label, color in (
        ("mean", "Mean", "#287b8e"),
        ("median", "Median", "#6155a6"),
        ("p95", "95th percentile", "#d87523"),
    ):
        axes[0, 1].plot(
            horizons, [entry["xyz_error_mm"][key] for entry in report["by_horizon"]], label=label, color=color
        )
    axes[0, 1].axvline(0.3, color="0.5", ls="--", lw=1)
    axes[0, 1].set(title="XYZ error by prediction horizon", xlabel="Future reference time (s)", ylabel="XYZ error (mm)")
    axes[0, 1].legend()
    move_values = np.unique(moves[moves < 15])
    for length, offset in ((9, -0.18), (63, 0.18)):
        means = [chunk_means[length][moves == move].mean() for move in move_values]
        axes[1, 0].bar(move_values + 1 + offset, means, width=0.36, label=names[length], color=colors[length])
    axes[1, 0].set(
        title="Accuracy across the 15 disk moves",
        xlabel="Move number",
        ylabel="Mean XYZ error (mm)",
        xticks=move_values + 1,
    )
    axes[1, 0].legend()
    jaw_error = (predicted[..., 3] >= 0.5) != (target[..., 3] >= 0.5)
    for length in (9, 63):
        wrong = (jaw_error[:, :length] & valid[:, :length]).sum(axis=1) / valid[:, :length].sum(axis=1) * 100
        axes[1, 1].plot(indices / 30, wrong, lw=0.7, color=colors[length], alpha=0.8, label=names[length])
    axes[1, 1].set(
        title="Jaw intent mismatches", xlabel="Episode time (s)", ylabel="Incorrect references per chunk (%)"
    )
    axes[1, 1].legend()
    figure.suptitle(
        f"Hanoi forward policy — training episode 0, {len(indices):,} fresh anchors\nRecorded observations at every query; padded targets excluded",
        fontsize=15,
    )
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    figure.savefig(output_dir / "accuracy.png", dpi=170)
    figure.savefig(output_dir / "accuracy.pdf")
    plt.close(figure)

    cases = [
        ("Episode start", 0),
        ("Largest 9-reference error", int(np.argmax(chunk_means[9]))),
        ("Largest 63-reference error", int(np.argmax(chunk_means[63]))),
    ]
    figure, axes = plt.subplots(len(cases), 4, figsize=(16, 9), constrained_layout=True)
    with h5py.File(report["episode_path"], "r") as episode:
        for row, (label, selected) in enumerate(cases):
            anchor = int(indices[selected])
            axes[row, 0].imshow(episode["pixels"][anchor])
            axes[row, 0].set_title(f"{label}\nrow {anchor}, {anchor / 30:.2f} s", fontsize=10)
            axes[row, 0].axis("off")
            for dim, name in enumerate("XYZ", start=1):
                mask = valid[selected]
                axes[row, dim].plot(
                    horizons[mask], target[selected, mask, dim - 1] * 1000, color="#287b8e", label="Recorded reference"
                )
                axes[row, dim].plot(
                    horizons[mask], predicted[selected, mask, dim - 1] * 1000, color="#d87523", label="Predicted"
                )
                axes[row, dim].axvline(0.3, color="0.5", ls="--", lw=1)
                axes[row, dim].set(xlabel="Future reference time (s)", ylabel=f"{name} position (mm)")
                axes[row, dim].grid(alpha=0.15)
    axes[0, 1].legend()
    figure.suptitle("Recorded and predicted absolute Cartesian trajectories", fontsize=15)
    figure.savefig(output_dir / "trajectory_examples.png", dpi=170)
    plt.close(figure)

    lines = [
        "# Forward Hanoi episode accuracy",
        "",
        f"Evaluated all {len(indices):,} eligible anchors."
        if len(indices) == report["eligible_anchors"]
        else f"Evaluated {len(indices):,} sampled anchors.",
        f"Source: {report['source_rows']:,} rows; {report['excluded_stale_anchors']} stale observation anchors excluded.",
        "The source episode is in the training split. Each request uses its recorded RGB and measured seven-value state.",
        "Targets are action_abs[t:t+63], with no label shift or additional XYZ offset. Terminal padding is excluded.",
        "One reproducible noise draw per anchor; ten sampling steps. These are offline action errors, not robot success rates.",
        "",
        "| Prediction range | Mean XYZ error (mm) | P95 XYZ error (mm) | Jaw accuracy | Balanced jaw accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("first_reference", "Next reference"),
        ("prefix_9", "First 9 / 0.3 s"),
        ("horizon_63", "Full 63 / 2.1 s"),
    ):
        item = report[key]
        lines.append(
            f"| {label} | {item['mean_valid_xyz_mm']:.4f} | {item['xyz_error_mm']['p95']:.4f} | {item['jaw_accuracy']:.4%} | {item['jaw_balanced_accuracy']:.4%} |"
        )
    lines += [
        "",
        "Mean XYZ error first averages valid references within each chunk, then averages anchors.",
        "P95 pools valid reference errors; overlapping chunks share reference targets.",
        "",
        "## Holding the observed XYZ as a baseline",
        "",
    ]
    for key, value in report["hold_position_baseline_xyz_mm"].items():
        lines.append(f"- {key}: {value:.4f} mm.")
    lines += ["", "## Largest errors", ""]
    for label, selected in cases:
        lines.append(
            f"- {label}: row {int(indices[selected])}, first reference {chunk_means[1][selected]:.3f} mm, first nine {chunk_means[9][selected]:.3f} mm, full horizon {chunk_means[63][selected]:.3f} mm."
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- [Accuracy overview](accuracy.png)",
        "- [Trajectory comparisons](trajectory_examples.png)",
        "- [Full metrics](report.json)",
        "- Predictions, targets, valid masks, measured states and noise: `predictions.npz`.",
        "",
        "## Reproduce",
        "",
        "From the repository root:",
        "",
        "```bash",
        'OPENPI_DATA_HOME="$PWD/.cache/openpi" XLA_PYTHON_CLIENT_PREALLOCATE=false \\',
        "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 JAX_PLATFORMS=cuda \\",
        "  .venv/bin/python -m examples.hanoi.deployment.validation.evaluate_episode",
        ".venv/bin/python -m examples.hanoi.deployment.validation.plot_episode_accuracy",
        "```",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))
    print(f"Saved accuracy plots and report to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
