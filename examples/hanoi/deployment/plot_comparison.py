"""Bar chart of average task progress per method, expert versus non-expert data.

Measured bars come from the hardware trials under ``data/hanoi/deployment``: per task the best ``--best`` trials
(3) by progress, averaged per task and then over the tasks, with a trial's progress as in the scoreboard. Expert
data: the six-task Cosmos and pi0.5 models. Non-expert data: the play-trained models under protocol A (the final
goal's sentence for the whole trial) on the six full tower moves. A bar with no trials behind it is a placeholder
value from ``PLACEHOLDERS`` drawn hatched grey: not yet tested; a measured bar that does not yet have ``--best``
trials on all six tasks is marked ``*``. Writes ``exp_vid/six_task_progress.png`` and ``.pdf`` in both checkouts.

    ./run_plot_comparison.sh
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.pyplot as plt

from examples.hanoi.deployment import scoreboard
from examples.hanoi.deployment.trial_report import COSMOS_ROOT, OPENPI_ROOT

METHODS = (("Cosmos Policy", "Cosmos\nPolicy", "cosmos", "#2a78d6"),
           ("$\\pi_{0.5}$ (VLA)", "$\\pi_{0.5}$\n(VLA)", "pi05", "#eb6834"),
           ("V-JEPA 2-AC", "V-JEPA\n2-AC", None, None),
           ("Ours", "Ours", None, None))
GROUPS = ("Expert data", "Non-expert data")
# The policy family behind a method in each group: six-task models on the optimal demonstrations, play models on play.
FAMILY_SUFFIX = {"Expert data": "_multitask", "Non-expert data": "_play"}
# Values for bars with no trials behind them yet, by group then method index. Drawn hatched grey.
PLACEHOLDERS = {"Expert data": {2: 20, 3: 97}, "Non-expert data": {0: 5, 1: 5, 2: 10, 3: 97}}
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e6e3"
PLACEHOLDER_FILL, PLACEHOLDER_HATCH = "#deded9", "#b9b8b2"


def comparison_rows(since: str | None = None) -> list[dict]:
    """Six-task trials plus the play trials that are comparable to them: protocol A on the full tower moves."""
    play = [r for r in scoreboard.collect(since=since, play=True) if r["goal_protocol"] == "final" and r["distance"] == 15]
    return scoreboard.collect(since=since) + play


def values(rows: list[dict], best: int) -> dict:
    """(group, method index) -> (percent, measured?, trials counted, complete?)."""
    out = {}
    for group in GROUPS:
        for k, (_, _, stem, _) in enumerate(METHODS):
            family = stem + FAMILY_SUFFIX[group] if stem else None
            measured = scoreboard.average_progress(rows, family, best) if family else None
            if measured is None:
                out[(group, k)] = (float(PLACEHOLDERS.get(group, {}).get(k, 0)), False, 0, False)
            else:
                trials = sum(len(v) for v in scoreboard.best_trials(rows, family, best).values())
                out[(group, k)] = (100 * measured, True, trials, trials == best * len(scoreboard.TASKS))
    return out


def draw(vals: dict, note: str):
    fig = plt.figure(figsize=(11.46, 4.91), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0.13, 0.25, 0.74, 0.53], facecolor=SURFACE)
    slot, width = 1.0, 0.86
    centers = {}
    for g, group in enumerate(GROUPS):
        for k in range(len(METHODS)):
            centers[(group, k)] = g * (len(METHODS) + 1.4) * slot + k * slot
    ax.set_xlim(-0.9, max(centers.values()) + 0.9)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.set_ylabel("Average task progress (%) \u2191", fontsize=12, color=INK, labelpad=10)
    ax.set_xticks([centers[key] for key in centers])
    ax.set_xticklabels([METHODS[k][1] for (_, k) in centers], fontsize=9, linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=5)
    for label, key in zip(ax.get_xticklabels(), centers):
        label.set_color(SECONDARY if vals[key][1] else MUTED)
        if METHODS[key[1]][0] == "Ours":
            label.set_fontweight("bold")
    fig.canvas.draw()  # freeze the transforms: bars are drawn in inches so their rounded caps stay round
    radius = 0.045
    for key, x in centers.items():
        value, measured, _, complete = vals[key]
        if value > 0:
            (x0, y0), (x1, y1) = (ax.transData.transform(point) / fig.dpi for point in ((x - width / 2, 0), (x + width / 2, value)))
            style = dict(facecolor=METHODS[key[1]][3], linewidth=0) if measured else dict(
                facecolor=PLACEHOLDER_FILL, edgecolor=PLACEHOLDER_HATCH, hatch="////", linewidth=0)
            ax.add_patch(FancyBboxPatch((x0, y0 - radius), x1 - x0, y1 - y0 + radius, transform=fig.dpi_scale_trans,
                                        boxstyle=f"round,pad=0,rounding_size={min(radius, (y1 - y0) / 2)}", **style))
        ax.text(x, value + 2, f"{value:.0f}" + ("*" if measured and not complete else ""), ha="center", va="bottom",
                fontsize=12.5, color=INK if measured else MUTED)
    for g, group in enumerate(GROUPS):
        middle = (centers[(group, 0)] + centers[(group, len(METHODS) - 1)]) / 2
        ax.text(middle, -0.21, group, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13, color=INK)
    handles = [Patch(facecolor=color, linewidth=0, label=name) for name, _, family, color in METHODS if family]
    handles.append(Patch(facecolor=PLACEHOLDER_FILL, edgecolor=PLACEHOLDER_HATCH, hatch="////", linewidth=0, label="Placeholder, not yet tested"))
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=len(handles), frameon=False,
                        fontsize=11.5, handlelength=1.5, handleheight=1.0, columnspacing=2.0)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.text(0.5, 0.03, note, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best", type=int, default=scoreboard.TARGET, help="trials kept per task per policy (the best by progress)")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; only runs from this date on")
    parser.add_argument("--dest", type=Path, nargs="*", default=None, help="folders to write to (default: exp_vid of both checkouts)")
    args = parser.parse_args()
    rows = comparison_rows(args.since)
    vals = values(rows, args.best)
    counts = []
    for group in GROUPS:
        for k, (_, tick, stem, _) in enumerate(METHODS):
            value, measured, trials, complete = vals[(group, k)]
            if not measured:
                continue
            family = stem + FAMILY_SUFFIX[group]
            counts.append(f"{group.split()[0].lower()} {tick.replace(chr(10), ' ')} {trials}")
            print(f"{family}: average progress {value:.1f}% over {trials} trials" + ("" if complete else " (incomplete)"))
            for task, scores in scoreboard.best_trials(rows, family, args.best).items():
                print(f"  {task}: " + (", ".join(f"{100 * x:.0f}%" for x in scores) if scores else "no trials"))
    incomplete = any(measured and not complete for _, measured, _, complete in vals.values())
    note = (f"Tower of Hanoi on the real arm: mean task progress over the {len(scoreboard.TASKS)} tower moves, best {args.best} trials per task "
            f"(trials: {', '.join(counts)})." + (f" * fewer than {args.best * len(scoreboard.TASKS)} trials so far." if incomplete else "")
            + " Hatched grey bars are placeholder values, not yet tested.")
    fig = draw(vals, note)
    dests = args.dest or ([OPENPI_ROOT / "exp_vid"] + ([COSMOS_ROOT / "exp_vid"] if COSMOS_ROOT.is_dir() else []))
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            fig.savefig(dest / f"six_task_progress.{suffix}", facecolor=SURFACE)
            print(f"Wrote {dest / f'six_task_progress.{suffix}'}")


if __name__ == "__main__":
    main()
