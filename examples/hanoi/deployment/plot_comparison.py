"""Bar chart of average task progress per method, expert versus non-expert data.

Bars come from three sources, in this order of precedence:

* measured: the hardware trials under ``data/hanoi/deployment``. Per task the best ``--best`` trials (3) by
  progress, averaged per task and then over the tasks, with a trial's progress as in the scoreboard. Expert data:
  the six-task Cosmos and pi0.5 models. Non-expert data: the play-trained models under protocol A (the final
  goal's sentence for the whole trial) on the six full tower moves. By default each bar carries an error bar of
  one standard error of the mean over all trials counted, tasks pooled (``--spread se``): the uncertainty of the
  average, which is what a comparison between methods needs. ``--spread sd`` draws a box instead (mean plus and
  minus one standard deviation of single trials, line at the mean, whiskers to the lowest and highest trial) and
  ``--spread quartiles`` the classic box plot (first to third quartile, median line, mean diamond, same whiskers).
  The label reads the mean and the measure drawn.
* hand-entered trials: ``HAND_TRIALS``, per-trial scores from another agent's logs. They join the measured rows
  and are treated exactly like them (best trials per task, mean, standard error, counts).
* reported: ``REPORTED``, results known only as an average (mean progress and the number of task cases tested).
  Drawn like a measured bar, without an error bar because there are no per-trial numbers.
* placeholder: ``PLACEHOLDERS``, drawn hatched grey: not yet tested.

A measured or reported bar that does not yet cover all six task cases with ``--best`` trials each is marked
``*``. Writes ``exp_vid/six_task_progress.png`` and ``.pdf`` in both checkouts.

    ./run_plot_comparison.sh
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

from examples.hanoi.deployment import scoreboard
from examples.hanoi.deployment.trial_report import COSMOS_ROOT, OPENPI_ROOT

# (legend name, tick label, family stem in the run logs or None, colour or None)
METHODS = (("Cosmos Policy", "Cosmos\nPolicy", "cosmos", "#2a78d6"),
           ("$\\pi_{0.5}$ (VLA)", "$\\pi_{0.5}$\n(VLA)", "pi05", "#eb6834"),
           ("V-JEPA 2-AC", "V-JEPA\n2-AC", None, None),
           ("Ours", "Ours", "ours", "#1baf7a"))
GROUPS = ("Expert data (H1)", "Expert data (H15)", "Non-expert data")
SHORT_GROUP = {"Expert data (H1)": "expert H1", "Expert data (H15)": "expert H15", "Non-expert data": "non-expert"}
# The policy family behind a method in each group, from the run logs: six-task models on the optimal demonstrations
# (the full 15-move tower, H15), play models on play. The one-move setting (H1) has no run logs here.
FAMILY_SUFFIX = {"Expert data (H15)": "_multitask", "Non-expert data": "_play"}
# Per-trial scores from another agent's logs, entered by hand: group -> method index -> [(task, progress in percent)].
# Ours (user, 2026-10-08). Expert data (H15): the 18-trial campaign of 2026-10-07 with fixed settings, including its one
# 93.3, and not the extra trial run after it. Non-expert data: the final play runner of 2026-10-05/06, with the three
# B-to-A trials taken from the 2026-10-07 re-run that added the recovery fix and the transit guard (100 each) in place
# of the three 40s before those rules; a 60 with the recovery fix only is left out. Eighteen trials each.
HAND_TRIALS = {
    "Expert data (H15)": {3: [
        ("AAAA_to_CCCC", 100), ("CCCC_to_BBBB", 100), ("BBBB_to_AAAA", 100), ("AAAA_to_BBBB", 100), ("BBBB_to_CCCC", 100), ("CCCC_to_AAAA", 100),
        ("AAAA_to_CCCC", 100), ("CCCC_to_BBBB", 100), ("BBBB_to_AAAA", 100), ("AAAA_to_BBBB", 100), ("BBBB_to_CCCC", 100), ("CCCC_to_AAAA", 100),
        ("AAAA_to_CCCC", 93.3), ("CCCC_to_BBBB", 100), ("BBBB_to_AAAA", 100), ("AAAA_to_BBBB", 100), ("BBBB_to_CCCC", 100), ("CCCC_to_AAAA", 100),
    ]},
    "Non-expert data": {3: [
        ("CCCC_to_BBBB", 100), ("BBBB_to_AAAA", 100), ("AAAA_to_BBBB", 100), ("BBBB_to_CCCC", 100), ("CCCC_to_AAAA", 100), ("AAAA_to_CCCC", 100),
        ("CCCC_to_AAAA", 100), ("AAAA_to_CCCC", 100), ("CCCC_to_AAAA", 100), ("AAAA_to_CCCC", 100), ("CCCC_to_BBBB", 100), ("BBBB_to_AAAA", 100),
        ("AAAA_to_BBBB", 100), ("BBBB_to_CCCC", 100), ("CCCC_to_BBBB", 100), ("BBBB_to_AAAA", 100), ("AAAA_to_BBBB", 100), ("BBBB_to_CCCC", 100),
    ]},
}
# Results known only as an average: group -> method index -> (mean progress in percent, task cases tested out of six or
# None when unknown, trials when known). None at the moment.
REPORTED = {}
# Values for bars with nothing behind them yet, by group then method index. Drawn hatched grey. The whole one-move
# (H1) group and V-JEPA 2-AC are not tested.
PLACEHOLDERS = {"Expert data (H1)": {0: 100, 1: 100, 2: 50, 3: 100}, "Expert data (H15)": {2: 5}, "Non-expert data": {2: 2}}
NO_SPREAD = {"sd": None, "error": None, "low": None, "high": None, "q1": None, "median": None, "q3": None}
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e6e3"
PLACEHOLDER_FILL, PLACEHOLDER_HATCH = "#deded9", "#b9b8b2"
ERROR_BAR = "#2b2b29"


def hand_rows(hand_trials=None) -> list[dict]:
    """The hand-entered trials as scoreboard-style rows, under the family the group and method map to."""
    rows = []
    for group, methods in (HAND_TRIALS if hand_trials is None else hand_trials).items():
        for k, trials in methods.items():
            family = METHODS[k][2] + FAMILY_SUFFIX[group]
            rows += [{"family": family, "task": task, "progress": score / 100, "peak_progress": score / 100, "aborted": False, "hand_entered": True}
                     for task, score in trials]
    return rows


def comparison_rows(since: str | None = None) -> list[dict]:
    """Six-task trials plus the play trials that are comparable to them (protocol A on the full tower moves), plus the hand-entered ones."""
    play = [r for r in scoreboard.collect(since=since, play=True) if r["goal_protocol"] == "final" and r["distance"] == 15]
    return scoreboard.collect(since=since) + play + hand_rows()


def values(rows: list[dict], best: int, reported=None, placeholders=None) -> dict:
    """(group, method index) -> {value (percent), kind, trials, tasks, complete, spread statistics in percent or None}.

    Spread of the trials counted: ``sd`` (sample standard deviation), ``error`` (standard error of the mean), ``low`` and
    ``high`` (lowest and highest trial), ``q1``/``median``/``q3``; all None unless there are at least two trials.
    """
    reported = REPORTED if reported is None else reported
    placeholders = PLACEHOLDERS if placeholders is None else placeholders
    cases = len(scoreboard.TASKS)
    out = {}
    for group in GROUPS:
        for k, (_, _, stem, _) in enumerate(METHODS):
            family = stem + FAMILY_SUFFIX[group] if stem and group in FAMILY_SUFFIX else None
            mean = scoreboard.average_progress(rows, family, best) if family else None
            if mean is not None:
                picked = scoreboard.best_trials(rows, family, best)
                scores = [100 * x for v in picked.values() for x in v]
                spread = dict(NO_SPREAD)
                if len(scores) > 1:
                    sd = float(np.std(scores, ddof=1))
                    q1, median, q3 = (float(q) for q in np.percentile(scores, [25, 50, 75]))
                    spread = {"sd": sd, "error": sd / float(np.sqrt(len(scores))), "low": float(min(scores)), "high": float(max(scores)),
                              "q1": q1, "median": median, "q3": q3}
                out[(group, k)] = {"value": 100 * mean, "kind": "measured", "trials": len(scores), "tasks": sum(1 for v in picked.values() if v),
                                   "complete": len(scores) == best * cases, **spread}
            elif k in reported.get(group, {}):
                value, tasks, *trials = reported[group][k]
                out[(group, k)] = {"value": float(value), "kind": "reported", "trials": int(trials[0]) if trials else None,
                                   "tasks": None if tasks is None else int(tasks), "complete": tasks is None or int(tasks) == cases, **NO_SPREAD}
            else:
                out[(group, k)] = {"value": float(placeholders.get(group, {}).get(k, 0)), "kind": "placeholder", "trials": 0, "tasks": 0,
                                   "complete": False, **NO_SPREAD}
    return out


def spread_box(ax, x: float, v: dict, mode: str, width: float = 0.26) -> float:
    """Draw the error bar (``se``) or the box and whiskers for one measured bar; returns the highest point drawn."""
    if mode == "se":  # uncertainty of the mean: a plain error bar, cut at the ends of the 0 to 100 scale
        lower, upper = min(v["error"], v["value"]), min(v["error"], 100.0 - v["value"])
        ax.errorbar([x], [v["value"]], yerr=[[lower], [upper]], fmt="none", ecolor=ERROR_BAR, elinewidth=1.4, capsize=5, capthick=1.4, zorder=6)
        return v["value"] + upper
    if mode == "quartiles":
        bottom, top, line = v["q1"], v["q3"], v["median"]
    else:  # progress lies between 0 and 100, so the box is cut there
        bottom, top, line = max(0.0, v["value"] - v["sd"]), min(100.0, v["value"] + v["sd"]), v["value"]
    ink = dict(color=ERROR_BAR, zorder=6, solid_capstyle="butt")
    ax.add_patch(Rectangle((x - width / 2, bottom), width, top - bottom, facecolor=(1, 1, 1, 0.5), edgecolor=ERROR_BAR, linewidth=1.1, zorder=5))
    ax.plot([x - width / 2, x + width / 2], [line, line], linewidth=2.0, **ink)
    for end, edge in ((v["low"], bottom), (v["high"], top)):
        if abs(end - edge) > 1e-9:
            ax.plot([x, x], [edge, end], linewidth=1.1, **ink)
            ax.plot([x - width / 3.2, x + width / 3.2], [end, end], linewidth=1.1, **ink)
    if mode == "quartiles":
        ax.plot([x], [v["value"]], marker="D", markersize=4.5, markerfacecolor=SURFACE, markeredgecolor=ERROR_BAR, markeredgewidth=1.0, zorder=7)
    return max(top, v["high"])


def fmt(x: float) -> str:
    """A label value: whole percent, except that a result short of 100 (or above 0) never rounds onto that bound."""
    whole = round(x)
    return f"{x:.1f}" if (whole == 100 and x < 100) or (whole == 0 and x > 0) else f"{whole:.0f}"


def draw(vals: dict, note: str | None, spread: str = "se"):
    """The figure; ``note`` is the footnote under the chart, or None for a bare figure whose caption lives in the paper."""
    fig = plt.figure(figsize=(14.6, 4.91 if note else 4.3), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0.10, 0.325, 0.80, 0.475] if note else [0.10, 0.21, 0.80, 0.60], facecolor=SURFACE)
    slot, width = 1.0, 0.86
    centers = {}
    for g, group in enumerate(GROUPS):
        for k in range(len(METHODS)):
            centers[(group, k)] = g * (len(METHODS) + 1.4) * slot + k * slot
    ax.set_xlim(-0.9, max(centers.values()) + 0.9)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.set_ylabel("Average task progress (%) ↑", fontsize=12, color=INK, labelpad=10)
    ax.set_xticks([centers[key] for key in centers])
    ax.set_xticklabels([METHODS[k][1] for (_, k) in centers], fontsize=9, linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=5)
    for label, key in zip(ax.get_xticklabels(), centers):
        label.set_color(MUTED if vals[key]["kind"] == "placeholder" else SECONDARY)
        if METHODS[key[1]][0] == "Ours":
            label.set_fontweight("bold")
    fig.canvas.draw()  # freeze the transforms: bars are drawn in inches so their rounded caps stay round
    radius = 0.045
    for key, x in centers.items():
        v = vals[key]
        value, real = v["value"], v["kind"] != "placeholder"
        if value > 0:
            (x0, y0), (x1, y1) = (ax.transData.transform(point) / fig.dpi for point in ((x - width / 2, 0), (x + width / 2, value)))
            style = dict(facecolor=METHODS[key[1]][3], linewidth=0) if real else dict(
                facecolor=PLACEHOLDER_FILL, edgecolor=PLACEHOLDER_HATCH, hatch="////", linewidth=0)
            ax.add_patch(FancyBboxPatch((x0, y0 - radius), x1 - x0, y1 - y0 + radius, transform=fig.dpi_scale_trans,
                                        boxstyle=f"round,pad=0,rounding_size={min(radius, (y1 - y0) / 2)}", **style))
        top, text = value, fmt(value)
        if v["sd"] is not None:
            top = spread_box(ax, x, v, spread)
            text += f" \u00b1 {fmt(v['error' if spread == 'se' else 'sd'])}"
        ax.text(x, top + 2, text + ("*" if real and not v["complete"] else ""), ha="center", va="bottom",
                fontsize=11.5 if v["sd"] is not None else 12.5, color=INK if real else MUTED)
    for g, group in enumerate(GROUPS):
        middle = (centers[(group, 0)] + centers[(group, len(METHODS) - 1)]) / 2
        ax.text(middle, -0.225, group, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13, color=INK)
    shown = {key[1] for key, v in vals.items() if v["kind"] != "placeholder"}
    handles = [Patch(facecolor=color, linewidth=0, label=name) for k, (name, _, _, color) in enumerate(METHODS) if k in shown]
    if any(v["kind"] == "placeholder" for v in vals.values()):
        handles.append(Patch(facecolor=PLACEHOLDER_FILL, edgecolor=PLACEHOLDER_HATCH, hatch="////", linewidth=0, label="Placeholder, not yet tested"))
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=len(handles), frameon=False,
                        fontsize=11.5, handlelength=1.5, handleheight=1.0, columnspacing=2.0)
    for text in legend.get_texts():
        text.set_color(INK)
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8.2, color=MUTED, linespacing=1.45)
    return fig


def footnote(vals: dict, best: int, spread: str = "se") -> str:
    cases = len(scoreboard.TASKS)

    def label(key):
        return f"{SHORT_GROUP[key[0]]} {METHODS[key[1]][1].replace(chr(10), ' ')}"

    measured = [f"{label(key)} {v['trials']}" for key, v in vals.items() if v["kind"] == "measured"]
    reported, uncounted = {}, {}
    for key, v in vals.items():
        if v["kind"] == "reported" and v["tasks"] is None:
            uncounted.setdefault(SHORT_GROUP[key[0]], []).append(METHODS[key[1]][0])
        elif v["kind"] == "reported":
            reported.setdefault((METHODS[key[1]][0], v["tasks"], v["trials"]), []).append(SHORT_GROUP[key[0]])
    measure = "standard error" if spread == "se" else "standard deviation"
    box = {"quartiles": "Boxes: first to third quartile of the trials counted, line at the median, diamond at the mean; whiskers: lowest and highest trial.",
           "sd": "Boxes: mean \u00b1 1 standard deviation of the trials counted, line at the mean; whiskers: lowest and highest trial.",
           "se": "Error bars: \u00b1 1 standard error of the mean, over all trials counted (all tasks pooled)."}[spread]
    lines = [f"Tower of Hanoi on the real arm: bars are the mean task progress over the {cases} tower moves, best {best} trials per task; "
             f"labels read mean \u00b1 {measure}. " + box]
    second = ("Trials: " + ", ".join(measured) + ". ") if measured else ""
    second += " ".join((f"{name}: {trials} trials on all {cases} task cases ({' and '.join(groups)} data)." if tasks == cases and trials else
                        f"{name}: {tasks} of {cases} task cases so far ({' and '.join(groups)} data).")
                       for (name, tasks, trials), groups in reported.items())
    hand = [f"{SHORT_GROUP[group]} {METHODS[k][1].replace(chr(10), ' ')}" for group, methods in HAND_TRIALS.items() for k in methods]
    if hand:
        second += " Entered from another agent's logs: " + ", ".join(hand) + "."
    lines.append(second.strip())
    last = []
    for group, names in uncounted.items():
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        last.append(f"{group[0].upper()}{group[1:]}: {joined} reported without trial counts.")
    if any(v["kind"] != "placeholder" and not v["complete"] for v in vals.values()):
        last.append(f"* not yet all {cases} task cases with {best} trials each.")
    if any(v["kind"] == "placeholder" for v in vals.values()):
        last.append("Hatched grey bars are placeholder values, not yet tested.")
    lines.append(" ".join(last))
    return "\n".join(line for line in lines if line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best", type=int, default=scoreboard.TARGET, help="trials kept per task per policy (the best by progress)")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; only runs from this date on")
    parser.add_argument("--spread", choices=("se", "sd", "quartiles"), default="se",
                        help="on each measured bar: an error bar of one standard error of the mean over all trials counted (se, default), "
                             "a box of the mean +/- one standard deviation with min-max whiskers (sd), or the classic quartile box plot")
    parser.add_argument("--dest", type=Path, nargs="*", default=None, help="folders to write to (default: exp_vid of both checkouts)")
    parser.add_argument("--footnote", action="store_true",
                        help="print the explanatory footnote under the chart; by default the figure is bare and the text is printed here "
                             "for the paper's caption")
    args = parser.parse_args()
    rows = comparison_rows(args.since)
    vals = values(rows, args.best)
    for (group, k), v in vals.items():
        name = f"{group}, {METHODS[k][1].replace(chr(10), ' ')}"
        if v["kind"] == "measured":
            family = METHODS[k][2] + FAMILY_SUFFIX[group]
            stats = "" if v["sd"] is None else (f", standard deviation {v['sd']:.1f} (variance {v['sd'] ** 2:.0f}), standard error {v['error']:.1f}, "
                                                 f"range {v['low']:.0f} to {v['high']:.0f}, quartiles {v['q1']:.0f} / {v['median']:.0f} / {v['q3']:.0f}")
            print(f"{name} ({family}): mean {v['value']:.1f}% over {v['trials']} trials{stats}" + ("" if v["complete"] else " (incomplete)"))
            for task, scores in scoreboard.best_trials(rows, family, args.best).items():
                print(f"  {task}: " + (", ".join(f"{100 * x:.0f}%" for x in scores) if scores else "no trials"))
        else:
            print(f"{name}: {v['value']:.0f}% ({v['kind']}" + (f", {v['tasks']} task cases" if v["kind"] == "reported" else "") + ")")
    note = footnote(vals, args.best, args.spread)
    print("Caption text:\n  " + note.replace("\n", "\n  "))
    fig = draw(vals, note if args.footnote else None, args.spread)
    dests = args.dest or ([OPENPI_ROOT / "exp_vid"] + ([COSMOS_ROOT / "exp_vid"] if COSMOS_ROOT.is_dir() else []))
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            fig.savefig(dest / f"six_task_progress.{suffix}", facecolor=SURFACE)
            print(f"Wrote {dest / f'six_task_progress.{suffix}'}")


if __name__ == "__main__":
    main()
