"""Progress table over a series of trials (runs), not a binary success count.

Selects runs under ``data/hanoi/deployment`` by ``--tag`` (the client's ``--tag``), ``--family``
(``cosmos_dense``, ``pi05_dense``), ``--config`` and/or ``--since``, or takes run directories directly,
and writes ``exp_vid/<name>/trials.md`` and ``trials.csv`` in the family's checkout. Per trial:

* moves completed, how many were optimal / detours / null / illegal, the optimal prefix (leading moves matching
  the 15-move solution),
* remaining moves to the goal from the final board and ``progress = (15 - remaining) / 15``, and the peak
  progress over the run (the final board is unscorable after an illegal stacking),
* solved or not, solve time, how the run ended and why, brakes, tracking error.

Runs that ended before the policy commanded anything (no camera, server down) are listed as aborted and
left out of the aggregates.

    ./run_trial_report.sh --tag cosmos_h16_trials --name cosmos_h16_trials
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np

from examples.hanoi.deployment.progress import reconstruct

OPENPI_ROOT = Path(__file__).resolve().parents[3]
COSMOS_ROOT = OPENPI_ROOT.parent / "cosmos-policy"


def failure_reason(summary: dict, events: list) -> str:
    status = summary.get("status")
    if status in ("task_solved",):
        return "solved"
    if status == "missed_grasp":
        return f"missed grasp, stroke {1000 * (summary.get('missed_grasp_stroke_m') or 0):.1f} mm"
    for e in reversed(events):
        if e["event"] in ("stopped_on_rejected_command", "failure"):
            return f"{status}: {str(e.get('error') or e.get('reason'))[:70]}"
        if e["event"] == "rejected" and status == "rejected_command":
            return f"rejected: {str(e.get('reason'))[:70]}"
    return status or "?"


def trial_row(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    r = reconstruct(events, summary.get("start_peg") or "A", summary.get("goal_peg") or "C", goal_board=summary.get("goal_board")).report()
    solved_t = next((e["monotonic_s"] - events[0]["monotonic_s"] for e in events if e["event"] == "task_solved"), None)
    if solved_t is None and r["solved"] and r["moves"]:
        solved_t = r["moves"][-1]["t_s"]
    commands = sum(1 for e in events if e["event"] == "command")
    return {
        "run": run.name, "date": datetime.datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "tag": summary.get("tag", ""), "family": summary.get("policy_family"), "config": summary.get("config_name"),
        "task": summary.get("task_direction") or "AAAA_to_CCCC",
        "goal_board": r["goal_board"], "distance": r["distance"], "goal_protocol": summary.get("goal_protocol") or "final",
        "trial_rules": summary.get("trial_rules") or "duration", "first_error": r["moves_before_first_error"], "ring_moves": r["ring_moves"],
        "start": summary.get("start") or "",
        "status": summary.get("status"), "aborted": commands == 0, "commands": commands,
        "moves": r["moves_completed"], "legal": r["legal_moves"], **{k: r["move_counts"][k] for k in ("optimal", "detour", "null", "illegal")},
        "clean": r["clean"],
        "optimal_prefix": r["optimal_prefix"], "remaining": r["remaining_moves"], "progress": r["progress"],
        "peak_progress": r["peak_progress"],
        "solved": r["solved"], "solve_time_s": None if solved_t is None else round(solved_t, 1),
        "duration_s": round(events[-1]["monotonic_s"] - events[0]["monotonic_s"]),
        "reason": failure_reason(summary, events), "brakes": summary.get("brakes"),
        "tracking_p95_mm": None if summary.get("tracking_error_p95_mm") is None else round(summary["tracking_error_p95_mm"], 1),
        "final_board": json.dumps(r["final_board"]).replace(" ", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--since", default=None, help="YYYY-MM-DD")
    parser.add_argument("--name", default=None, help="campaign folder under exp_vid (default: the tag)")
    parser.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args()
    runs = list(args.runs)
    if not runs:
        since = datetime.datetime.strptime(args.since, "%Y-%m-%d").timestamp() if args.since else 0
        for run in sorted((OPENPI_ROOT / "data/hanoi/deployment").glob("*_live_*")):
            if not (run / "summary.json").exists() or run.stat().st_mtime < since:
                continue
            summary = json.loads((run / "summary.json").read_text())
            if args.tag and summary.get("tag") != args.tag:
                continue
            if args.family and summary.get("policy_family") != args.family:
                continue
            if args.config and summary.get("config_name") != args.config:
                continue
            runs.append(run)
    if not runs:
        raise SystemExit("No runs matched")
    rows = [trial_row(run.resolve()) for run in runs]
    family = rows[0]["family"] or ""
    name = args.name or args.tag or f"{family}_trials"
    dest = args.dest or ((COSMOS_ROOT if family.startswith("cosmos") else OPENPI_ROOT) / "exp_vid" / name)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    trials = [r for r in rows if not r["aborted"]]
    aborted = len(rows) - len(trials)
    progress = [r["progress"] for r in trials if r["progress"] is not None]
    peak = [r["peak_progress"] for r in trials if r["peak_progress"] is not None]
    solved = sum(1 for r in trials if r["solved"])
    unscored = len(trials) - len(progress)
    lines = [f"# {name}: {len(trials)} trials" + (f" ({aborted} aborted before the policy ran, not counted)" if aborted else ""), "",
             f"Solved {solved}/{len(trials)}; mean final progress {np.mean(progress) if progress else float('nan'):.2f} "
             f"(median {np.median(progress) if progress else float('nan'):.2f}"
             + (f", {unscored} unscorable after an illegal stacking" if unscored else "") + "); "
             f"mean peak progress {np.mean(peak) if peak else float('nan'):.2f}; "
             f"mean optimal prefix {np.mean([r['optimal_prefix'] for r in trials]) if trials else float('nan'):.1f} moves; "
             f"median moves {np.median([r['moves'] for r in trials]) if trials else float('nan'):.0f}; "
             f"solve time {np.mean([r['solve_time_s'] for r in trials if r['solve_time_s']]) if solved else float('nan'):.0f} s mean over solved runs", "",
             "| # | run | task | status | moves (optimal / detour / null / illegal) | optimal prefix | remaining | progress | peak | solve time | brakes | tracking p95 | reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, 1):
        pct = f"{100 * r['progress']:.0f}%" if r['progress'] is not None else ""
        peak_pct = f"{100 * r['peak_progress']:.0f}%" if r['peak_progress'] is not None else ""
        status = "aborted" if r["aborted"] else r["status"]
        kinds = f"{r['moves']} ({r['optimal']} / {r['detour']} / {r['null']} / {r['illegal']})"
        lines.append(f"| {k} | {r['run']} | {r['task']} | {status} | {kinds} | {r['optimal_prefix']} | {r['remaining']} | {pct} | {peak_pct} | "
                     f"{r['solve_time_s'] or ''} | {r['brakes']} | {r['tracking_p95_mm']} | {r['reason']} |")
    text = "\n".join(lines) + "\n"
    (dest / "trials.md").write_text(text)
    print(text)
    print(f"Wrote {dest / 'trials.md'} and trials.csv")


if __name__ == "__main__":
    main()
