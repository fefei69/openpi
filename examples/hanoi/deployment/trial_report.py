"""Progress table over a series of trials (runs), not a binary success count.

Selects runs under ``data/hanoi/deployment`` by ``--tag`` (the client's ``--tag``), ``--family``
(``cosmos_dense``, ``pi05_dense``), ``--config`` and/or ``--since``, or takes run directories directly,
and writes ``exp_vid/<name>/trials.md`` and ``trials.csv`` in the family's checkout. Per trial:

* moves completed and how many were legal, the optimal prefix (leading moves matching the 15-move solution),
* remaining moves to the goal from the final board and ``progress = (15 - remaining) / 15``,
* solved or not, solve time, how the run ended and why, brakes, tracking error.

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
    r = reconstruct(events).report()
    solved_t = next((e["monotonic_s"] - events[0]["monotonic_s"] for e in events if e["event"] == "task_solved"), None)
    if solved_t is None and r["solved"] and r["moves"]:
        solved_t = r["moves"][-1]["t_s"]
    return {
        "run": run.name, "date": datetime.datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "tag": summary.get("tag", ""), "family": summary.get("policy_family"), "config": summary.get("config_name"),
        "status": summary.get("status"), "moves": r["moves_completed"], "legal": r["legal_moves"],
        "optimal_prefix": r["optimal_prefix"], "remaining": r["remaining_moves"], "progress": r["progress"],
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
    progress = [r["progress"] for r in rows if r["progress"] is not None]
    solved = sum(1 for r in rows if r["solved"])
    lines = [f"# {name}: {len(rows)} trials", "",
             f"Solved {solved}/{len(rows)}; mean progress {np.mean(progress):.2f} (median {np.median(progress):.2f}); "
             f"mean optimal prefix {np.mean([r['optimal_prefix'] for r in rows]):.1f} moves; "
             f"median moves {np.median([r['moves'] for r in rows]):.0f}; "
             f"solve time {np.mean([r['solve_time_s'] for r in rows if r['solve_time_s']]) if solved else float('nan'):.0f} s mean over solved runs", "",
             "| # | run | status | moves (legal) | optimal prefix | remaining | progress | solve time | brakes | tracking p95 | reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, 1):
        pct = f"{100 * r['progress']:.0f}%" if r['progress'] is not None else ""
        lines.append(f"| {k} | {r['run']} | {r['status']} | {r['moves']} ({r['legal']}) | {r['optimal_prefix']} | {r['remaining']} | {pct} | "
                     f"{r['solve_time_s'] or ''} | {r['brakes']} | {r['tracking_p95_mm']} | {r['reason']} |")
    text = "\n".join(lines) + "\n"
    (dest / "trials.md").write_text(text)
    print(text)
    print(f"Wrote {dest / 'trials.md'} and trials.csv")


if __name__ == "__main__":
    main()
