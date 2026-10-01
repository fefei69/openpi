"""Six-task campaign scoreboard: success rate, progress and move quality per task per policy, kept up to date.

Every dense client run of a six-task policy (Cosmos or pi0.5) regenerates ``exp_vid/six_task_scoreboard.md`` (and
``six_task_trials.csv``) in both checkouts from all recorded runs under ``data/hanoi/deployment``; run it by hand with
``./run_scoreboard.sh`` (``--since YYYY-MM-DD``, ``--exclude <run>`` to leave runs out, ``--target`` trials per case).

* Trials per task per policy against the campaign target (3 each: 6 tasks x 3 x 2 policies = 36 runs), solved count and
  success rate, mean progress. A trial's progress is ``(15 - remaining moves) / 15`` from its final board; when an illegal
  stacking makes the final board unscorable it is the peak progress (best board reached), marked with ``*``.
* Move quality: how many moves were optimal (on a shortest path), detours (legal, off it), null (put back on the same
  peg) or illegal (larger ring onto a smaller one), and how many trials were clean (every move optimal).
* Runs that ended before the policy commanded anything (no camera frame, server down) are listed but not counted.
  To leave a run out for good, move its folder out of ``data/hanoi/deployment``.

The play-trained policies (goal board sentences) get their own page, ``exp_vid/play_scoreboard.md`` and
``play_trials.csv``, laid out by the arm protocol (``docs/hanoi_play_arm_protocol.md`` in cosmos-policy): protocol A
(the final goal's sentence for the whole trial) at distance 15 on the six tasks and at distances 1, 3 and 7 on two,
protocol C (the next board's sentence after every move) at distance 15, three trials per pair, 54 per policy; per pair
the trials done, solved count, mean moves before the first error and mean progress, then by distance and goal board.
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np

from examples.hanoi.deployment.progress import DIRECTIONS, goal_at_distance
from examples.hanoi.deployment.trial_report import COSMOS_ROOT, OPENPI_ROOT, trial_row

TASKS = DIRECTIONS
FAMILIES = (("cosmos_multitask", "Cosmos"), ("pi05_multitask", "pi0.5"))
TARGET = 3
RUNS_DIR = OPENPI_ROOT / "data/hanoi/deployment"
FILENAME = "six_task_scoreboard.md"
PLAY_FILENAME = "play_scoreboard.md"
PLAY_NAMES = {"cosmos_play": "Cosmos Policy (play)", "pi05_play": "pi0.5 (play)"}
PROTOCOL_NAMES = {"final": "Protocol A: final goal for the whole trial (the comparison)",
                  "next": "Protocol C: next board after every move (diagnostic)"}
# The arm protocol's pairs in their fixed order: (sentence protocol, task, distance).
PLAY_PAIRS = ([("final", task, 15) for task in TASKS]
              + [("final", task, d) for task in ("AAAA_to_BBBB", "AAAA_to_CCCC") for d in (1, 3, 7)]
              + [("next", task, 15) for task in TASKS])


def collect(runs_dir: Path = RUNS_DIR, since: str | None = None, exclude=(), play: bool = False) -> list[dict]:
    """Trial rows for every six-task run (or, with ``play``, every play-policy run), oldest first."""
    cutoff = datetime.datetime.strptime(since, "%Y-%m-%d").timestamp() if since else 0
    rows = []
    for run in sorted(Path(runs_dir).glob("*_live_*")):
        if run.name in exclude or not (run / "summary.json").exists() or run.stat().st_mtime < cutoff:
            continue
        summary = json.loads((run / "summary.json").read_text())
        family = summary.get("policy_family") or ""
        if (not family.endswith("_play")) if play else (family not in dict(FAMILIES)):
            continue
        rows.append(trial_row(run.resolve()))
    return rows


def trial_progress(r: dict) -> float | None:
    return r["progress"] if r["progress"] is not None else r["peak_progress"]


def best_trials(rows: list[dict], family: str, best: int = TARGET) -> dict:
    """Per task, the progress of a policy's ``best`` highest-scoring counted trials (all of them when it has no more)."""
    picked = {}
    for task in TASKS:
        scores = [trial_progress(r) for r in rows if r["family"] == family and r["task"] == task and not r["aborted"]]
        picked[task] = sorted((x for x in scores if x is not None), reverse=True)[:best]
    return picked


def average_progress(rows: list[dict], family: str, best: int = TARGET) -> float | None:
    """Mean over the tasks tried of each task's mean progress over its best trials."""
    means = [float(np.mean(v)) for v in best_trials(rows, family, best).values() if v]
    return float(np.mean(means)) if means else None


def pct(x) -> str:
    return "" if x is None else f"{100 * x:.0f}%"


def _case(rows: list[dict]) -> dict:
    trials = [r for r in rows if not r["aborted"]]
    scores = [trial_progress(r) for r in trials]
    scores = [x for x in scores if x is not None]
    solved = sum(1 for r in trials if r["solved"])
    counts = {k: sum(r[k] for r in trials) for k in ("moves", "optimal", "detour", "null", "illegal")}
    return {"trials": len(trials), "solved": solved, "rate": None if not trials else solved / len(trials),
            "progress": None if not scores else float(np.mean(scores)), "peak_used": sum(1 for r in trials if r["progress"] is None),
            "clean": sum(1 for r in trials if r["clean"]), **counts}


def render(rows: list[dict], target: int = TARGET, after: str | None = None) -> str:
    by = {(f, t): [r for r in rows if r["family"] == f and r["task"] == t] for f, _ in FAMILIES for t in TASKS}
    cases = {k: _case(v) for k, v in by.items()}
    totals = {f: _case([r for r in rows if r["family"] == f]) for f, _ in FAMILIES}
    done = sum(min(c["trials"], target) for c in cases.values())
    goal = target * len(TASKS) * len(FAMILIES)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["# Six-task Hanoi scoreboard", "",
             f"Updated {stamp}" + (f" after `{after}`" if after else "") + f". Target {target} trials per task per policy "
             f"({goal} runs): {done} of {goal} done. Progress = (15 - remaining moves) / 15 from the final board; `*` marks a trial "
             "(or a mean including one) scored by its peak because an illegal stacking left the final board unscorable. "
             "Aborted runs (no policy command) are listed at the end and not counted.", "",
             "## Success rate and progress", "",
             "| Task | " + " | ".join(f"{n} trials | {n} solved | {n} progress" for _, n in FAMILIES) + " |",
             "|---|" + "---|" * (3 * len(FAMILIES))]
    for t in TASKS + ("All six",):
        cells = []
        for f, _ in FAMILIES:
            c = cases[(f, t)] if t in TASKS else totals[f]
            want = target if t in TASKS else target * len(TASKS)
            cells += [f"{c['trials']} of {want}", f"{c['solved']} ({pct(c['rate'])})" if c["trials"] else "",
                      pct(c["progress"]) + ("*" if c["peak_used"] else "")]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Move quality", "",
              "| Policy | Task | Trials | Moves | Optimal | Detour | Null | Illegal | Clean trials |", "|---|---|---|---|---|---|---|---|---|"]
    for f, n in FAMILIES:
        for t in TASKS + ("All six",):
            c = cases[(f, t)] if t in TASKS else totals[f]
            if not c["trials"]:
                continue
            lines.append(f"| {n} | {t} | {c['trials']} | {c['moves']} | {c['optimal']} | {c['detour']} | {c['null']} | {c['illegal']} | {c['clean']} |")
    lines += ["", "## Still to run", ""]
    for f, n in FAMILIES:
        left = [f"{t} x{target - cases[(f, t)]['trials']}" for t in TASKS if cases[(f, t)]["trials"] < target]
        lines.append(f"- {n}: " + (", ".join(left) if left else "done"))
    for f, n in FAMILIES:
        fam = [r for r in rows if r["family"] == f]
        lines += ["", f"## {n} trials ({f})", ""]
        if not fam:
            lines.append("none yet")
            continue
        lines += ["| # | date | task | run | status | moves (optimal / detour / null / illegal) | optimal prefix | progress | peak | solve time | reason |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        k = 0
        for r in fam:
            if r["aborted"]:
                continue
            k += 1
            prog = pct(r["progress"]) if r["progress"] is not None else pct(r["peak_progress"]) + "*"
            kinds = f"{r['moves']} ({r['optimal']} / {r['detour']} / {r['null']} / {r['illegal']})"
            lines.append(f"| {k} | {r['date']} | {r['task']} | {r['run']} | {r['status']} | {kinds} | {r['optimal_prefix']} | {prog} | "
                         f"{pct(r['peak_progress'])} | {r['solve_time_s'] or ''} | {r['reason']} |")
        aborted = [r for r in fam if r["aborted"]]
        if aborted:
            lines += ["", "Aborted, not counted: " + "; ".join(f"{r['run']} ({r['task']}, {r['reason']})" for r in aborted)]
    return "\n".join(lines) + "\n"


def _pair(rows: list[dict]) -> dict:
    trials = [r for r in rows if not r["aborted"]]
    scores = [x for x in (trial_progress(r) for r in trials) if x is not None]
    solved = sum(1 for r in trials if r["solved"])
    return {"trials": len(trials), "solved": solved, "clean": None if not trials else float(np.mean([r["first_error"] for r in trials])),
            "progress": None if not scores else float(np.mean(scores)), "peak_used": sum(1 for r in trials if r["progress"] is None)}


def _pair_cells(c: dict, want: int) -> str:
    if not c["trials"]:
        return f"0 of {want} |  |  |  "
    return f"{c['trials']} of {want} | {c['solved']} | {c['clean']:.1f} | {pct(c['progress'])}{'*' if c['peak_used'] else ''}"


def render_play(rows: list[dict], target: int = TARGET, after: str | None = None) -> str:
    """The play-policy page: the arm protocol's pairs per policy, then by distance and goal board, and the trials."""
    families = list(dict.fromkeys(["cosmos_play"] + [r["family"] for r in rows]))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    per_policy = target * len(PLAY_PAIRS)
    lines = ["# Play-trained policies: arm protocol scoreboard", "",
             f"Updated {stamp}" + (f" after `{after}`" if after else "") + f". Target {target} trials per pair, {per_policy} per policy. "
             "A trial starts from the task's full tower and ends at the goal board, after twice the goal distance in moves, on a ring "
             "placed on a smaller one, a dropped ring, or 60 s without a completed move. Clean moves = moves before the first error "
             "(a move that did not reduce the distance to the goal). Progress = (distance - remaining moves) / distance from the final "
             "board; `*` marks a peak used because the final board was unscorable. Aborted runs are listed and not counted."]
    for family in families:
        fam = [r for r in rows if r["family"] == family]
        by = {pair: [r for r in fam if (r["goal_protocol"], r["task"], r["distance"]) == pair] for pair in PLAY_PAIRS}
        done = sum(min(_pair(v)["trials"], target) for v in by.values())
        lines += ["", f"## {PLAY_NAMES.get(family, family)} ({family}): {done} of {per_policy} done"]
        for protocol in ("final", "next"):
            pairs = [pair for pair in PLAY_PAIRS if pair[0] == protocol]
            lines += ["", f"### {PROTOCOL_NAMES[protocol]}", "",
                      "| Task | Distance | Goal board | Trials | Solved | Mean clean moves | Mean progress |", "|---|---|---|---|---|---|---|"]
            for pair in pairs:
                lines.append(f"| {pair[1]} | {pair[2]} | {goal_at_distance(pair[1], pair[2])} | {_pair_cells(_pair(by[pair]), target)} |")
            for d in sorted({pair[2] for pair in pairs}, reverse=True):
                group = [pair for pair in pairs if pair[2] == d]
                lines.append(f"| All at distance {d} | {d} |  | {_pair_cells(_pair([r for pair in group for r in by[pair]]), target * len(group))} |")
        final = [r for pair in PLAY_PAIRS if pair[0] == "final" for r in by[pair] if not r["aborted"]]
        if final:
            lines += ["", "### By goal board (protocol A)", "", "| Goal board | Trials | Solved | Mean clean moves | Mean progress |", "|---|---|---|---|---|"]
            for board in dict.fromkeys(goal_at_distance(pair[1], pair[2]) for pair in PLAY_PAIRS if pair[0] == "final"):
                c = _pair([r for r in final if r["goal_board"] == board])
                if c["trials"]:
                    lines.append(f"| {board} | {c['trials']} | {c['solved']} | {c['clean']:.1f} | {pct(c['progress'])}{'*' if c['peak_used'] else ''} |")
        lines += ["", "### Still to run", ""]
        for protocol, label in (("final", "Protocol A"), ("next", "Protocol C")):
            left = [f"{pair[1]} d{pair[2]} x{target - _pair(by[pair])['trials']}" for pair in PLAY_PAIRS
                    if pair[0] == protocol and _pair(by[pair])["trials"] < target]
            lines.append(f"- {label}: " + (", ".join(left) if left else "done"))
        counted = [r for r in fam if not r["aborted"]]
        if counted:
            lines += ["", "### Trials", "",
                      "| # | date | protocol | task | distance | goal | run | status | moves (optimal / detour / null / illegal) | clean moves | progress | reason |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for k, r in enumerate(counted, 1):
                prog = pct(r["progress"]) if r["progress"] is not None else pct(r["peak_progress"]) + "*"
                kinds = f"{r['ring_moves']} ({r['optimal']} / {r['detour']} / {r['null']} / {r['illegal']})"
                lines.append(f"| {k} | {r['date']} | {'A' if r['goal_protocol'] == 'final' else 'C'} | {r['task']} | {r['distance']} | {r['goal_board']} | "
                             f"{r['run']} | {r['status']} | {kinds} | {r['first_error']} | {prog} | {r['reason']} |")
        aborted = [r for r in fam if r["aborted"]]
        if aborted:
            lines += ["", "Aborted, not counted: " + "; ".join(f"{r['run']} ({r['task']}, {r['reason']})" for r in aborted)]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def update(after: str | None = None, runs_dir: Path = RUNS_DIR, dests=None, target: int = TARGET, since=None, exclude=()) -> list[Path]:
    """Regenerate both scoreboards in every destination folder (default: exp_vid of both checkouts); returns the files written."""
    rows = collect(runs_dir, since=since, exclude=exclude)
    play_rows = collect(runs_dir, since=since, exclude=exclude, play=True)
    text = render(rows, target=target, after=after)
    play_text = render_play(play_rows, target=target, after=after)
    if dests is None:
        dests = [OPENPI_ROOT / "exp_vid"] + ([COSMOS_ROOT / "exp_vid"] if COSMOS_ROOT.is_dir() else [])
    written = []
    for dest in dests:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / FILENAME).write_text(text)
        (dest / PLAY_FILENAME).write_text(play_text)
        _write_csv(dest / "six_task_trials.csv", rows)
        _write_csv(dest / "play_trials.csv", play_rows)
        written += [dest / FILENAME, dest / PLAY_FILENAME]
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; only runs from this date on")
    parser.add_argument("--exclude", nargs="*", default=(), help="run folder names to leave out")
    parser.add_argument("--target", type=int, default=TARGET, help="trials per task per policy")
    parser.add_argument("--dest", type=Path, nargs="*", default=None, help="folders to write to (default: exp_vid of both checkouts)")
    args = parser.parse_args()
    for path in update(runs_dir=RUNS_DIR, dests=args.dest, target=args.target, since=args.since, exclude=tuple(args.exclude)):
        print(f"Wrote {path}")
    if args.dest is None:
        print((OPENPI_ROOT / "exp_vid" / FILENAME).read_text())
        print((OPENPI_ROOT / "exp_vid" / PLAY_FILENAME).read_text())


if __name__ == "__main__":
    main()
