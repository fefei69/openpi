import json

from examples.hanoi.deployment import progress, scoreboard

PEG_Y = {"A": -0.057, "B": 0.014, "C": 0.089}


def write_run(root, name, family, task, moves, status="duration_reached", aborted=False):
    """A minimal run folder: summary.json plus gripper command events for the given (src, dst) moves."""
    run = root / name
    run.mkdir()
    start, goal = task[0], task[-1]
    events = [{"event": "task", "monotonic_s": 0.0}]
    t = 1.0
    for src, dst in moves:
        for peg, jaw_open in ((src, False), (dst, True)):
            events.append({"event": "command", "kind": "gripper", "monotonic_s": t, "tick": int(30 * t), "jaw_open": jaw_open,
                           "target_xyz_m": [0.496, PEG_Y[peg], 0.0775 if not jaw_open else 0.08]})
            t += 5.0
    if not aborted:
        events.append({"event": "command", "kind": "cartesian", "monotonic_s": t, "tick": int(30 * t), "target_xyz_m": [0.496, 0.0, 0.19]})
    events.append({"event": "tick", "monotonic_s": t + 1})
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (run / "summary.json").write_text(json.dumps({"status": status, "policy_family": family, "config_name": f"{family}_v6",
                                                  "task_direction": task, "start_peg": start, "goal_peg": goal, "brakes": 0}))
    return run


def test_scoreboard_counts_trials_per_task_and_policy_and_skips_aborted_runs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    solution = [(s, d) for _, s, d in progress.optimal_solution(4, "A", "C")]
    write_run(runs, "dense_live_1", "cosmos_multitask", "AAAA_to_CCCC", solution, status="task_solved")
    write_run(runs, "dense_live_2", "cosmos_multitask", "AAAA_to_CCCC", solution[:9], status="rejected_command")
    write_run(runs, "dense_live_3", "cosmos_multitask", "CCCC_to_AAAA", [], status="failed", aborted=True)
    # pi0.5: two optimal moves, a null move, then an illegal stacking (ring 2 onto ring 1) -> scored by its peak.
    write_run(runs, "dense_live_4", "pi05_multitask", "AAAA_to_CCCC", [("A", "B"), ("A", "C"), ("B", "B"), ("B", "A"), ("C", "A")],
              status="operator_stop")
    write_run(runs, "dense_live_5", "pi05_dense", "AAAA_to_CCCC", solution, status="task_solved")  # single-task: not in the campaign
    rows = scoreboard.collect(runs)
    assert [r["run"] for r in rows] == ["dense_live_1", "dense_live_2", "dense_live_3", "dense_live_4"]
    assert rows[3]["progress"] is None and rows[3]["peak_progress"] == round(2 / 15, 3)
    assert (rows[3]["optimal"], rows[3]["detour"], rows[3]["null"], rows[3]["illegal"]) == (2, 1, 1, 1)
    text = scoreboard.render(rows, target=3, after="dense_live_4")
    assert "| AAAA_to_CCCC | 2 of 3 | 1 (50%) | 80% | 1 of 3 | 0 (0%) | 13%* |" in text
    assert "| CCCC_to_AAAA | 0 of 3 |  |  | 0 of 3 |  |  |" in text
    assert "| All six | 2 of 18 | 1 (50%) | 80% | 1 of 18 | 0 (0%) | 13%* |" in text
    assert "| Cosmos | AAAA_to_CCCC | 2 | 24 | 24 | 0 | 0 | 0 | 2 |" in text
    assert "| pi0.5 | AAAA_to_CCCC | 1 | 5 | 2 | 1 | 1 | 1 | 0 |" in text
    assert "- Cosmos: AAAA_to_CCCC x1, CCCC_to_AAAA x3, AAAA_to_BBBB x3" in text
    assert "Aborted, not counted: dense_live_3" in text and "3 of 36 done" in text
    dest = tmp_path / "exp_vid"
    written = scoreboard.update(after="dense_live_4", runs_dir=runs, dests=[dest])
    assert written == [dest / "six_task_scoreboard.md"] and (dest / "six_task_trials.csv").exists()
    assert (dest / "six_task_scoreboard.md").read_text().splitlines()[3:] == text.splitlines()[3:]  # same but for the time stamp
