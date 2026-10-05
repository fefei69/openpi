from examples.hanoi.deployment import plot_comparison


def row(family, task, progress):
    return {"family": family, "task": task, "progress": progress, "peak_progress": progress, "aborted": False}


def test_values_prefer_measured_then_reported_then_placeholder():
    rows = [row("cosmos_multitask", "AAAA_to_CCCC", x) for x in (1.0, 0.5, 0.0)]
    vals = plot_comparison.values(rows, best=3, reported={"Expert data": {3: (100.0, 2)}, "Non-expert data": {3: (100.0, 6)}},
                                  placeholders={"Expert data": {2: 5}, "Non-expert data": {2: 2}})
    measured = vals[("Expert data", 0)]
    assert (measured["kind"], measured["value"], measured["trials"], measured["tasks"], measured["complete"]) == ("measured", 50.0, 3, 1, False)
    assert round(measured["error"], 2) == 28.87  # standard deviation 50 over the three trials, divided by sqrt(3)
    assert vals[("Expert data", 3)] == {"value": 100.0, "kind": "reported", "trials": None, "tasks": 2, "complete": False, "error": None}
    assert vals[("Non-expert data", 3)]["complete"]  # all six task cases reported
    assert (vals[("Expert data", 2)]["kind"], vals[("Expert data", 2)]["value"], vals[("Non-expert data", 2)]["value"]) == ("placeholder", 5.0, 2.0)
    assert vals[("Non-expert data", 0)]["kind"] == "placeholder"  # no play trials in these rows
    note = plot_comparison.footnote(vals, 3)
    assert "standard error" in note and "Trials: expert Cosmos Policy 3." in note
    assert "Ours: 2 of 6 task cases so far (expert data)." in note and "* not yet all 6 task cases with 3 trials each." in note
    assert "placeholder values" in note


def test_a_single_trial_has_no_error_bar():
    vals = plot_comparison.values([row("pi05_play", "CCCC_to_BBBB", 0.2)], best=3, reported={}, placeholders={})
    assert vals[("Non-expert data", 1)]["value"] == 20.0 and vals[("Non-expert data", 1)]["error"] is None
