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
    assert (measured["sd"], measured["low"], measured["high"], measured["q1"], measured["median"], measured["q3"]) == (50.0, 0.0, 100.0, 25.0, 50.0, 75.0)
    reported = vals[("Expert data", 3)]
    assert (reported["value"], reported["kind"], reported["trials"], reported["tasks"], reported["complete"], reported["sd"]) == (100.0, "reported", None, 2, False, None)
    assert vals[("Non-expert data", 3)]["complete"]  # all six task cases reported
    assert (vals[("Expert data", 2)]["kind"], vals[("Expert data", 2)]["value"], vals[("Non-expert data", 2)]["value"]) == ("placeholder", 5.0, 2.0)
    assert vals[("Non-expert data", 0)]["kind"] == "placeholder"  # no play trials in these rows
    note = plot_comparison.footnote(vals, 3)
    assert "standard error of the mean" in note and "standard deviation" not in note and "Trials: expert Cosmos Policy 3." in note
    assert "mean \u00b1 1 standard deviation" in plot_comparison.footnote(vals, 3, "sd")
    assert "first to third quartile" in plot_comparison.footnote(vals, 3, "quartiles")
    assert "Ours: 2 of 6 task cases so far (expert data)." in note and "* not yet all 6 task cases with 3 trials each." in note
    assert "placeholder values" in note


def test_a_single_trial_has_no_spread_box():
    vals = plot_comparison.values([row("pi05_play", "CCCC_to_BBBB", 0.2)], best=3, reported={}, placeholders={})
    assert vals[("Non-expert data", 1)]["value"] == 20.0 and vals[("Non-expert data", 1)]["sd"] is None


def test_both_spread_styles_render(tmp_path):
    rows = [row("cosmos_multitask", task, x) for task in ("AAAA_to_CCCC", "CCCC_to_AAAA") for x in (1.0, 0.6, 0.2)]
    vals = plot_comparison.values(rows, best=3, reported={"Non-expert data": {3: (100.0, 2)}}, placeholders={"Expert data": {2: 5, 3: 100}})
    assert plot_comparison.footnote(vals, 3) == plot_comparison.footnote(vals, 3, "se")  # the standard error is the default
    for spread in ("sd", "se", "quartiles"):
        fig = plot_comparison.draw(vals, plot_comparison.footnote(vals, 3, spread), spread)
        fig.savefig(tmp_path / f"{spread}.png")
        assert (tmp_path / f"{spread}.png").stat().st_size > 10_000
