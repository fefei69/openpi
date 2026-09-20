import os
import subprocess

import pytest

from examples.hanoi.pipeline import scheduling


@pytest.mark.parametrize("reason", ["Priority", "QOSGrpGRES"])
def test_pending_policy_requires_eligible_wait_and_meaningful_improvement(reason):
    pending = scheduling.Pending("PENDING", 0, 20000, reason)
    assert not scheduling.reassess(pending, 1799)
    assert scheduling.reassess(pending, 1800)
    kwargs = {
        "now": 1800,
        "alternative_start": 7200,
        "remaining_seconds": 3600,
        "alternative_remaining_seconds": 3600,
        "replacement_times": [],
    }
    assert scheduling.worthwhile_replacement(pending, **kwargs)
    assert not scheduling.worthwhile_replacement(pending, **{**kwargs, "alternative_remaining_seconds": 30000})
    assert not scheduling.worthwhile_replacement(pending, **{**kwargs, "replacement_times": [1700]})
    assert not scheduling.worthwhile_replacement(pending, **{**kwargs, "replacement_times": [0, 1]})


@pytest.mark.parametrize("reason", ["Priority", "QOSGrpGRES"])
def test_unknown_estimate_is_not_infinite_and_held_time_does_not_count(reason):
    assert not scheduling.reassess(scheduling.Pending("PENDING", 0, None, reason), 7199)
    assert scheduling.reassess(scheduling.Pending("PENDING", 0, None, reason), 7200)
    for state, excluded_reason in (("RUNNING", "None"), ("PENDING", "Dependency"), ("PENDING", "JobHeldUser")):
        assert not scheduling.reassess(scheduling.Pending(state, 0, None, excluded_reason), 10000)


@pytest.mark.parametrize("after_state", ["RUNNING", "CANCELLED"])
def test_pending_cancel_race_uses_controller_filter(tmp_path, after_state):
    commands = []
    states = iter(["PENDING", after_state])

    def runner(args, **kwargs):
        commands.append(args)
        if args[0] == "scontrol":
            output = f"JobId=123 JobName=hanoi-unit Account=torch_pr_595_tandon_advanced JobState={next(states)} "
            output += f"WorkDir={tmp_path} UserId=test({os.getuid()})"
            return subprocess.CompletedProcess(args, 0, output, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    slurm = scheduling.Slurm(tmp_path, runner=runner)
    assert slurm.cancel_pending("123", expected_name="hanoi-unit") == (after_state == "CANCELLED")
    assert ["scancel", "--ctld", "--state=PENDING", "123"] in commands


def test_unowned_job_is_never_canceled(tmp_path):
    def runner(args, **kwargs):
        assert args[0] == "scontrol"
        return subprocess.CompletedProcess(args, 0, "JobId=123 JobName=other JobState=PENDING", "")

    with pytest.raises(ValueError, match="outside this managed run"):
        scheduling.Slurm(tmp_path, runner=runner).cancel_pending("123", expected_name="hanoi-unit")


def test_test_only_is_not_submission_and_has_no_manual_partition_or_qos(tmp_path):
    def runner(args, **kwargs):
        assert "--test-only" in args
        assert not any(arg.startswith(("--partition", "--qos")) for arg in args)
        return subprocess.CompletedProcess(args, 0, "", "sbatch: Job 123 to start at 2026-09-14T05:00:00 a")

    slurm = scheduling.Slurm(tmp_path, runner=runner)
    args = slurm.arguments(
        scheduling.PROFILES[0],
        job_name="hanoi-unit",
        minutes=30,
        script="examples/hanoi/scripts/qualify.sbatch",
        script_args=[],
    )
    assert slurm.estimate(args) == scheduling.parse_time("2026-09-14T05:00:00")


def test_walltime_bounds():
    assert (
        scheduling.walltime_minutes(startup=10, step_seconds=1, remaining_steps=10, save_seconds=10, save_interval=250)
        == 60
    )
    assert (
        scheduling.walltime_minutes(
            startup=100, step_seconds=5, remaining_steps=30000, save_seconds=100, save_interval=50
        )
        == 720
    )


def test_periodic_checkpoint_overhead_affects_walltime():
    assert (
        scheduling.walltime_minutes(
            startup=0, step_seconds=1, remaining_steps=10000, save_seconds=900, save_interval=250
        )
        == 720
    )
    assert (
        scheduling.walltime_minutes(startup=0, step_seconds=1, remaining_steps=10000, save_seconds=1, save_interval=250)
        == 210
    )


def test_batched_queue_query_falls_back_to_accounting_for_forgotten_jobs(tmp_path):
    def runner(args, **kwargs):
        if args[0] == "squeue":
            assert args[args.index("--jobs") + 1] == "123,124"
            return subprocess.CompletedProcess(args, 1, "", "Invalid job id specified")
        if args[0] == "scontrol":
            return subprocess.CompletedProcess(args, 1, "", "Invalid job id specified")
        job_id = args[args.index("--jobs") + 1]
        return subprocess.CompletedProcess(
            args, 0, f"{job_id}|COMPLETED|0:0|torch_pr_595_tandon_advanced|hanoi-unit\n", ""
        )

    jobs = scheduling.Slurm(tmp_path, runner=runner).get_jobs(["123", "124"])
    assert set(jobs) == {"123", "124"}
    assert all(job["JobState"] == "COMPLETED" for job in jobs.values())


def test_offset_timestamps_preserve_their_timezone():
    assert scheduling.parse_time("2026-09-14T01:00:00-04:00") == scheduling.parse_time("2026-09-14T05:00:00+00:00")
