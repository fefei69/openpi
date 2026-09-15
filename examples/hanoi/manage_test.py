import datetime
import json
import subprocess

import pytest

from examples.hanoi import manage
from examples.hanoi import scheduling


class FakeSlurm:
    account = "torch_pr_595_tandon_advanced"

    def __init__(self, root):
        self.delegate = scheduling.Slurm(root)
        self.jobs = {}
        self.submissions = []
        self.cancelled = []
        self.timeout_after_accepting = False

    def arguments(self, *args, **kwargs):
        return self.delegate.arguments(*args, **kwargs)

    def run(self, args):
        job_id = str(100 + len(self.jobs))
        name = next(value.partition("=")[2] for value in args if value.startswith("--job-name="))
        self.jobs[job_id] = {"JobName": name, "Account": self.account, "JobState": "PENDING", "Restarts": "0"}
        self.submissions.append(args)
        if self.timeout_after_accepting:
            raise subprocess.TimeoutExpired(args, 45)
        return subprocess.CompletedProcess(args, 0, job_id, "")

    def get_job(self, job_id):
        return dict(self.jobs[job_id])

    def get_jobs(self, job_ids):
        return {job_id: self.get_job(job_id) for job_id in job_ids}

    def find_submission(self, name, since):
        return [job_id for job_id, job in self.jobs.items() if job["JobName"] == name]

    def cancel_pending(self, job_id, *, expected_name):
        assert self.jobs[job_id]["JobName"] == expected_name
        if self.jobs[job_id]["JobState"] != "PENDING":
            return False
        self.jobs[job_id]["JobState"] = "CANCELLED"
        self.cancelled.append(job_id)
        return True

    def cancel_owned(self, job_id, *, expected_name):
        assert self.jobs[job_id]["JobName"] == expected_name
        self.cancelled.append(job_id)


@pytest.fixture
def manager(tmp_path):
    return manage.Manager(tmp_path, "unit", submit=True, slurm=FakeSlurm(tmp_path), now=lambda: 10000.0)


def add_pilot(manager, profile="h100_1"):
    path = manager.run_dir / f"{profile}.json"
    path.write_text(
        json.dumps({"startup_seconds": 60, "step_seconds_p95": 1, "checkpoint_save_seconds": 10, "save_interval": 250})
    )
    manager.state["qualified"][profile] = str(path)


def test_ambiguous_submission_reconciles_after_manager_restart_without_duplicate(manager):
    manager.slurm.timeout_after_accepting = True
    job = manager.submit_job("qualify", "h100_1", "h100_1")
    assert job["status"] == "AMBIGUOUS"
    assert job["job_id"] is None
    restarted = manage.Manager(manager.root, "unit", submit=True, slurm=manager.slurm)
    with pytest.raises(ValueError, match="already active"):
        restarted.submit_job("qualify", "h100_1", "h100_1")
    restarted.refresh()
    assert restarted.active()[0]["job_id"] == "100"
    assert len(manager.slurm.submissions) == 1


def test_qualification_allocation_allows_full_save_restore_and_inference(manager):
    job = manager.submit_job("qualify", "h100_1", "h100_1")
    assert "--time=60" in job["arguments"]


def test_slow_periodic_saves_are_included_in_planned_allocation_budget(manager):
    add_pilot(manager)
    pilot_path = manager.run_dir / "h100_1.json"
    pilot = json.loads(pilot_path.read_text())
    pilot["checkpoint_save_seconds"] = 900
    pilot_path.write_text(json.dumps(pilot))
    job = manager.submit_job("train", "aaaa_to_cccc", "h100_1")
    assert job["allocation_limit"] == 5
    assert "--time=720" in job["arguments"]
    assert manager.remaining_seconds("h100_1", "aaaa_to_cccc") == 147360


def test_unknown_submission_stays_active_and_reserves_gpu_budget(manager):
    job = manager.submit_job("qualify", "a100_2", "a100_2")
    job.update(job_id=None, status="SUBMITTING")
    manager.slurm.jobs.clear()
    manager.refresh()
    assert manager.active() == [job]
    with pytest.raises(ValueError, match="two-GPU budget"):
        manager.submit_job("qualify", "h100_1", "h100_1")


def test_training_and_evaluation_share_the_serial_storage_budget(manager):
    add_pilot(manager)
    manager.submit_job("train", "aaaa_to_cccc", "h100_1")
    with pytest.raises(ValueError, match="one training/evaluation"):
        manager.submit_job("evaluate", "cccc_to_aaaa", "h100_1")


def test_retry_submission_is_reconciled_if_old_terminal_handling_was_interrupted(manager):
    job = manager.submit_job("qualify", "h100_1", "h100_1")
    job["status"] = "NODE_FAIL"
    manager.handle_terminal(job)
    assert len(manager.slurm.submissions) == 2
    assert manager.active()[0]["failure_retries"] == 1
    job["handled"] = False  # Simulate loss of the final parent bookkeeping write.
    manager.handle_terminal(job)
    assert len(manager.slurm.submissions) == 2
    assert job["handled"]


def test_two_no_progress_infrastructure_failures_stop_automatic_retries(manager):
    job = manager.submit_job("qualify", "h100_1", "h100_1")
    job["status"] = "NODE_FAIL"
    manager.handle_terminal(job)
    continuation = manager.active()[0]
    continuation["status"] = "NODE_FAIL"
    manager.handle_terminal(continuation)
    assert manager.state["attention"]
    assert len(manager.slurm.submissions) == 2


def test_auto_requeues_count_toward_no_progress_budget(manager):
    job = manager.submit_job("qualify", "h100_1", "h100_1")
    manager.slurm.jobs[job["job_id"]].update(JobState="PENDING", Restarts="2")
    manager.refresh()
    assert job["failure_retries"] == 2
    assert job["no_progress"] == 2
    assert manager.state["attention"]
    assert manager.slurm.cancelled == [job["job_id"]]


def test_preempted_requeue_is_not_replaced_after_an_arbitrary_delay(manager):
    job = manager.submit_job("qualify", "h100_1", "h100_1")
    manager.slurm.jobs[job["job_id"]].update(JobState="PREEMPTED", Requeue="1")
    manager.refresh()
    manager.now = lambda: 10700.0
    manager.refresh()
    assert job["status"] == "AWAITING_REQUEUE"
    manager.handle_terminal(job)
    assert len(manager.slurm.submissions) == 1
    manager.now = lambda: 13700.0
    manager.refresh()
    assert manager.state["attention"]
    assert len(manager.slurm.submissions) == 1


def test_completed_job_without_required_artifact_is_not_marked_handled(manager):
    job = manager.submit_job("qualify", "h100_1", "h100_1")
    job["status"] = "COMPLETED"
    with pytest.raises(FileNotFoundError):
        manager.handle_terminal(job)
    assert not job["handled"]
    assert not manager.state["qualified"]


def test_progressing_timeout_is_a_planned_allocation_not_an_infrastructure_failure(manager, monkeypatch):
    add_pilot(manager)
    job = manager.submit_job("train", "aaaa_to_cccc", "h100_1")
    job["status"] = "TIMEOUT"
    monkeypatch.setattr(manager, "progress", lambda _: 100)
    manager.handle_terminal(job)
    resumed = manager.active()[0]
    assert resumed["allocations"] == 2
    assert resumed["failure_retries"] == 0
    assert resumed["start_progress"] == 100


def test_pending_replacement_recovers_cancel_before_submission_crash(manager):
    add_pilot(manager)
    job = manager.submit_job("train", "aaaa_to_cccc", "h100_1")
    job["replacement_intent"] = {"profile": "h100_1", "preemptible": True, "time": 10000.0}
    manager.persist()
    manager.slurm.jobs[job["job_id"]]["JobState"] = "CANCELLED"
    restarted = manage.Manager(manager.root, "unit", submit=True, slurm=manager.slurm)
    job = restarted.state["jobs"][0]
    restarted.refresh()
    restarted.advance_replacement(job)
    assert job["status"] == "REPLACED"
    assert len(manager.slurm.submissions) == 2
    assert restarted.active()[0]["allocations"] == 1
    assert restarted.active()[0]["replacement_times"] == [10000.0]
    restarted.advance_replacement(job)
    assert len(manager.slurm.submissions) == 2


def test_replacement_withdraws_if_job_has_started(manager):
    add_pilot(manager)
    job = manager.submit_job("train", "aaaa_to_cccc", "h100_1")
    job["replacement_intent"] = {"profile": "h100_1", "preemptible": True, "time": 10000.0}
    manager.slurm.jobs[job["job_id"]]["JobState"] = "RUNNING"
    manager.advance_replacement(job)
    assert job["status"] == "RUNNING"
    assert "replacement_intent" not in job
    assert len(manager.slurm.submissions) == 1
    assert not manager.slurm.cancelled


def test_only_committed_full_checkpoints_count_as_resume_progress(tmp_path):
    for step, metadata in ((1, {}), (2, {"commit_timestamp_nsecs": 123})):
        directory = tmp_path / str(step)
        (directory / "params").mkdir(parents=True)
        (directory / "train_state").mkdir()
        (directory / "_CHECKPOINT_METADATA").write_text(json.dumps(metadata))
    (tmp_path / "3.orbax-checkpoint-tmp").mkdir()
    assert manage.complete_step(tmp_path) == 2


def optional_pilot(manager, *, waited=7200, reason="QOSGrpGRES", state="PENDING", qualified=True, estimate=None):
    if qualified:
        add_pilot(manager)
    job = manager.submit_job("qualify", "a100_2", "a100_2")
    observed = manager.slurm.jobs[job["job_id"]]
    observed.update(
        JobState=state,
        Reason=reason,
        EligibleTime=datetime.datetime.fromtimestamp(manager.now() - waited, datetime.UTC).isoformat(),
        StartTime="Unknown"
        if estimate is None
        else datetime.datetime.fromtimestamp(estimate, datetime.UTC).isoformat(),
    )
    job.update(status=state, observed=observed)
    return job


@pytest.mark.parametrize(
    ("waited", "reason", "state", "qualified", "estimate", "deferred"),
    [
        (7199, "QOSGrpGRES", "PENDING", True, None, False),
        (7200, "QOSGrpGRES", "PENDING", True, None, True),
        (7200, "Resources", "PENDING", True, 10001, True),
        (1800, "Priority", "PENDING", True, 18000, True),
        (1799, "Priority", "PENDING", True, 18000, False),
        (7200, "Resources", "RUNNING", True, None, False),
        (7200, "JobHeldUser", "PENDING", True, None, False),
        (7200, "Dependency", "PENDING", True, None, False),
        (7200, "Resources", "PENDING", False, None, False),
    ],
)
def test_optional_pilot_wait_is_bounded_without_canceling_required_or_running_work(
    manager, waited, reason, state, qualified, estimate, deferred
):
    job = optional_pilot(manager, waited=waited, reason=reason, state=state, qualified=qualified, estimate=estimate)
    manager.defer_optional_qualification(job)
    assert bool(manager.slurm.cancelled) == deferred
    assert bool(manager.state["deferred_qualifications"]) == deferred
    assert not manager.state["unsupported"]
    if deferred:
        assert job["status"] == "CANCELLED"
        assert job["handled"]
        assert not manager.active()


def test_optional_deferral_recovers_cancel_before_persist_crash(manager, monkeypatch):
    job = optional_pilot(manager)

    def interrupted_cancel(job_id, *, expected_name):
        assert manager.slurm.jobs[job_id]["JobName"] == expected_name
        manager.slurm.jobs[job_id]["JobState"] = "CANCELLED"
        raise subprocess.TimeoutExpired("scancel", 45)

    monkeypatch.setattr(manager.slurm, "cancel_pending", interrupted_cancel)
    with pytest.raises(subprocess.TimeoutExpired):
        manager.defer_optional_qualification(job)
    restarted = manage.Manager(manager.root, "unit", submit=True, slurm=manager.slurm)
    restarted.refresh()
    recovered = restarted.state["jobs"][0]
    restarted.advance_qualification_deferral(recovered)
    restarted.handle_terminal(recovered)
    assert recovered["qualification_deferred"]
    assert recovered["handled"]
    assert "a100_2" in restarted.state["deferred_qualifications"]
    assert not restarted.state["attention"]
    assert not restarted.active()
    assert len(manager.slurm.submissions) == 1


def test_optional_deferral_withdraws_on_pending_to_running_race(manager, monkeypatch):
    job = optional_pilot(manager)

    def started(job_id, *, expected_name):
        assert manager.slurm.jobs[job_id]["JobName"] == expected_name
        manager.slurm.jobs[job_id]["JobState"] = "RUNNING"
        return False

    monkeypatch.setattr(manager.slurm, "cancel_pending", started)
    manager.defer_optional_qualification(job)
    manager.advance_qualification_deferral(job)
    assert job["status"] == "RUNNING"
    assert "qualification_deferral_intent" not in job
    assert not manager.state["deferred_qualifications"]
    assert not manager.slurm.cancelled


def test_storage_shortfall_waits_but_invalid_quota_still_requires_diagnosis(manager, monkeypatch):
    add_pilot(manager)

    def shortfall(*args, **kwargs):
        raise ValueError("Scratch quota has 1.0 GB available; forecast needs 2.0 GB")

    monkeypatch.setattr(manage.storage, "validate", shortfall)
    assert not manager.storage_ready("h100_1")
    assert not manager.state["attention"]
    monkeypatch.setattr(manage.storage, "validate", lambda *args, **kwargs: {"passed": True})
    assert manager.storage_ready("h100_1")

    def invalid(*args, **kwargs):
        raise ValueError("Could not parse scratch user quota")

    monkeypatch.setattr(manage.storage, "validate", invalid)
    with pytest.raises(ValueError, match="Could not parse"):
        manager.storage_ready("h100_1")


def test_deferred_optional_work_is_not_recreated_and_storage_precedes_submission(manager, monkeypatch):
    from examples.hanoi import dataset
    from examples.hanoi import qualify

    add_pilot(manager)
    pilot_path = manager.run_dir / "h100_1.json"
    pilot = json.loads(pilot_path.read_text())
    pilot["training_code_sha256"] = "fixed"
    pilot_path.write_text(json.dumps(pilot))
    manager.state["deferred_qualifications"]["a100_2"] = {"job_id": "old"}
    monkeypatch.setattr(manager, "adopt_preparation", lambda: None)
    monkeypatch.setattr(qualify, "require_validated_data", dict)
    monkeypatch.setattr(dataset, "training_code_identity", lambda: "fixed")

    def probe(profile, *, kind, task):
        assert profile == "h100_1"
        assert kind == "train"
        return manager.now()

    monkeypatch.setattr(manager, "probe", probe)
    monkeypatch.setattr(manager, "storage_ready", lambda profile: False)
    manager.tick()
    assert not manager.slurm.submissions
    assert not manager.state["attention"]
    monkeypatch.setattr(manager, "storage_ready", lambda profile: True)
    manager.tick()
    assert len(manager.slurm.submissions) == 1
    assert manager.active()[0]["kind"] == "train"
    assert manager.active()[0]["profile"] == "h100_1"


@pytest.mark.parametrize("prior_state", ["none", "placement", "checkpoint", "qualified_restore"])
def test_queue_replacement_changes_mesh_only_for_fresh_or_explicitly_qualified_restore(
    manager, monkeypatch, prior_state
):
    add_pilot(manager)
    add_pilot(manager, "a100_2")
    job = manager.submit_job("train", "aaaa_to_cccc", "h100_1")
    observed = manager.slurm.jobs[job["job_id"]]
    observed.update(Reason="QOSGrpGRES", EligibleTime="1970-01-01T00:00:00+00:00", StartTime="Unknown")
    job["observed"] = observed
    directory = manager.root / "checkpoints/pi05_hanoi_aaaa_to_cccc" / manager.exp_name
    if prior_state == "placement":
        directory.mkdir(parents=True)
        (directory / "hanoi_placement.json").write_text('{"fsdp_devices": 1}')
    elif prior_state in ("checkpoint", "qualified_restore"):
        checkpoint = directory / "125"
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "train_state").mkdir()
        (checkpoint / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 123}')
    if prior_state == "qualified_restore":
        path = manager.run_dir / "a100_2.json"
        pilot = json.loads(path.read_text())
        pilot["cross_mesh_restore_from"] = [1]
        path.write_text(json.dumps(pilot))

    def probe(profile, *, kind, task, preemptible):
        return manager.now() + 60 if profile == "a100_2" else None

    monkeypatch.setattr(manager, "probe", probe)
    manager.replace_long_wait(job)
    if prior_state in ("none", "qualified_restore"):
        assert job["status"] == "REPLACED"
        assert manager.active()[0]["profile"] == "a100_2"
        assert manager.active()[0]["allocations"] == 1
        assert manager.slurm.cancelled == [job["job_id"]]
        assert len(manager.slurm.submissions) == 2
    else:
        assert job["status"] == "PENDING"
        assert not manager.slurm.cancelled
        assert len(manager.slurm.submissions) == 1
