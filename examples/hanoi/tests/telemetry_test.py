import json

import pytest

from examples.hanoi.pipeline import telemetry


def test_device_identity_uses_this_process_and_deduplicates_gpu_contexts():
    output = "111, GPU-other\n222, GPU-b\n222, GPU-a\n222, GPU-a\n"
    assert telemetry.process_gpu_uuids(output, 222, 2) == ["GPU-a", "GPU-b"]
    with pytest.raises(ValueError, match="JAX allocated 1"):
        telemetry.process_gpu_uuids(output, 222, 1)


def test_checkpoint_telemetry_ignores_incomplete_saves(tmp_path):
    for step, committed in ((100, 123_000_000_000), (200, None)):
        path = tmp_path / str(step) / "_CHECKPOINT_METADATA"
        path.parent.mkdir()
        path.write_text(json.dumps({"commit_timestamp_nsecs": committed}))
    assert telemetry.checkpoint_progress(tmp_path)["checkpoint_step"] == 100
