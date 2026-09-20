"""The bag recorder runs ros2 bag record as a session-leader subprocess and stops it with SIGINT."""
import os
import stat
import time

import pytest

from examples.hanoi.deployment import bag


@pytest.fixture
def fake_ros2(tmp_path, monkeypatch):
    script = tmp_path / "ros2"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# emulate ros2 bag record: create the output dir, write while alive, close cleanly on SIGINT\n"
        "out=''; while [[ $# -gt 0 ]]; do case $1 in -o) out=$2; shift;; esac; shift; done\n"
        "printf '%s\\n' \"$*\" > \"$out.args\" 2>/dev/null || true\n"
        "mkdir -p \"$out\"; echo header > \"$out/bag.mcap\"\n"
        "trap 'echo closed >> \"$out/bag.mcap\"; exit 0' INT\n"
        "while true; do sleep 0.1; done\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return script


def test_start_and_stop_record_topics_and_close_cleanly(tmp_path, fake_ros2):
    recorder = bag.start_bag(tmp_path / "camera_bag", ("/camera/camera/color/image_raw",), settle_s=0.3)
    assert recorder.process.poll() is None and (tmp_path / "camera_bag.log").exists()
    time.sleep(0.2)
    report = recorder.stop()
    assert report["outcome"] == "clean" and report["returncode"] == 0 and report["size_bytes"] > 0
    assert (tmp_path / "camera_bag" / "bag.mcap").read_text().splitlines() == ["header", "closed"]
    assert report["topics"] == ["/camera/camera/color/image_raw"] and report["duration_s"] > 0.4
    with pytest.raises(FileExistsError):
        bag.start_bag(tmp_path / "camera_bag", ("/x",), settle_s=0.0)


def test_missing_ros2_and_bad_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="ros2 is not on PATH"):
        bag.start_bag(tmp_path / "b", ("/x",))
    with pytest.raises(ValueError, match="preset"):
        bag.start_bag(tmp_path / "b", ("/x",), storage_preset="lz4")
    with pytest.raises(ValueError, match="topic"):
        bag.start_bag(tmp_path / "b", ())


def test_early_exit_is_reported(tmp_path, monkeypatch):
    script = tmp_path / "ros2"; script.write_text("#!/usr/bin/env bash\necho 'no such topic' >&2; exit 3\n"); script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(RuntimeError, match="exited immediately with code 3"):
        bag.start_bag(tmp_path / "b", ("/x",), settle_s=0.3)
