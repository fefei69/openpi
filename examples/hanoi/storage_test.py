import os

import pytest

from examples.hanoi import storage


def test_quota_uses_user_headroom_and_rounding_margin():
    reading = "/scratch $SCRATCH NO/YES 5.00TB/5.00M \x1b[0;31m4.50TB(90.02%)/505521(10.00%)\x1b[0m"
    assert storage.scratch_headroom(reading) == 490_000_000_000
    with pytest.raises(ValueError, match="authoritative quota"):
        storage.scratch_headroom("Filesystem 550TB available")


def test_unique_size_does_not_charge_hardlinks_twice(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "weights").write_bytes(b"model parameters")
    os.link(first / "weights", second / "weights")
    assert storage.unique_bytes([first, second]) == len(b"model parameters")


def test_forecast_includes_overlap_candidates_and_reserve():
    expected = 4 * 48_000_000_000 + 8 * 12_000_000_000 + 50 * 2**30
    assert storage.forecast(full_bytes=48_000_000_000, inference_bytes=12_000_000_000) == expected
    assert (
        storage.forecast(full_bytes=48_000_000_000, inference_bytes=12_000_000_000, occupied_bytes=48_000_000_000)
        == expected - 48_000_000_000
    )
    with pytest.raises(ValueError, match="internally consistent"):
        storage.forecast(full_bytes=1, inference_bytes=2)
