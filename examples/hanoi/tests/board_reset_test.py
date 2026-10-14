import numpy as np
import pytest

from examples.hanoi.deployment import board_reset, progress
from examples.hanoi.deployment.hardware import MissedGraspError


class FakeArm:
    """Arrives wherever it is sent; the jaw reads 20 mm after a close and 34 mm after an open."""

    def __init__(self, start_xyz, grasp_stroke=0.020):
        self.xyz, self.jaw, self.grasp_stroke, self.commands = np.array(start_xyz, float), 0.034, grasp_stroke, []

    def read(self, *, require_orientation=True):
        return np.r_[self.xyz, 0.0, 0.0, 0.0, self.jaw]

    def dispatch(self, command, xyz_min, xyz_max, limits):
        self.commands.append(command)
        if command.kind == "cartesian":
            self.xyz = command.sample(np.array([1.0]))[0]
        else:
            self.jaw = 0.034 if command.jaw_open else self.grasp_stroke


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(board_reset.time, "sleep", lambda s: None)


BOUNDS = (np.array([0.40, -0.10, 0.04]), np.array([0.55, 0.12, 0.25]))


def test_reset_plan_is_the_shortest_legal_path_back():
    assert board_reset.reset_moves(progress.stacks_from_board("BAAA"), "AAAA") == [(1, "B", "A")]
    assert len(board_reset.reset_moves(progress.all_on("C"), "AAAA")) == 15
    assert board_reset.reset_moves(progress.all_on("A"), "AAAA") == []
    assert board_reset.reset_moves({"A": [4, 3, 1, 2], "B": [], "C": []}, "AAAA") is None  # illegal stacking: by hand
    assert board_reset.grasp_z(1) == pytest.approx(0.0571) and board_reset.grasp_z(4) == pytest.approx(0.0877)


def test_one_move_reset_runs_the_pick_and_place_script(no_sleep):
    arm = FakeArm(board_reset.PEG_HOVER_XYZ["B"])
    report = board_reset.execute_reset(arm, BOUNDS, progress.stacks_from_board("BAAA"), "AAAA")
    assert report["status" if "status" in report else "planned_moves"] in ("reset", 1)
    assert report["moves"] == [{"ring": 1, "from": "B", "to": "A"}] and progress.board_string(report["board"]) == "AAAA"
    kinds = [(c.kind, c.jaw_open) for c in arm.commands]
    assert kinds == [("cartesian", None)] * 2 + [("gripper", False)] + [("cartesian", None)] * 3 + [("gripper", True), ("cartesian", None)]
    ends = [c.sample(np.array([1.0]))[0] for c in arm.commands if c.kind == "cartesian"]
    b, a = board_reset.PEG_HOVER_XYZ["B"], board_reset.PEG_HOVER_XYZ["A"]
    np.testing.assert_allclose(ends[1], [b[0], b[1], 0.0571], atol=1e-9)         # grasp ring 1 alone on B at level 1
    np.testing.assert_allclose(ends[2], [b[0], b[1], board_reset.HOVER_Z], atol=1e-9)
    np.testing.assert_allclose(ends[3], [a[0], a[1], board_reset.HOVER_Z], atol=1e-9)  # travel at hover height
    np.testing.assert_allclose(ends[4], [a[0], a[1], board_reset.RELEASE_Z], atol=1e-9)
    assert all(leg["error_mm"] < 1e-6 for leg in report["legs"] if leg["event"] == "reset_leg")


def test_reset_grasps_the_top_ring_at_its_level(no_sleep):
    arm = FakeArm(board_reset.PEG_HOVER_XYZ["C"])
    board = progress.stacks_from_board("CCAA")  # rings 1 and 2 on C: ring 1 sits at level 2
    report = board_reset.execute_reset(arm, BOUNDS, board, "AAAA")
    grasps = [c.sample(np.array([1.0]))[0][2] for i, c in enumerate(arm.commands) if c.kind == "cartesian" and i + 1 < len(arm.commands)
              and arm.commands[i + 1].kind == "gripper" and arm.commands[i + 1].jaw_open is False]
    assert [round(z, 4) for z in grasps] == [0.0678, 0.0571, 0.0571]  # ring 1 from level 2 on C, ring 2 alone on C, ring 1 alone on B
    assert progress.board_string(report["board"]) == "AAAA" and len(report["moves"]) == 3


def test_a_missed_grasp_stops_the_reset_with_the_board_so_far(no_sleep):
    arm = FakeArm(board_reset.PEG_HOVER_XYZ["B"], grasp_stroke=0.004)
    with pytest.raises(MissedGraspError):
        board_reset.execute_reset(arm, BOUNDS, progress.stacks_from_board("BAAA"), "AAAA")


def test_a_leg_that_does_not_arrive_stops_the_reset(no_sleep):
    class StuckArm(FakeArm):
        def dispatch(self, command, *args):
            self.commands.append(command)  # never moves

    with pytest.raises(RuntimeError, match="from its target"):
        board_reset.execute_reset(StuckArm(board_reset.PEG_HOVER_XYZ["B"]), BOUNDS, progress.stacks_from_board("BAAA"), "AAAA")
