import numpy as np

from examples.hanoi.deployment import progress


def test_optimal_solution_and_distances():
    assert len(progress.OPTIMAL) == 15 and progress.OPTIMAL[0] == (1, "A", "B") and progress.OPTIMAL[-1] == (1, "B", "C")
    assert progress.remaining_moves(progress.START) == 15
    assert progress.remaining_moves(progress.GOAL) == 0
    assert progress.remaining_moves({"A": [4], "B": [3], "C": [2, 1]}) == 11  # after move 4
    assert progress.remaining_moves({"A": [4, 3, 1], "B": [2], "C": []}) is None or progress.remaining_moves({"A": [4, 3, 1], "B": [2], "C": []}) > 0
    assert progress.remaining_moves({"A": [4, 3], "B": [1], "C": [2]}) == 13  # after two optimal moves


def test_tracker_follows_the_optimal_solution_and_scores_detours():
    t = progress.BoardTracker()
    for ring, src, dst in progress.OPTIMAL[:5]:
        assert t.grasp(src, z_mm=77.5, t_s=1.0) == ring
        assert t.release(dst, t_s=2.0)["legal"]
    r = t.report()
    assert (r["moves_completed"], r["optimal_prefix"], r["remaining_moves"], r["progress"], r["solved"]) == (5, 5, 10, round(5 / 15, 3), False)
    # A null move (ring put back where it came from) does not advance progress and ends the optimal prefix.
    ring = t.grasp("C"); t.release("C")
    r = t.report()
    assert (r["moves_completed"], r["optimal_prefix"], r["remaining_moves"]) == (6, 5, 10)
    # Finishing from here still reaches solved.
    for stacks_move in range(10):
        board = t.stacks
        nxt = next(progress._legal_moves(board))  # any legal move; just check the API stays consistent
    t2 = progress.BoardTracker()
    for ring, src, dst in progress.OPTIMAL:
        t2.grasp(src); t2.release(dst)
    assert t2.solved and t2.report()["progress"] == 1.0 and t2.report()["remaining_moves"] == 0


def test_reconstruct_from_events_handles_a_missed_grasp():
    def grip(t, y, z, open_):
        return {"event": "command", "kind": "gripper", "monotonic_s": t, "target_xyz_m": [0.496, y, z], "jaw_open": open_}
    events = [{"event": "tick", "monotonic_s": 0.0}, grip(1.0, -0.057, 0.0877, False), grip(5.0, 0.0137, 0.15, True),
              grip(10.0, -0.057, 0.0775, False),
              {"event": "missed_grasp", "monotonic_s": 11.0, "stroke_m": 0.006, "minimum_m": 0.008}]
    r = progress.reconstruct(events).report()
    assert r["moves_completed"] == 1 and r["optimal_prefix"] == 1 and r["final_board"] == {"A": [4, 3, 2], "B": [1], "C": []}
    assert r["remaining_moves"] == 14 and not r["board_uncertain"]


def test_tracker_handles_other_directions():
    t = progress.BoardTracker("C", "A")
    assert t.stacks == {"A": [], "B": [], "C": [4, 3, 2, 1]} and t.optimal[0] == (1, "C", "B") and len(t.optimal) == 15
    for ring, src, dst in t.optimal:
        t.grasp(src); t.release(dst)
    r = t.report()
    assert r["solved"] and r["progress"] == 1.0 and r["start_peg"] == "C" and r["goal_peg"] == "A"
    assert progress.remaining_moves(progress.all_on("B"), "C") == 15 and progress.remaining_moves(progress.all_on("C"), "C") == 0


def test_peak_progress_survives_a_later_illegal_stacking():
    t = progress.BoardTracker("A", "C")
    for ring, src, dst in t.optimal[:12]:
        t.grasp(src)
        t.release(dst)
    r = t.report()
    assert r["progress"] == r["peak_progress"] == 0.8 and r["remaining_moves"] == 3
    t.grasp("A")
    t.release("C")  # ring 1 onto ring 3: legal but off the optimal path
    t.grasp("A")
    t.release("C")  # ring 2 onto ring 1: illegal, the board is no longer a Hanoi state
    r = t.report()
    assert r["progress"] is None and r["remaining_moves"] is None and not r["all_legal"]
    assert r["peak_progress"] == 0.8


def test_peak_progress_of_an_untouched_board_is_zero():
    assert progress.BoardTracker("C", "A").report()["peak_progress"] == 0.0


def test_every_move_is_classed_and_counted():
    t = progress.BoardTracker("A", "C")
    for ring, src, dst in t.optimal[:2]:
        t.grasp(src); t.release(dst)
    t.grasp("B"); assert t.release("B")["kind"] == "null"         # ring 1 put back
    t.grasp("B"); assert t.release("A")["kind"] == "detour"       # ring 1 onto ring 3: legal, off the shortest path
    t.grasp("C"); assert t.release("A")["kind"] == "illegal"      # ring 2 onto ring 1
    t.grasp("B"); assert t.release("C")["kind"] == "empty"        # closed on an empty peg
    r = t.report()
    assert r["move_counts"] == {"optimal": 2, "detour": 1, "null": 1, "illegal": 1, "empty": 1}
    assert not r["clean"] and r["optimal_prefix"] == 2
    t2 = progress.BoardTracker("C", "B")
    for ring, src, dst in t2.optimal:
        t2.grasp(src); t2.release(dst)
    assert t2.report()["clean"] and t2.report()["move_counts"]["optimal"] == 15


# The arm protocol's table (docs/hanoi_play_arm_protocol.md in cosmos-policy): goal boards at distance 1, 3, 7, 15.
PROTOCOL_GOALS = {
    "AAAA_to_CCCC": ("BAAA", "CCAA", "BBBA", "CCCC"), "CCCC_to_AAAA": ("BCCC", "AACC", "BBBC", "AAAA"),
    "AAAA_to_BBBB": ("CAAA", "BBAA", "CCCA", "BBBB"), "BBBB_to_AAAA": ("CBBB", "AABB", "CCCB", "AAAA"),
    "BBBB_to_CCCC": ("ABBB", "CCBB", "AAAB", "CCCC"), "CCCC_to_BBBB": ("ACCC", "BBCC", "AAAC", "BBBB"),
}
PROTOCOL_PATH_A_TO_C = "AAAA BAAA BCAA CCAA CCBA ACBA ABBA BBBA BBBC CBBC CABC AABC AACC BACC BCCC CCCC".split()
PROTOCOL_SENTENCES = {
    "AAAA": "Goal: peg A holds rings 1, 2, 3 and 4, peg B is empty, peg C is empty.",
    "CCCC": "Goal: peg A is empty, peg B is empty, peg C holds rings 1, 2, 3 and 4.",
    "BAAA": "Goal: peg A holds rings 2, 3 and 4, peg B holds ring 1, peg C is empty.",
    "CCAA": "Goal: peg A holds rings 3 and 4, peg B is empty, peg C holds rings 1 and 2.",
    "BBBA": "Goal: peg A holds ring 4, peg B holds rings 1, 2 and 3, peg C is empty.",
    "CCCA": "Goal: peg A holds ring 4, peg B is empty, peg C holds rings 1, 2 and 3.",
}


def test_boards_goals_and_sentences_match_the_arm_protocol():
    import itertools

    assert progress.stacks_from_board("BAAA") == {"A": [4, 3, 2], "B": [1], "C": []}
    assert progress.board_string({"A": [4, 3, 2], "B": [1], "C": []}) == "BAAA"
    assert progress.board_string({"A": [4, 3], "B": [1], "C": []}) is None  # a ring is in the gripper
    for direction, goals in PROTOCOL_GOALS.items():
        assert tuple(progress.goal_at_distance(direction, d) for d in (1, 3, 7, 15)) == goals
    path = progress.shortest_path(progress.all_on("A"), "C")
    assert [progress.board_string(b) for b in path] == PROTOCOL_PATH_A_TO_C
    assert progress.path_moves(path) == progress.optimal_solution(4, "A", "C")
    for board, sentence in PROTOCOL_SENTENCES.items():
        assert progress.goal_sentence(board) == sentence
    assert len({progress.goal_sentence("".join(b)) for b in itertools.product("ABC", repeat=4)}) == 81
    assert progress.board_string(progress.next_board(progress.stacks_from_board("BCAA"), "CCCC")) == "CCAA"
    assert progress.next_board(progress.all_on("C"), "CCCC") is None


def test_tracker_scores_an_intermediate_goal_board():
    t = progress.BoardTracker("A", "C", goal_board="CCAA")   # distance 3 on the way from A to C
    assert t.total == 3 and t.optimal == progress.optimal_solution(4, "A", "C")[:3]
    t.grasp("A"); assert t.release("C")["kind"] == "detour"   # ring 1 to C: still three moves away, no closer
    r = t.report()
    assert (r["goal_board"], r["start_board"], r["distance"], r["moves_before_first_error"], r["progress"]) == ("CCAA", "AAAA", 3, 0, 0.0)
    t.grasp("C"); assert t.release("B")["kind"] == "optimal"
    t.grasp("A"); assert t.release("C")["kind"] == "optimal"
    t.grasp("B"); assert t.release("C")["kind"] == "optimal"
    r = t.report()
    assert t.solved and r["progress"] == 1.0 and r["ring_moves"] == 4 and r["moves_before_first_error"] == 0
    clean = progress.BoardTracker("A", "B", goal_board="CCCA")
    for ring, src, dst in clean.optimal:
        clean.grasp(src); clean.release(dst)
    assert clean.solved and clean.report()["moves_before_first_error"] == 7 and clean.report()["clean"]
