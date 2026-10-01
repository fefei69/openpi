"""Task progress for the four-ring Tower of Hanoi, from the gripper events of a run.

Progress is not binary. From the sequence of grasps (jaw closes) and releases (jaw opens) at each
peg, ``BoardTracker`` follows the ring stacks from the standard start A = [4, 3, 2, 1] and reports:

* ``moves``: every ring move made, with its legality and kind (``optimal`` = on a shortest path to the goal,
  ``detour`` = legal but off it, ``null`` = put back on the same peg, ``illegal`` = a larger ring onto a
  smaller one, ``empty`` = closed on nothing), and ``move_counts`` per kind;
* ``optimal_prefix``: how many leading moves match the optimal 15-move solution;
* ``remaining``: the fewest legal moves from the current board to the goal (breadth-first search over
  the 81 board states), so ``progress = (15 - remaining) / 15`` credits any legal path, and null
  moves or detours cost exactly what they cost;
* ``peak_progress``: the best ``progress`` over the boards reached, so a run that later stacks a ring
  illegally (an unscorable board, ``progress`` None) still gets credit for how far it got;
* ``solved``: the board is the goal (C = [4, 3, 2, 1] by default);
* ``moves_before_first_error``: leading moves that each reduced the graph distance to the goal.

The goal is a full tower by default, or any of the 81 boards (``goal_board``, peg per ring with ring 1 the
smallest first, e.g. ``BAAA``), which is what the play-trained policies are asked for: ``goal_sentence`` gives a
board's trained sentence, ``goal_at_distance`` the board a number of moves along a task's shortest path, and
``next_board`` the board one move closer (the next-board protocol).

The same tracker runs live in the dense client (to stop a trial when the puzzle is solved) and
offline in publishing and the trial report (from ``events.jsonl``).
"""

from collections import deque
import dataclasses

import numpy as np

RINGS = 4
PEGS = ("A", "B", "C")
def all_on(peg: str) -> dict:
    return {p: ([4, 3, 2, 1] if p == peg else []) for p in PEGS}


START = all_on("A")
GOAL = all_on("C")
LEVELS_MM = np.array([57.1, 67.8, 77.5, 87.7])
# How each ring move is classed: on a shortest path to the goal, legal but off it, put back on the same peg,
# a larger ring onto a smaller one, or a close-and-open that held nothing.
MOVE_KINDS = ("optimal", "detour", "null", "illegal", "empty")


def peg_of(y_m: float) -> str:
    return "A" if y_m < -0.02 else ("B" if y_m < 0.05 else "C")


DIRECTIONS = ("AAAA_to_CCCC", "CCCC_to_AAAA", "AAAA_to_BBBB", "BBBB_to_AAAA", "BBBB_to_CCCC", "CCCC_to_BBBB")


def stacks_from_board(board: str) -> dict:
    """``BAAA`` (peg per ring, ring 1 the smallest first) -> stacks, each bottom to top."""
    if not isinstance(board, str) or len(board) != RINGS or any(peg not in PEGS for peg in board):
        raise ValueError(f"Not a board: {board!r}")
    return {peg: [ring for ring in range(RINGS, 0, -1) if board[ring - 1] == peg] for peg in PEGS}


def board_string(stacks: dict) -> str | None:
    """Stacks -> ``BAAA``; None unless each of the four rings is on exactly one peg."""
    where = [(ring, peg) for peg in PEGS for ring in stacks[peg]]
    if sorted(ring for ring, _ in where) != list(range(1, RINGS + 1)):
        return None
    return "".join(peg for _, peg in sorted(where))


def goal_stacks(goal) -> dict:
    """A goal given as stacks, a peg letter (the full tower there) or a board string."""
    if isinstance(goal, dict):
        return goal
    return all_on(goal) if len(goal) == 1 else stacks_from_board(goal)


def goal_sentence(board: str) -> str:
    """The play policies' trained sentence for a goal board; must match the server's byte for byte."""
    stacks_from_board(board)
    clauses = []
    for peg in PEGS:
        rings = [str(i + 1) for i in range(RINGS) if board[i] == peg]
        if not rings:
            clauses.append(f"peg {peg} is empty")
        elif len(rings) == 1:
            clauses.append(f"peg {peg} holds ring {rings[0]}")
        else:
            clauses.append(f"peg {peg} holds rings " + ", ".join(rings[:-1]) + " and " + rings[-1])
    return "Goal: " + ", ".join(clauses) + "."


def spare_of(source: str, target: str) -> str:
    return next(p for p in PEGS if p not in (source, target))


def optimal_solution(n: int = RINGS, source="A", target="C", spare=None) -> list:
    spare = spare or spare_of(source, target)
    if n == 0:
        return []
    return optimal_solution(n - 1, source, spare, target) + [(n, source, target)] + optimal_solution(n - 1, spare, target, source)


OPTIMAL = optimal_solution()


def _key(stacks: dict) -> tuple:
    return tuple(tuple(stacks[p]) for p in PEGS)


def _legal_moves(stacks: dict):
    for src in PEGS:
        if not stacks[src]:
            continue
        ring = stacks[src][-1]
        for dst in PEGS:
            if dst != src and (not stacks[dst] or stacks[dst][-1] > ring):
                nxt = {p: list(v) for p, v in stacks.items()}
                nxt[src].pop()
                nxt[dst].append(ring)
                yield nxt


def shortest_path(stacks: dict, goal="C") -> list | None:
    """Boards from ``stacks`` to the goal inclusive along a shortest legal path; None if ``stacks`` is not a valid Hanoi state.

    The goal is a peg letter (the full tower there), a board string or stacks. Between two full towers, and to the
    boards the arm protocol uses, the shortest path is unique; elsewhere the first one found is returned.
    """
    rings = sorted(r for p in PEGS for r in stacks[p])
    if rings != list(range(1, RINGS + 1)) or any(list(stacks[p]) != sorted(stacks[p], reverse=True) for p in PEGS):
        return None
    first = {p: list(v) for p, v in stacks.items()}
    start, target = _key(first), _key(goal_stacks(goal))
    parent = {start: None}
    boards = {start: first}
    queue = deque([first])
    while queue and target not in parent:
        board = queue.popleft()
        for nxt in _legal_moves(board):
            k = _key(nxt)
            if k not in parent:
                parent[k], boards[k] = _key(board), nxt
                queue.append(nxt)
    if target not in parent:
        return None
    path, k = [], target
    while k is not None:
        path.append(boards[k])
        k = parent[k]
    return path[::-1]


def remaining_moves(stacks: dict, goal="C") -> int | None:
    """Fewest legal moves from ``stacks`` to the goal (peg letter, board string or stacks); None if the board is not a valid Hanoi state."""
    path = shortest_path(stacks, goal)
    return None if path is None else len(path) - 1


def next_board(stacks: dict, goal="C") -> dict | None:
    """The board one move closer to the goal; None at the goal or from an invalid board."""
    path = shortest_path(stacks, goal)
    return path[1] if path and len(path) > 1 else None


def path_moves(path: list) -> list:
    """(ring, source, target) for each step of a board path."""
    moves = []
    for before, after in zip(path, path[1:]):
        src = next(p for p in PEGS if len(before[p]) > len(after[p]))
        dst = next(p for p in PEGS if len(before[p]) < len(after[p]))
        moves.append((before[src][-1], src, dst))
    return moves


def goal_at_distance(direction: str, distance: int) -> str:
    """The board ``distance`` moves along a task's shortest path, e.g. (``AAAA_to_CCCC``, 3) -> ``CCAA``."""
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown task {direction!r}; choose from {DIRECTIONS}")
    path = shortest_path(all_on(direction[0]), direction[-1])
    if not 1 <= distance < len(path):
        raise ValueError(f"Distance must be between 1 and {len(path) - 1}")
    return board_string(path[distance])


@dataclasses.dataclass
class BoardTracker:
    start_peg: str = "A"
    goal_peg: str = "C"
    stacks: dict = None
    held: tuple | None = None  # (ring, source peg) while a ring is in the gripper
    moves: list = dataclasses.field(default_factory=list)
    grasps: list = dataclasses.field(default_factory=list)
    uncertain: bool = False  # a ring was lost mid-carry; the board is no longer known
    goal_board: str | None = None  # any of the 81 boards; default: the full tower on ``goal_peg``
    start_board: str | None = None  # default: the full tower on ``start_peg``

    def __post_init__(self):
        if self.stacks is None:
            self.stacks = stacks_from_board(self.start_board) if self.start_board else all_on(self.start_peg)
        self.initial = {p: list(v) for p, v in self.stacks.items()}
        self.goal = stacks_from_board(self.goal_board) if self.goal_board else all_on(self.goal_peg)
        path = shortest_path(self.initial, self.goal)
        if path is None or len(path) < 2:
            raise ValueError("The start board must be a valid Hanoi state different from the goal")
        self.total = len(path) - 1
        self.optimal = path_moves(path)

    def grasp(self, peg: str, z_mm: float | None = None, t_s: float | None = None):
        ring = self.stacks[peg][-1] if self.stacks[peg] else None
        level_error = None
        if z_mm is not None:
            level_error = float(z_mm - LEVELS_MM[np.abs(LEVELS_MM - z_mm).argmin()])
        self.grasps.append({"t_s": t_s, "peg": peg, "ring": ring, "z_mm": z_mm, "level_error_mm": level_error})
        if ring is not None:
            self.stacks[peg].pop()
        self.held = (ring, peg)
        return ring

    def release(self, peg: str, t_s: float | None = None) -> dict:
        ring, src = self.held if self.held else (None, None)
        legal = ring is not None and (not self.stacks[peg] or self.stacks[peg][-1] > ring)
        before = remaining_moves(self._with_held_back(), self.goal)
        if ring is not None:
            self.stacks[peg].append(ring)
        after = remaining_moves(self.stacks, self.goal)
        if ring is None:
            kind = "empty"  # closed on nothing, then opened: not a ring move
        elif not legal:
            kind = "illegal"
        elif src == peg:
            kind = "null"
        elif before is not None and after is not None and after == before - 1:
            kind = "optimal"
        else:
            kind = "detour"  # legal, but not on a shortest path to the goal
        move = {"t_s": t_s, "ring": ring, "from": src, "to": peg, "legal": legal, "kind": kind,
                "board": {p: list(v) for p, v in self.stacks.items()}}
        self.moves.append(move)
        self.held = None
        return move

    def lost_ring(self):
        """A held ring slipped somewhere unknown (missed-grasp watchdog during a carry)."""
        if self.held and self.held[0] is not None:
            self.uncertain = True
        self.held = None

    @property
    def solved(self) -> bool:
        return not self.uncertain and self.held is None and self.stacks == self.goal

    @property
    def ring_moves(self) -> int:
        """Rings released on a peg (a close-and-open that held nothing is not a move)."""
        return sum(1 for m in self.moves if m["kind"] != "empty")

    @property
    def moves_before_first_error(self) -> int:
        """Leading moves that each reduced the graph distance to the goal."""
        n = 0
        for move in self.moves:
            if move["kind"] != "optimal":
                break
            n += 1
        return n

    @property
    def optimal_prefix(self) -> int:
        n = 0
        for move, (ring, src, dst) in zip(self.moves, self.optimal):
            if move["legal"] and (move["ring"], move["from"], move["to"]) == (ring, src, dst):
                n += 1
            else:
                break
        return n

    def report(self) -> dict:
        remaining = None if self.uncertain else remaining_moves(self.stacks if self.held is None else self._with_held_back(), self.goal)
        total = self.total

        def fraction(left):  # a board farther from the goal than the start scores zero, not negative
            return None if left is None else round(max(0.0, (total - left) / total), 3)
        # Best board reached along the way: an illegal stacking later on makes the final board unscorable,
        # but the run still got as far as it got.
        boards = [self.initial] + [m["board"] for m in self.moves]
        scored = [remaining_moves(b, self.goal) for b in boards]
        peak = min((r for r in scored if r is not None), default=None)
        return {
            "start_peg": self.start_peg,
            "goal_peg": self.goal_peg,
            "start_board": board_string(self.initial),
            "goal_board": board_string(self.goal),
            "distance": total,
            "ring_moves": self.ring_moves,
            "moves_before_first_error": self.moves_before_first_error,
            "moves_completed": len(self.moves),
            "legal_moves": sum(1 for m in self.moves if m["legal"]),
            "all_legal": all(m["legal"] for m in self.moves),
            "optimal_prefix": self.optimal_prefix,
            "move_counts": {k: sum(1 for m in self.moves if m["kind"] == k) for k in MOVE_KINDS},
            "clean": bool(self.moves) and all(m["kind"] == "optimal" for m in self.moves),
            "remaining_moves": remaining,
            "progress": fraction(remaining),
            "peak_progress": fraction(peak),
            "solved": self.solved,
            "board_uncertain": self.uncertain,
            "final_board": {p: list(v) for p, v in self.stacks.items()},
            "held_at_end": self.held,
            "moves": self.moves,
            "grasps": self.grasps,
        }

    def _with_held_back(self) -> dict:
        board = {p: list(v) for p, v in self.stacks.items()}
        if self.held and self.held[0] is not None:
            board[self.held[1]].append(self.held[0])
        return board


def reconstruct(events: list, start_peg: str = "A", goal_peg: str = "C", goal_board: str | None = None) -> BoardTracker:
    """Replay a run's logged gripper commands (and a missed-grasp stop) through a tracker."""
    tracker = BoardTracker(start_peg, goal_peg, goal_board=goal_board)
    t0 = events[0]["monotonic_s"]
    for e in events:
        if e["event"] == "command" and e["kind"] == "gripper":
            x = np.asarray(e["target_xyz_m"]) * 1000
            peg = peg_of(x[1] / 1000)
            t = round(e["monotonic_s"] - t0, 1)
            if e["jaw_open"]:
                tracker.release(peg, t_s=t)
            else:
                tracker.grasp(peg, z_mm=round(float(x[2]), 1), t_s=t)
        elif e["event"] == "missed_grasp" and tracker.held is not None:
            # Closed on nothing (ring stays on its peg) or slipped mid-carry (ring position unknown).
            ring, src = tracker.held
            if e.get("stroke_m", 0) <= 0.008 and ring is not None and not any(
                    c["event"] == "command" and c["kind"] == "cartesian" and c["monotonic_s"] > e["monotonic_s"] - 30 and c["monotonic_s"] < e["monotonic_s"]
                    and abs(c["target_xyz_m"][2] - 0.19) < 0.02 for c in events):
                tracker.stacks[src].append(ring)  # never left the peg
                tracker.held = None
            else:
                tracker.lost_ring()
    return tracker
