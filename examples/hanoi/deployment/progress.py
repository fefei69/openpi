"""Task progress for the four-ring Tower of Hanoi, from the gripper events of a run.

Progress is not binary. From the sequence of grasps (jaw closes) and releases (jaw opens) at each
peg, ``BoardTracker`` follows the ring stacks from the standard start A = [4, 3, 2, 1] and reports:

* ``moves``: every ring move made, with its legality;
* ``optimal_prefix``: how many leading moves match the optimal 15-move solution;
* ``remaining``: the fewest legal moves from the current board to the goal (breadth-first search over
  the 81 board states), so ``progress = (15 - remaining) / 15`` credits any legal path, and null
  moves or detours cost exactly what they cost;
* ``solved``: the board is C = [4, 3, 2, 1].

The same tracker runs live in the dense client (to stop a trial when the puzzle is solved) and
offline in publishing and the trial report (from ``events.jsonl``).
"""

from collections import deque
import dataclasses

import numpy as np

RINGS = 4
PEGS = ("A", "B", "C")
START = {"A": [4, 3, 2, 1], "B": [], "C": []}
GOAL = {"A": [], "B": [], "C": [4, 3, 2, 1]}
LEVELS_MM = np.array([57.1, 67.8, 77.5, 87.7])


def peg_of(y_m: float) -> str:
    return "A" if y_m < -0.02 else ("B" if y_m < 0.05 else "C")


def optimal_solution(n: int = RINGS, source="A", target="C", spare="B") -> list:
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


def remaining_moves(stacks: dict) -> int | None:
    """Fewest legal moves from ``stacks`` to the goal; None if the board is not a valid Hanoi state."""
    rings = sorted(r for p in PEGS for r in stacks[p])
    if rings != list(range(1, RINGS + 1)) or any(list(stacks[p]) != sorted(stacks[p], reverse=True) for p in PEGS):
        return None
    start, goal = _key(stacks), _key(GOAL)
    if start == goal:
        return 0
    seen = {start}
    queue = deque([(stacks, 0)])
    while queue:
        board, depth = queue.popleft()
        for nxt in _legal_moves(board):
            k = _key(nxt)
            if k == goal:
                return depth + 1
            if k not in seen:
                seen.add(k)
                queue.append((nxt, depth + 1))
    return None


@dataclasses.dataclass
class BoardTracker:
    stacks: dict = dataclasses.field(default_factory=lambda: {p: list(v) for p, v in START.items()})
    held: tuple | None = None  # (ring, source peg) while a ring is in the gripper
    moves: list = dataclasses.field(default_factory=list)
    grasps: list = dataclasses.field(default_factory=list)
    uncertain: bool = False  # a ring was lost mid-carry; the board is no longer known

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
        if ring is not None:
            self.stacks[peg].append(ring)
        move = {"t_s": t_s, "ring": ring, "from": src, "to": peg, "legal": legal,
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
        return not self.uncertain and self.held is None and self.stacks == GOAL

    @property
    def optimal_prefix(self) -> int:
        n = 0
        for move, (ring, src, dst) in zip(self.moves, OPTIMAL):
            if move["legal"] and (move["ring"], move["from"], move["to"]) == (ring, src, dst):
                n += 1
            else:
                break
        return n

    def report(self) -> dict:
        remaining = None if self.uncertain else remaining_moves(self.stacks if self.held is None else self._with_held_back())
        total = len(OPTIMAL)
        return {
            "moves_completed": len(self.moves),
            "legal_moves": sum(1 for m in self.moves if m["legal"]),
            "all_legal": all(m["legal"] for m in self.moves),
            "optimal_prefix": self.optimal_prefix,
            "remaining_moves": remaining,
            "progress": None if remaining is None else round((total - remaining) / total, 3),
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


def reconstruct(events: list) -> BoardTracker:
    """Replay a run's logged gripper commands (and a missed-grasp stop) through a tracker."""
    tracker = BoardTracker()
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
