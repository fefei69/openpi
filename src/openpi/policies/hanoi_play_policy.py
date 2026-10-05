"""Contract seven (play): language-goal hindsight labels over the play recording; the goal sentence is the only goal signal.

Rows, goals, labels and sentences follow the Cosmos build `cosmos_policy/datasets/hanoi_play_data.py` (contract
`hanoi_play_k5_cosmos_v1`): the CIDM manifest's rows, one goal board per row drawn once by Cosmos (taken from its
archive, never re-drawn here), the dense v5 chunk cut at the last row of the goal move, and one fixed sentence per
board. Serving requires one of the 81 sentences verbatim and echoes the goal board it resolved to.
"""

import collections
import dataclasses
import hashlib
import itertools

import numpy as np

from openpi import transforms
from openpi.policies import hanoi_dense_policy

HORIZON = 16
FRAMESKIP = hanoi_dense_policy.FRAMESKIP
EXECUTION_PREFIX = hanoi_dense_policy.EXECUTION_PREFIX
REFERENCE_RATE_HZ = hanoi_dense_policy.REFERENCE_RATE_HZ
SAMPLING_STEPS = hanoi_dense_policy.SAMPLING_STEPS
REPO_ID = "local/hanoi_play_k5_raw"
ASSET_ID = "local/hanoi_play_k5"
CONFIG_NAME = "pi05_hanoi_play_k5"
DATA_ROOT = "data/hanoi/play_k5_pi05"
DATASETS_DIR = "/scratch/cw5167/datasets"
PLAY_FILE = "hanoi_wm_20260924_210743.h5"
EXPERT_FILE = "hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5"
FILES = (PLAY_FILE, EXPERT_FILE)  # file index 0 (play) and 1 (expert), as in the Cosmos archive
MANIFEST = f"{DATASETS_DIR}/dataset_manifest_v1/manifest.json"
COSMOS_ARCHIVE = "/scratch/cw5167/workspace/cosmos-policy/data/hanoi_cosmos/play_k5"
COSMOS_CONTRACT = "hanoi_play_k5_cosmos_v1"
HORIZON_CAP_MOVES = 5
EXPERT_EPISODE_OFFSET = 1000  # global episode id of expert episode e is 1000 + e
PEGS = "ABC"
RINGS = 4
BOARDS = tuple("".join(b) for b in itertools.product(PEGS, repeat=RINGS))  # peg per ring, smallest ring first
BOARD_INDEX = {board: index for index, board in enumerate(BOARDS)}
FULL_STACKS = tuple(peg * RINGS for peg in PEGS)
STAGE_NAMES = {
    0: "unknown", 1: "open", 2: "approach_source", 3: "descend_source", 4: "grasp", 5: "lift", 6: "transit",
    7: "insert_target", 8: "release", 9: "retreat", 10: "return_initial_xy", 11: "return_initial_z", 12: "hold",
}
DECISION_STAGES = (2, 6)  # approach_source (which peg to pick from) and transit (where to place)
SEGMENT_KINDS = ("walk", "crop", "clip")
PROMPT_TEMPLATE = (
    '"Goal: peg A <contents>, peg B <contents>, peg C <contents>." with rings numbered 1 (smallest) to 4; '
    '"is empty", "holds ring N", or "holds rings a, b and c" in ascending order'
)


def prompt_for_board(board: str) -> str:
    """One sentence per board, pegs always in the order A, B, C (the Cosmos template, character for character)."""
    if len(board) != RINGS or any(peg not in PEGS for peg in board):
        raise ValueError(f"Not a board: {board!r}")
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


PROMPTS = tuple(prompt_for_board(board) for board in BOARDS)
BOARD_BY_PROMPT = {prompt: index for index, prompt in enumerate(PROMPTS)}
PROMPTS_SHA256 = hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest()


def legal_moves(board: str):
    """Boards reachable by one legal move: only the top (smallest) ring of a peg moves, never onto a smaller ring."""
    for source in PEGS:
        rings = [i for i in range(RINGS) if board[i] == source]
        if not rings:
            continue
        top = rings[0]
        for target in PEGS:
            if target == source:
                continue
            target_rings = [i for i in range(RINGS) if board[i] == target]
            if target_rings and target_rings[0] < top:
                continue
            yield board[:top] + target + board[top + 1 :]


def _distances() -> np.ndarray:
    table = np.zeros((len(BOARDS), len(BOARDS)), np.int8)
    for start in BOARDS:
        seen = {start: 0}
        queue = collections.deque([start])
        while queue:
            board = queue.popleft()
            for other in legal_moves(board):
                if other not in seen:
                    seen[other] = seen[board] + 1
                    queue.append(other)
        for board, d in seen.items():
            table[BOARD_INDEX[start], BOARD_INDEX[board]] = d
    return table


DISTANCE = _distances()  # graph distance between boards; diameter 15


def board_string(row) -> str:
    """The recording's `board` row (peg index per ring) as a board string."""
    return "".join(PEGS[int(peg)] for peg in row)


CONTRACT = {
    **{k: v for k, v in hanoi_dense_policy.CONTRACT.items() if k not in ("prompt", "recording")},
    "version": 7,
    "action_horizon": HORIZON,
    "execution_prefix": EXECUTION_PREFIX,
    "conditioning": "the goal board's sentence is the only goal signal; one of the 81 sentences is required verbatim on "
    "every request, no default goal, no goal image, no task id",
    "prompt_template": PROMPT_TEMPLATE,
    "prompts_sha256": PROMPTS_SHA256,
    "goal_sentences": len(PROMPTS),
    "recordings": ["hanoi_wm_20260924_210743", "hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC"],
    "manifest": "dataset_manifest_v1 (78 whole walks + 18 one-move crops + 4 expert clips train; 10 walks val; 10 walks test)",
    "observation_rule": "every manifest row whose image is neither stale nor repeated and whose telemetry, proprio and joints are finite",
    "goal_rule": "one goal per row, drawn once by the Cosmos build (seed 195) uniformly over the row's move and the next 4 moves of "
    "its walk, clipped at the walk end; a crop or clip has its single move; taken from the Cosmos archive, not re-drawn",
    "label_rule": "slot j is row t + 3j: XYZ = reference_pose[row, 0:3], jaw = action_abs[row, 3]; slots past the last row of "
    "the goal move repeat that row and are padded (trained as hold targets, excluded from metrics)",
    "horizon_cap_moves": HORIZON_CAP_MOVES,
    "cosmos_contract": COSMOS_CONTRACT,
    "cosmos_execution_prefix": 8,  # the Cosmos v7 deployment contract executes 8 slots; this policy keeps the six-task prefix 3
}


def resolve_board(prompt) -> int:
    if not isinstance(prompt, str):
        prompt = str(np.asarray(prompt).item()) if np.asarray(prompt).shape == () else prompt
    if prompt not in BOARD_BY_PROMPT:
        raise ValueError("Hanoi play requests must carry the goal board's sentence verbatim (one of 81); no default goal")
    return BOARD_BY_PROMPT[prompt]


@dataclasses.dataclass(frozen=True)
class HanoiPlayInputs(transforms.DataTransformFn):
    """Dense inputs plus a mandatory verbatim goal sentence; the resolved board index rides along for the reply."""

    horizon: int = HORIZON

    def __call__(self, data: dict) -> dict:
        if "prompt" not in data:
            raise ValueError("Hanoi play requests must carry the goal board's sentence verbatim (one of 81); no default goal")
        board = resolve_board(data["prompt"])
        result = hanoi_dense_policy.HanoiDenseInputs(horizon=self.horizon)(data)
        result["prompt"] = PROMPTS[board]
        result["goal_board"] = np.asarray(board, dtype=np.int32)
        return result


@dataclasses.dataclass(frozen=True)
class HanoiPlayOutputs(transforms.DataTransformFn):
    """Absolute reference poses with the executed timing, echoing the goal board the request resolved to."""

    horizon: int = HORIZON

    def __call__(self, data: dict) -> dict:
        result = hanoi_dense_policy.HanoiDenseOutputs(horizon=self.horizon)(data)
        if "goal_board" in data:
            index = int(np.asarray(data["goal_board"]).reshape(-1)[0])
            result["goal_board"] = index
            result["goal_board_string"] = BOARDS[index]
        return result


def serving_metadata(train_config, checkpoint_dir) -> dict:
    metadata = hanoi_dense_policy.serving_metadata(train_config, checkpoint_dir)
    block = metadata.pop("hanoi_dense")
    block["contract"] = CONTRACT
    block["prompts"] = list(PROMPTS)
    block["boards"] = list(BOARDS)
    block["prompts_sha256"] = PROMPTS_SHA256
    block["prompt"] = None
    metadata["hanoi_play"] = block
    return metadata
