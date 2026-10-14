"""One-move expert set (hanoi_expert6_h1): the first move of the six tower tasks under the play goal-sentence interface.

Rows, goals, labels and sentences follow the Cosmos build `cosmos_policy/datasets/hanoi_first_move_data.py` (contract
`hanoi_expert6_h1_cosmos_v1`): move 0 of every expert episode (motion stages open to retreat), the six-task split per
file, the goal of every row is the board after the move, the dense v5 chunk cut at the move's last row. Every case is
demonstrated in training: this is the execution control for the play run, not a composition test. Transforms and
serving are the play ones (`hanoi_play_policy`): one of the 81 sentences verbatim, the six-task prompts are refused.
"""

from openpi.policies import hanoi_multitask_policy as multitask
from openpi.policies import hanoi_play_policy as play

HORIZON = play.HORIZON
FRAMESKIP = play.FRAMESKIP
REPO_ID = "local/hanoi_expert6_h1_raw"
ASSET_ID = "local/hanoi_expert6_h1"
CONFIG_NAME = "pi05_hanoi_expert6_h1"
DATA_ROOT = "data/hanoi/expert6_h1_pi05"
DATASETS_DIR = multitask.DATASETS_DIR
COSMOS_ARCHIVE = "/scratch/cw5167/workspace/cosmos-policy/data/hanoi_cosmos/expert6_h1"
VJEPA_ARCHIVE = "/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_first_move"
COSMOS_CONTRACT = "hanoi_expert6_h1_cosmos_v1"
SPLIT_EPISODES = {"train": tuple(range(8)), "val": (8,), "test": (9,)}
MOVE_STAGES = (1, 9)  # open .. retreat
EPISODE_ID = "task index * 10000 + expert episode * 100"
SEGMENT_KIND = 2  # 'clip', as in the Cosmos archive


def case_boards(task: multitask.Task) -> tuple[str, str]:
    """(board before, board after) of the task's first move: ring 1 off the full stack onto the spare peg."""
    return task.start * play.RINGS, task.first_move_target + task.start * (play.RINGS - 1)


CASES = tuple(
    {
        "task_index": task.index,
        "recording": task.direction,
        "file": task.file,
        "before": case_boards(task)[0],
        "after": case_boards(task)[1],
        "prompt": play.prompt_for_board(case_boards(task)[1]),
    }
    for task in multitask.TASKS
)
GOAL_BOARDS = tuple(play.BOARD_INDEX[case["after"]] for case in CASES)
CONTRACT = {
    **{k: v for k, v in play.CONTRACT.items() if k not in ("manifest", "observation_rule", "goal_rule", "recordings", "horizon_cap_moves", "cosmos_contract")},
    "dataset": "expert6_h1",
    "recordings": ["hanoi_wm_roundtrip_20260925_171442", "hanoi_wm_roundtrip_20260926_011840"],
    "episode_split_per_file": {split: list(episodes) for split, episodes in SPLIT_EPISODES.items()},
    "episode_rule": "move 0 of an expert episode, motion stages 1 to 9 (open to retreat), one contiguous block of rows",
    "observation_rule": "every row of the first move whose image is neither stale nor repeated (telemetry, proprio and joints are finite throughout)",
    "goal_rule": "the goal of every row is the board after the move, read at the move's last usable row; one move, nothing sampled",
    "horizon_cap_moves": 1,
    "cases": [{k: case[k] for k in ("recording", "before", "after", "prompt")} for case in CASES],
    "scope": "every case is demonstrated in training (8 times); execution control for the play run, not a composition test",
    "cosmos_contract": COSMOS_CONTRACT,
}
