"""Contract six: six directed tower moves, the instruction string as the only task signal, dense v5 labels."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import hanoi_dense_policy
from openpi.policies import hanoi_policy

HORIZON = 16
FRAMESKIP = hanoi_dense_policy.FRAMESKIP
EXECUTION_PREFIX = hanoi_dense_policy.EXECUTION_PREFIX
REFERENCE_RATE_HZ = hanoi_dense_policy.REFERENCE_RATE_HZ
SAMPLING_STEPS = hanoi_dense_policy.SAMPLING_STEPS
REPO_ID = "local/hanoi_multitask_v6_raw"
ASSET_ID = "local/hanoi_multitask_v6"
CONFIG_NAME = "pi05_hanoi_multitask_v6"
DATA_ROOT = "data/hanoi/multitask_v6_pi05"
DATASETS_DIR = "/scratch/cw5167/datasets"


@dataclasses.dataclass(frozen=True)
class Task:
    index: int
    direction: str
    start: str
    goal: str
    first_move_target: str
    file: str

    @property
    def prompt(self) -> str:
        position = {"A": "the left peg", "B": "the middle peg", "C": "the right peg"}[self.goal]
        return (
            f"Move all four rings from peg {self.start} to peg {self.goal} following Tower of Hanoi rules. "
            f"The goal is peg {self.goal}, {position}."
        )


# Same order, wording and files as the Cosmos six-task build (cosmos_policy/datasets/hanoi_multitask_data.py).
TASKS = (
    Task(0, "AAAA_to_CCCC", "A", "C", "B", "hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5"),
    Task(1, "CCCC_to_AAAA", "C", "A", "B", "hanoi_wm_roundtrip_20260925_171442_CCCC_to_AAAA.h5"),
    Task(2, "AAAA_to_BBBB", "A", "B", "C", "hanoi_wm_roundtrip_20260926_011840_AAAA_to_BBBB.h5"),
    Task(3, "BBBB_to_AAAA", "B", "A", "C", "hanoi_wm_roundtrip_20260926_011840_BBBB_to_AAAA.h5"),
    Task(4, "BBBB_to_CCCC", "B", "C", "A", "hanoi_wm_roundtrip_20260926_011840_BBBB_to_CCCC.h5"),
    Task(5, "CCCC_to_BBBB", "C", "B", "A", "hanoi_wm_roundtrip_20260926_011840_CCCC_to_BBBB.h5"),
)
PROMPTS = tuple(task.prompt for task in TASKS)
TASK_BY_PROMPT = {task.prompt: task for task in TASKS}
TASK_BY_DIRECTION = {task.direction: task for task in TASKS}
SAME_START_PAIRS = tuple((a.index, b.index) for a in TASKS for b in TASKS if a.index < b.index and a.start == b.start)
REVERSE_OF = {task.index: TASK_BY_DIRECTION[f"{task.goal * 4}_to_{task.start * 4}"].index for task in TASKS}
CONTRACT = {
    **{k: v for k, v in hanoi_dense_policy.CONTRACT.items() if k not in ("prompt", "recording")},
    "version": 6,
    "action_horizon": HORIZON,
    "execution_prefix": EXECUTION_PREFIX,
    "conditioning": "instruction string only; one of six verbatim prompts is required on every request, no default task",
    "tasks": [
        {"index": t.index, "direction": t.direction, "start_peg": t.start, "goal_peg": t.goal, "prompt": t.prompt}
        for t in TASKS
    ],
    "recordings": ["hanoi_wm_roundtrip_20260925_171442", "hanoi_wm_roundtrip_20260926_011840"],
    "episode_split_per_file": {"train": [0, 1, 2, 3, 4, 5, 6, 7], "val": [8], "test": [9]},
}


def resolve_task(prompt) -> Task:
    if not isinstance(prompt, str):
        prompt = str(np.asarray(prompt).item()) if np.asarray(prompt).shape == () else prompt
    if prompt not in TASK_BY_PROMPT:
        raise ValueError("Hanoi multitask requests must carry one of the six task prompts verbatim; no default task")
    return TASK_BY_PROMPT[prompt]


@dataclasses.dataclass(frozen=True)
class HanoiMultitaskInputs(transforms.DataTransformFn):
    """Dense inputs plus a mandatory verbatim task prompt; the resolved task index rides along for the reply."""

    horizon: int = HORIZON

    def __call__(self, data: dict) -> dict:
        if "prompt" not in data:
            raise ValueError("Hanoi multitask requests must carry one of the six task prompts verbatim; no default task")
        task = resolve_task(data["prompt"])
        result = hanoi_dense_policy.HanoiDenseInputs(horizon=self.horizon)(data)
        result["prompt"] = task.prompt
        result["task"] = np.asarray(task.index, dtype=np.int32)
        return result


@dataclasses.dataclass(frozen=True)
class HanoiMultitaskOutputs(transforms.DataTransformFn):
    """Absolute reference poses with the executed timing, echoing the task the request resolved to."""

    horizon: int = HORIZON

    def __call__(self, data: dict) -> dict:
        result = hanoi_dense_policy.HanoiDenseOutputs(horizon=self.horizon)(data)
        if "task" in data:
            index = int(np.asarray(data["task"]).reshape(-1)[0])
            result["task"] = index
            result["task_direction"] = TASKS[index].direction
        return result


def serving_metadata(train_config, checkpoint_dir) -> dict:
    metadata = hanoi_dense_policy.serving_metadata(train_config, checkpoint_dir)
    block = metadata.pop("hanoi_dense")
    block["contract"] = CONTRACT
    block["prompts"] = list(PROMPTS)
    block["prompt"] = None
    metadata["hanoi_multitask"] = block
    return metadata
