"""Latest-observation inference worker and timestamped, bounded action buffer.

Only the worker touches the WebSocket; only the control loop touches the robot.
The wire format is the existing OpenPI protocol. No LeRobot runtime is required.
"""

import copy
import dataclasses
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from openpi_client import hanoi
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from examples.hanoi.deployment.execution import Command
from examples.hanoi.deployment.execution import PolicyExecutor
from examples.hanoi.deployment.execution import ReferenceExecutor


@dataclasses.dataclass(frozen=True)
class Observation:
    tick: int
    captured_at: float
    image_received_at: float
    data: dict
    generation: int = 0
    velocity_override: dict | None = None

    @property
    def image_age_s(self) -> float:
        return self.captured_at - self.image_received_at


@dataclasses.dataclass(frozen=True)
class Prediction:
    observation: Observation
    actions: np.ndarray
    received_at: float
    inference_s: float
    request_id: int | None = None


class InferenceRecorder:
    """Persist actual requests before inference, and replies even if later discarded.

    Only the inference thread writes these files. Each input archive is complete
    before it is renamed, so interruption cannot corrupt earlier observations.
    """

    def __init__(self, run_dir: Path):
        self.inputs = run_dir / "inference_inputs"
        self.inputs.mkdir(exist_ok=False)
        self.path = run_dir / "inferences.jsonl"
        self.path.touch(exist_ok=False)

    def write(self, event: str, request_id: int, **fields):
        with self.path.open("a") as stream:
            stream.write(
                json.dumps({"event": event, "request_id": request_id, "monotonic_s": time.monotonic(), **fields}) + "\n"
            )

    def save_input(self, request_id: int, observation: Observation):
        path = self.inputs / f"{request_id:06d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **observation.data)
        temporary.replace(path)
        self.write(
            "inference_request",
            request_id,
            input_file=f"inference_inputs/{path.name}",
            observation_tick=observation.tick,
            generation=observation.generation,
            observation_captured_at_s=observation.captured_at,
            image_received_at_s=observation.image_received_at,
            image_age_s=observation.image_age_s,
            state=np.asarray(observation.data["observation/state"]).tolist(),
            prompt=observation.data["prompt"],
            velocity_override=observation.velocity_override,
        )


class WebSocketPolicy:
    """OpenPI transport with bounded connection/receive/shutdown times.

    The stock client retries connection forever and has no receive timeout or
    public close operation, which would make shutdown of a robot worker hang.
    """

    def __init__(self, uri: str, *, timeout_s: float = 1.0, warmup_timeout_s: float = 90.0):
        self.timeout_s = timeout_s
        self.warmup_timeout_s = warmup_timeout_s
        self.first = True
        self.ws = connect(uri, compression=None, max_size=16 * 1024 * 1024, open_timeout=5, close_timeout=0.2)
        try:
            self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=5))
            for key, expected in hanoi.CONTRACT.items():
                if self.metadata.get(key) != expected:
                    raise ValueError(f"Hanoi server contract mismatch: {key}")
        except BaseException:
            self.ws.close()
            raise

    def infer(self, observation: dict) -> dict:
        self.ws.send(msgpack_numpy.packb(observation))
        reply = self.ws.recv(timeout=self.warmup_timeout_s if self.first else self.timeout_s)
        self.first = False
        if isinstance(reply, str):
            raise RuntimeError(f"Policy server failed: {reply}")
        return msgpack_numpy.unpackb(reply)

    def close(self):
        self.ws.close()


class InferenceWorker:
    """One in-flight request, one replaceable pending observation, one result.

    Reset invalidates even an inference already running on the server. A slow
    model cannot accumulate old observations or block the control thread.
    """

    def __init__(self, policy: Any, *, record_dir: Path | None = None, horizon: int = 63):
        self.policy = policy
        self.horizon = horizon
        try:
            self.recorder = InferenceRecorder(record_dir) if record_dir is not None else None
        except BaseException:
            self.policy.close()
            raise
        self.request_count = 0
        self.generation = 0
        self.pending = None
        self.result = None
        self.error = None
        self.closed = False
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._run, name="hanoi-inference", daemon=True)
        self.thread.start()

    def submit(self, observation: Observation):
        if not 0 <= observation.image_age_s <= hanoi.CONTRACT["max_image_age_s"]:
            raise ValueError("Observation image is stale or from the future")
        # Own the arrays: the caller may reuse its camera/state buffers.
        observation = dataclasses.replace(
            observation,
            data=copy.deepcopy(observation.data),
            velocity_override=copy.deepcopy(observation.velocity_override),
        )
        with self.condition:
            if self.closed:
                raise RuntimeError("Inference worker is closed")
            if self.error is not None:
                raise RuntimeError("Inference worker failed") from self.error
            self.pending = dataclasses.replace(observation, generation=self.generation)
            self.condition.notify()

    def invalidate(self) -> int:
        with self.condition:
            self.generation += 1
            self.pending = self.result = None
            return self.generation

    def take(self) -> Prediction | None:
        with self.condition:
            if self.error is not None:
                raise RuntimeError("Inference worker failed") from self.error
            result, self.result = self.result, None
            return result

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                observation, self.pending = self.pending, None
            request_id = self.request_count
            self.request_count += 1
            started = None
            try:
                if self.recorder is not None:
                    self.recorder.save_input(request_id, observation)
                started = time.monotonic()
                actions = np.asarray(self.policy.infer(observation.data)["actions"], dtype=np.float64)
                if actions.shape != (self.horizon, 4) or not np.isfinite(actions).all():
                    raise ValueError(f"Expected {self.horizon} finite absolute XYZ/jaw references")
                received = time.monotonic()
                result = Prediction(observation, actions.copy(), received, received - started, request_id)
                if self.recorder is not None:
                    self.recorder.write(
                        "inference_response",
                        request_id,
                        started_at_s=started,
                        received_at_s=received,
                        inference_s=received - started,
                        latency_s=received - observation.captured_at,
                        actions=actions.tolist(),
                    )
            except Exception as exc:
                if self.recorder is not None:
                    try:
                        self.recorder.write(
                            "inference_error", request_id, started_at_s=started, error=repr(exc), cancelled=self.closed
                        )
                    except Exception:
                        logging.exception("Could not record inference failure")
                with self.condition:
                    if not self.closed:
                        self.error = exc
                    self.condition.notify_all()
                return
            with self.condition:
                if observation.generation == self.generation and not self.closed:
                    self.result = result
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = self.result = None
            self.condition.notify_all()
        # Closing the socket interrupts recv, including the long first compile.
        self.policy.close()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Inference worker did not stop")


class ActionBuffer:
    """Newest chunk replaces uncommitted actions; a dispatched prefix is fixed.

    A chunk's row k is due at observation_tick + k + 1. This representation
    bounds storage at one 63-row chunk without splicing unrelated trajectories.
    """

    def __init__(self, executor: ReferenceExecutor, *, generation: int = 0):
        self.executor = executor
        self.generation = generation
        self.prediction = None
        self.last_observation_tick = -1

    def reset(self, executor: ReferenceExecutor, generation: int):
        self.executor = executor
        self.generation = generation
        self.prediction = None
        self.last_observation_tick = -1

    def accept(self, prediction: Prediction) -> bool:
        obs = prediction.observation
        if isinstance(self.executor, PolicyExecutor) and self.executor.pending_jaw_open is not None:
            # Keep the source of the committed XYZ-then-gripper operation until
            # both commands finish; newer inference must not split that pair.
            return False
        if obs.generation != self.generation or obs.tick <= self.last_observation_tick:
            return False
        if obs.tick < self.executor.fresh_after_tick:
            return False
        self.prediction = prediction
        self.last_observation_tick = obs.tick
        return True

    def propose(self, tick: int) -> tuple[Command, ReferenceExecutor] | None:
        if tick < self.executor.available_tick or self.prediction is None:
            return None
        obs = self.prediction.observation
        # Transactional planning: failed driver dispatch must not advance state.
        proposed = copy.deepcopy(self.executor)
        command = proposed.plan(
            self.prediction.actions,
            observation_tick=obs.tick,
            now_tick=tick,
            image_age_s=obs.image_age_s,
            task=obs.data["prompt"],
        )
        return command, proposed

    def commit(self, command: Command, executor: ReferenceExecutor):
        self.executor = executor
        if command.kind == "gripper":
            self.prediction = None


def reference_executor(state: np.ndarray, *, jaw_open: bool, tick: int = 0) -> PolicyExecutor:
    """Start after a completed initialization/hold command, with zero reference derivatives.

    The measured velocity remains part of the policy observation. It is not the
    commanded endpoint velocity and must not be used as a stationary-arm gate.
    """
    state = np.asarray(state)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("Expected seven measured state values")
    return PolicyExecutor(state[:3].copy(), jaw_open=jaw_open, available_tick=tick, fresh_after_tick=tick)
