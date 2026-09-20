"""Move to a start pose and verify feedback, then return home; no camera/server needed.

``--start rod_a`` (default) is the pi0.5 hover above rod A; ``--start cosmos_v4`` is the
waypoint_v4 episode start the Cosmos client uses, for a dry run before a policy run.
"""

import dataclasses
import json
import logging
from pathlib import Path
import signal
import time
from typing import Literal

import numpy as np
import tyro

from examples.hanoi.deployment.hardware import INITIAL_XYZ
from examples.hanoi.deployment.hardware import TrossenArm


@dataclasses.dataclass
class Config:
    robot_ip: str = "192.168.1.3"
    output: Path = Path("data/hanoi/deployment")
    start: Literal["rod_a", "cosmos_v4", "above_peg_a"] = "rod_a"


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.output.mkdir(parents=True, exist_ok=True)
    path = config.output / f"initial_pose_{time.time_ns()}.json"
    if config.start in ("cosmos_v4", "above_peg_a"):
        from examples.hanoi.deployment.cosmos_client import START_POSES

        start_xyz = np.array(START_POSES["episode_start" if config.start == "cosmos_v4" else "above_peg_a"][0])
    else:
        start_xyz = INITIAL_XYZ
    result = {"status": "failed", "robot_ip": config.robot_ip, "start": config.start,
              "start_xyz_m": start_xyz.tolist(), "returned_home": False}
    arm = None
    return_home = True
    interrupted = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        try:
            logging.info("Initializing: joint home, XYZ %s m / 45-degree pitch, then open jaw to 0.034 m", start_xyz)
            arm = TrossenArm(config.robot_ip, initial_xyz=start_xyz)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            try:
                result.update(arm.initialize())
            except KeyboardInterrupt:
                interrupted = True
                raise
            except RuntimeError:
                return_home = False
                raise
        finally:
            if arm is not None:
                if arm.initial_proprio_alignment is not None:
                    result["initial_proprio_alignment"] = arm.initial_proprio_alignment
                if arm.last_initial_pose_report is not None:
                    result.update(arm.last_initial_pose_report)
                    logging.info(
                        "Measured XYZ (m): %s; rotation vector (rad): %s; jaw %.5f m",
                        result["measured_xyz_m"],
                        result["measured_rotvec_rad"],
                        result["measured_jaw_m"],
                    )
                try:
                    if arm.enabled and return_home:
                        try:
                            if interrupted:
                                result.update(arm.stop_and_release())
                            else:
                                logging.info("Returning arm to joint home")
                                result["home_joints_rad"] = arm.go_home()
                                result["returned_home"] = True
                        except BaseException:
                            arm.hold()
                            raise
                    else:
                        arm.hold()
                finally:
                    arm.close()
        result["status"] = "passed"
    except KeyboardInterrupt:
        result["status"] = "operator_stop"
        logging.info("Initial pose test stopped; returned home: %s", result["returned_home"])
        return result
    except BaseException as exc:
        result["error"] = repr(exc)
        logging.error("Initial pose test FAILED: %s", exc)
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")
        signal.signal(signal.SIGINT, previous_sigint)
        logging.info("Initial pose report: %s", path)
    logging.info(
        "Initial pose test PASSED: XYZ error %.3f mm, rotation error %.3f deg, jaw %.5f m. Returned home.",
        result["position_error_mm"],
        result["orientation_error_deg"],
        result["measured_jaw_m"],
    )
    return result


if __name__ == "__main__":
    main(tyro.cli(Config))
