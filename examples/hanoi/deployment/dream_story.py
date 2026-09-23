"""Dream-first video: the live view pauses, the model's dreams play, then reality plays.

For a Cosmos run with regenerated dreams (``dream_dense_run.py``), the video alternates:

1. **Dream phase.** The live view freezes at time ``s``. The right panel plays ``--dreams-per-pause``
   dream frames (default 10): the dream made at ``s`` shows the scene the model expects 1.6 s later, the
   dream made at ``s + 1.6`` shows 3.2 s later, and so on, so ten dreams cover the next 16 s.
2. **Reality phase.** The real footage of those seconds plays at real speed while the right panel holds the
   last dream, the cycle's final target, until the live view reaches it (``--reality-panel step`` instead steps
   through all the dreams). The next cycle's dreams only start once the live view has caught up.

The live panel is the camera bag cropped exactly as the policy sees it (224 x 224, upscaled), so the
two panels share one view. Needs the robot venv with the ROS environment sourced::

    ./run_dream_story.sh <run dir> <exp_vid folder>
    # -> <exp_vid folder>/dream_story_cycles/cycle_NN_tXXXs.mp4 (one per five dreams) and dream_story.mp4 (all of them)
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from openpi_client import hanoi

from examples.hanoi.deployment.bag import start_bag  # noqa: F401  (documents the settle used for alignment)
from examples.hanoi.deployment.bag_to_video import read_frames

AHEAD_S = 1.6
BAG_SETTLE_S = 1.5  # start_bag() sleeps this long before logging bag_started
PANEL = 448
HEADER = 40
GAP = 8


def load_dreams(folder: Path) -> dict:
    """request_id -> dream frame (224 x 224 x 3), from dream_frames/ or cut out of the strips."""
    from PIL import Image

    frames = {}
    if (folder / "dream_frames").is_dir():
        for p in sorted((folder / "dream_frames").glob("*.png")):
            frames[int(p.stem)] = np.asarray(Image.open(p).convert("RGB"))
    else:
        for p in sorted((folder / "dream_strips").glob("*.png")):
            strip = np.asarray(Image.open(p).convert("RGB"))
            frames[int(p.stem)] = strip[strip.shape[0] - 224 :, 224:448]
    if not frames:
        raise FileNotFoundError(f"No dreams under {folder}; run dream_dense_run.py first")
    return frames


def wall_offset(run: Path, events: list) -> float:
    """wall_s - monotonic_s for this run, from the recorder's start (logged wall time if present)."""
    started = next((e for e in events if e["event"] == "bag_started"), None)
    if started is None:
        raise ValueError("This run has no camera bag")
    if "wall_s" in started:
        return started["wall_s"] - started["monotonic_s"]
    # Older runs: the run directory's name is the wall time just before the recorder was started,
    # and bag_started was logged BAG_SETTLE_S after that.
    run_wall_s = int(run.name.rsplit("_", 1)[-1]) / 1e9
    return run_wall_s - (started["monotonic_s"] - BAG_SETTLE_S)


def draw(canvas_left, canvas_right, left_text, right_text, banner=None):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (2 * PANEL + GAP, PANEL + HEADER), (18, 18, 18))
    canvas.paste(Image.fromarray(canvas_left).resize((PANEL, PANEL), Image.NEAREST), (0, HEADER))
    canvas.paste(Image.fromarray(canvas_right).resize((PANEL, PANEL), Image.NEAREST), (PANEL + GAP, HEADER))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), left_text, fill=(235, 235, 235))
    d.text((PANEL + GAP + 6, 6), right_text, fill=(255, 220, 90))
    if banner:
        d.rectangle([0, HEADER + PANEL - 26, 2 * PANEL + GAP, HEADER + PANEL], fill=(18, 18, 18))
        d.text((6, HEADER + PANEL - 22), banner, fill=(235, 235, 235))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("folder", type=Path, help="exp_vid folder holding the dreams; the video is written there")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dreams-per-pause", type=int, default=10)
    parser.add_argument("--reality-panel", choices=["last", "step"], default="last",
                        help="during the reality phase show the last dream (the cycle's final target) or step through all of them")
    parser.add_argument("--dream-hold-s", type=float, default=0.6, help="how long each dream frame is shown")
    parser.add_argument("--ahead-s", type=float, default=AHEAD_S)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start-s", type=float, default=None, help="first pause, seconds after the first inference")
    parser.add_argument("--end-s", type=float, default=None)
    args = parser.parse_args()
    output = args.output or args.folder / "dream_story.mp4"
    events = [json.loads(l) for l in (args.run / "events.jsonl").read_text().splitlines()]
    requests = {}
    for line in (args.run / "inferences.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "inference_request":
            requests[int(e["request_id"])] = e["observation_captured_at_s"]
    dreams = load_dreams(args.folder)
    ids = np.array(sorted(i for i in dreams if i in requests))
    times = np.array([requests[i] for i in ids])  # monotonic
    offset = wall_offset(args.run, events)
    t0 = times[0]
    cycle_s = args.dreams_per_pause * args.ahead_s
    # Bag frames as policy crops, indexed by run time.
    bag_t, crops = [], []
    for stamp_wall, frame in read_frames(args.run / "camera_bag", "/camera/camera/color/image_raw"):
        t = stamp_wall - offset - t0
        bag_t.append(t)
        crops.append(hanoi.preprocess_camera(frame))
    bag_t = np.array(bag_t)
    if not len(bag_t):
        raise ValueError("Empty camera bag")

    def live_at(t):
        return crops[int(np.clip(np.searchsorted(bag_t, t), 0, len(bag_t) - 1))]

    def dream_made_at(t):
        """The dream from the inference nearest to run time t, and that inference's time."""
        k = int(np.clip(np.searchsorted(times - t0, t), 0, len(times) - 1))
        if k > 0 and abs(times[k - 1] - t0 - t) < abs(times[k] - t0 - t):
            k -= 1
        return dreams[int(ids[k])], times[k] - t0

    start = args.start_s if args.start_s is not None else 0.0
    end = args.end_s if args.end_s is not None else min(bag_t[-1], times[-1] - t0)
    cycles_dir = output.with_name(output.stem + "_cycles")
    cycles_dir.mkdir(exist_ok=True)
    size = f"{2 * PANEL + GAP}x{PANEL + HEADER}"

    def encoder_for(path):
        return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size,
                                 "-r", f"{args.fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    cycle_files = []
    s = start
    number = 0
    while s < end:
        # The five dreams of this cycle: dream j is made at s + j*ahead and targets s + (j+1)*ahead.
        plan = []
        for j in range(args.dreams_per_pause):
            made_t = s + j * args.ahead_s
            if made_t > end:
                break
            dream, actual_t = dream_made_at(made_t)
            plan.append((dream, actual_t, actual_t + args.ahead_s))
        if not plan:
            break
        number += 1
        path = cycles_dir / f"cycle_{number:02d}_t{int(round(s)):03d}s.mp4"
        cycle_files.append(path)
        encoder = encoder_for(path)
        frozen = live_at(s)
        hold = max(1, int(round(args.dream_hold_s * args.fps)))
        # Dream phase: live frozen, the dreams play forward in time, 1/5 to 5/5.
        for j, (dream, made, target) in enumerate(plan, 1):
            for _ in range(hold):
                encoder.stdin.write(draw(frozen, dream, f"LIVE paused at t = {s:5.1f} s",
                                         f"DREAM {j}/{len(plan)}: the model expects this at t = {target:5.1f} s (made at {made:5.1f} s)",
                                         f"the model dreams the next {plan[-1][2] - s:.0f} s first").tobytes())
        # Reality phase: the same dreams in the same order, each held until the live view reaches its target.
        cycle_end = min(plan[-1][2], end)
        n = int(round((cycle_end - s) * args.fps))
        for i in range(n):
            t = s + i / args.fps
            if args.reality_panel == "step":
                j = next((k for k, (_, _, target) in enumerate(plan) if t < target), len(plan) - 1)
            else:
                j = len(plan) - 1  # hold the final dream target while the live view catches up with it
            dream, made, target = plan[j]
            remaining = max(0.0, target - t)
            encoder.stdin.write(draw(live_at(t), dream, f"LIVE  t = {t:5.1f} s",
                                     f"DREAM {j + 1}/{len(plan)} for t = {target:5.1f} s   live reaches it in {remaining:3.1f} s",
                                     f"reality: the same {plan[-1][2] - s:.0f} s as they happened").tobytes())
        encoder.stdin.close()
        encoder.wait()
        if encoder.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {path}")
        s = cycle_end
    manifest = cycles_dir / "concat.txt"
    manifest.write_text("".join(f"file '{p.resolve()}'\n" for p in cycle_files))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy",
                    str(output)], check=True)
    total = sum(1 for _ in cycle_files)
    print(f"Wrote {total} cycle videos under {cycles_dir} and the concatenation {output} ({end - start:.0f} s of run)")


if __name__ == "__main__":
    main()
