"""Publish a deployment run into an easy-to-find experiment folder.

``exp_vid/<date>_<family>_h<horizon>_<moves>moves_<status>/`` in the OpenPI checkout for pi0.5 runs
and in the Cosmos checkout for Cosmos runs, containing:

* ``README.md``        what ran, the move-by-move reconstruction, legality, whether the puzzle was solved
* ``camera.mp4``       the full-frame camera bag rendered to video (when the run recorded one)
* ``policy_view.mp4``  the 224 x 224 crops the policy actually saw, at the inference rate
* ``summary.json``, ``moves.json``, ``source.txt``
* for Cosmos runs, ``dreams.mp4``, ``dream_strips/``, ``dreams_contact.png``, ``dream_report.json``
  (regenerated from the saved inputs with ``examples/hanoi/dream_dense_run.py`` in the Cosmos checkout), and
  ``dream_story.mp4`` (dream first, then reality; see dream_story.py) when the run has a camera bag

    ./run_publish_run.sh data/hanoi/deployment/dense_live_<id>
"""

import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

OPENPI_ROOT = Path(__file__).resolve().parents[3]
COSMOS_ROOT = OPENPI_ROOT.parent / "cosmos-policy"
from examples.hanoi.deployment.progress import reconstruct


def policy_view(run: Path, output: Path) -> int:
    """Render the saved policy inputs to a video at the run's inference rate."""
    from PIL import Image, ImageDraw

    stamps = {}
    for line in (run / "inferences.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "inference_request":
            stamps[int(e["request_id"])] = e["observation_captured_at_s"]
    inputs = sorted((run / "inference_inputs").glob("*.npz"), key=lambda p: int(p.stem))
    if not inputs:
        return 0
    times = np.array([stamps.get(int(p.stem), np.nan) for p in inputs])
    gaps = np.diff(times[np.isfinite(times)])
    fps = 1 / float(np.median(gaps)) if len(gaps) else 10.0
    t0 = np.nanmin(times)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "448x466",
                                "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    for p, t in zip(inputs, times):
        crop = np.load(p)["observation/image"]
        canvas = Image.new("RGB", (448, 466), (20, 20, 20))
        canvas.paste(Image.fromarray(crop).resize((448, 448), Image.NEAREST), (0, 18))
        ImageDraw.Draw(canvas).text((4, 3), f"policy input  req {int(p.stem)}  t={t - t0:6.1f} s", fill=(230, 230, 230))
        encoder.stdin.write(np.asarray(canvas).tobytes())
    encoder.stdin.close()
    encoder.wait()
    return len(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dest", type=Path, default=None, help="experiment folder root; default by policy family")
    parser.add_argument("--name", default=None, help="folder name; default <date>_<family>_h<horizon>_<moves>moves_<status>")
    parser.add_argument("--no-dreams", action="store_true", help="skip the Cosmos dream regeneration")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-end-s", type=float, default=None, help="cut the camera video here (seconds into the bag)")
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "summary.json").exists():
        raise FileNotFoundError(f"{run} has no summary.json: the run is still going or was killed before cleanup")
    summary = json.loads((run / "summary.json").read_text())
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    family = summary.get("policy_family") or "unknown"
    cosmos = family.startswith("cosmos")
    dest = args.dest or ((COSMOS_ROOT if cosmos else OPENPI_ROOT) / "exp_vid")
    moves = reconstruct(events).report()
    horizon = summary.get("action_horizon")
    variant = f"h{horizon}" if horizon else ((summary.get("config_name") or "").replace("pi05_hanoi_", "").replace("cosmos_hanoi_", "").replace("_aaaa_to_cccc", "") or "policy")
    status = "solved" if moves["solved"] else summary["status"]
    date = datetime.datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d")
    name = args.name or f"{date}_{family}_{variant}_{len(moves['moves']):02d}moves_{status}"
    if summary.get("tag"):
        name = f"{summary['tag']}_{name}"
    folder = dest / name
    if folder.exists() and not args.overwrite:
        raise FileExistsError(f"{folder} exists; pass --overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source.txt").write_text(str(run) + "\n")
    shutil.copy(run / "summary.json", folder / "summary.json")
    (folder / "moves.json").write_text(json.dumps(moves, indent=2) + "\n")
    duration = events[-1]["monotonic_s"] - events[0]["monotonic_s"]
    outputs = []
    bag = run / "camera_bag"
    if bag.is_dir() and any(bag.glob("*.mcap")) and (summary.get("camera_bag") or {}).get("outcome"):
        cut = ["--end-s", str(args.video_end_s)] if args.video_end_s else []
        subprocess.run([sys.executable, "-m", "examples.hanoi.deployment.bag_to_video", str(run), "--output", str(folder / "camera.mp4"), *cut],
                       cwd=OPENPI_ROOT, check=True)
        outputs.append("camera.mp4")
    if (run / "inference_inputs").is_dir():
        n = policy_view(run, folder / "policy_view.mp4")
        outputs.append(f"policy_view.mp4 ({n} inputs)")
    if cosmos and not args.no_dreams:
        python = COSMOS_ROOT / ".venv/bin/python"
        subprocess.run([str(python), "examples/hanoi/dream_dense_run.py", "--run-dir", str(run), "--output", str(folder)],
                       cwd=COSMOS_ROOT, check=True)
        outputs.append("dreams.mp4, dream_strips/, dreams_contact.png, dream_report.json")
        if "camera.mp4" in outputs:
            subprocess.run([sys.executable, "-m", "examples.hanoi.deployment.dream_story", str(run), str(folder)], cwd=OPENPI_ROOT, check=True)
            outputs.append("dream_story.mp4")
    lines = [f"# {name}", "",
             f"Run: `{run}`  ",
             f"Policy: {family}, config `{summary.get('config_name')}`, export `{(summary.get('server_export_sha256') or '')[:16]}`, "
             f"adapter `{summary.get('execution_adapter')}`, start `{summary.get('start')}`  ",
             f"Status: {summary['status']}; duration {duration:.0f} s; segments {summary.get('segments')}, brakes {summary.get('brakes')}, "
             f"rejected {summary.get('rejected_commands')}, tracking p95 {summary.get('tracking_error_p95_mm') or 0:.1f} mm, "
             f"inference {1000 * (summary.get('latency_median_s') or summary.get('inference_median_s') or 0):.0f} ms  ",
             f"Moves: {len(moves['moves'])}, all legal: {moves['all_legal']}, solved: {moves['solved']}, final board {moves['final_board']}  ",
             f"Progress: {moves['optimal_prefix']} optimal moves in a row; {moves['remaining_moves']} moves remaining to the goal; "
             f"progress {moves['progress']}" + (" (board uncertain after a slipped ring)" if moves["board_uncertain"] else ""),
             "", "| # | t (s) | ring | from | to | legal | grasp height error (mm) |", "|---|---|---|---|---|---|---|"]
    for k, (m, g) in enumerate(zip(moves["moves"], moves["grasps"]), 1):
        lines.append(f"| {k} | {m['t_s']} | {m['ring']} | {m['from']} | {m['to']} | {'yes' if m['legal'] else 'NO'} | {g['level_error_mm']:+.1f} |")
    lines += ["", "Files: " + ", ".join(outputs) + ", summary.json, moves.json, source.txt", ""]
    (folder / "README.md").write_text("\n".join(lines))
    print(f"Published {folder}")
    print("  " + "\n  ".join(outputs))


if __name__ == "__main__":
    main()
