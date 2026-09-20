"""Turn a run's camera bag into an MP4.

Reads the ``sensor_msgs/Image`` messages of ``<run>/camera_bag`` (mcap, rgb8) with rosbag2_py and
pipes the frames to ffmpeg (H.264, yuv420p, playable anywhere). The output frame rate is the bag's
own average rate unless ``--fps`` is given, in which case frames are picked by nearest timestamp.
Run in the robot venv with the ROS environment sourced::

    ./run_bag_to_video.sh data/hanoi/deployment/dense_live_<id>            # -> <run>/camera.mp4
    ./run_bag_to_video.sh <run> --fps 30 --output clip.mp4 --start-s 10 --end-s 40
"""

import argparse
import logging
from pathlib import Path
import subprocess

import numpy as np

IMAGE_TOPIC = "/camera/camera/color/image_raw"


def read_frames(bag: Path, topic: str):
    """Yield (stamp_s, rgb array) for every image message in the bag, in time order."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        _, data, stamp_ns = reader.read_next()
        message = deserialize_message(data, Image)
        if message.encoding != "rgb8":
            raise ValueError(f"Expected rgb8 frames, got {message.encoding}")
        frame = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step // 3, 3)[:, : message.width]
        yield stamp_ns / 1e9, frame


def bag_summary(bag: Path, topic: str):
    import rosbag2_py

    info = rosbag2_py.Info().read_metadata(str(bag), "mcap")
    count = next(t.message_count for t in info.topics_with_message_count if t.topic_metadata.name == topic)
    duration = info.duration.nanoseconds / 1e9
    return count, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory (containing camera_bag) or the bag directory itself")
    parser.add_argument("--output", type=Path, default=None, help="default <run>/camera.mp4")
    parser.add_argument("--topic", default=IMAGE_TOPIC)
    parser.add_argument("--fps", type=float, default=None, help="output rate; default the bag's average rate")
    parser.add_argument("--start-s", type=float, default=0.0, help="seconds from the first frame")
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--crf", type=int, default=20, help="x264 quality (lower is better, 18-23 is usual)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bag = args.run / "camera_bag" if (args.run / "camera_bag").is_dir() else args.run
    output = args.output or (args.run / "camera.mp4" if bag != args.run else bag.with_name("camera.mp4"))
    count, duration = bag_summary(bag, args.topic)
    native = count / duration if duration > 0 else 30.0
    fps = args.fps or native
    logging.info("%s: %d frames over %.1f s (%.1f Hz); writing %s at %.1f fps", bag, count, duration, native, output, fps)
    frames = read_frames(bag, args.topic)
    first_stamp, first = next(frames)
    height, width = first.shape[:2]
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    written = 0
    next_due = args.start_s
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        def consider(stamp, frame):
            nonlocal written, next_due
            t = stamp - first_stamp
            if t < args.start_s or (args.end_s is not None and t > args.end_s):
                return
            if args.fps is None or t + 0.5 / fps >= next_due:
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                written += 1
                if args.fps is not None:
                    next_due += 1 / fps
        consider(first_stamp, first)
        for stamp, frame in frames:
            consider(stamp, frame)
        encoder.stdin.close()
        encoder.wait()
    if encoder.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {encoder.returncode}")
    logging.info("Wrote %s: %d frames, %.1f s, %.1f MB", output, written, written / fps, output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
