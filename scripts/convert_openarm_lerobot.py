"""Convert an OpenArm diagnostic episode to a local LeRobot v2 dataset.

The conversion does not include the privileged full-rate replay signals; the
result is an interface test, not a sufficient training set or a deployable
controller. Run with the pinned OpenPI Python environment.
"""
import argparse
import json
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode", type=Path, nargs="+", help="One or more NPZ demonstrations")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    episodes = [np.load(path, allow_pickle=False) for path in args.episode]
    episode = episodes[0]
    meta = json.loads(str(episode["metadata_json"]))
    if meta["sample_hz"] != 10:
        raise ValueError("Expected 10 Hz demonstration")
    n = len(episode["time"])
    expected = {"state": (n, 8), "action": (n, 8),
                "table_image": (n, 224, 224, 3), "front_image": (n, 224, 224, 3)}
    for key, shape in expected.items():
        if episode[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {episode[key].shape}")
    features = {
        "table_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
        "front_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": meta["state_joint_names"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": meta["action_actuator_names"]},
    }
    ds = LeRobotDataset.create(repo_id="local/openarm_unscrew_diagnostic", root=args.output,
                                fps=10, robot_type="openarm", features=features,
                                use_videos=False, image_writer_threads=2)
    total = 0
    for episode in episodes:
        current_meta = json.loads(str(episode["metadata_json"]))
        if any(current_meta[k] != meta[k] for k in ("sample_hz", "state_joint_names", "action_actuator_names")):
            raise ValueError("Incompatible episode schema")
        count = len(episode["time"])
        for key, shape in expected.items():
            if episode[key].shape != (count, *shape[1:]):
                raise ValueError(f"Invalid {key} shape: {episode[key].shape}")
        for i in range(count):
            ds.add_frame({"table_image": episode["table_image"][i],
                          "front_image": episode["front_image"][i],
                          "state": episode["state"][i].astype(np.float32),
                          "actions": episode["action"][i].astype(np.float32),
                          "task": current_meta["task"]})
        ds.save_episode()
        total += count
    print(f"Saved {len(episodes)} episodes and {total} frames to {args.output}")


if __name__ == "__main__":
    main()
