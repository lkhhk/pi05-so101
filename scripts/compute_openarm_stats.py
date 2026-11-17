"""Compute diagnostic OpenPI statistics from a local OpenArm LeRobot dataset.

Use only training episodes. Two sample episodes are insufficient for a model;
these statistics validate the pipeline and must be recomputed for real training.
"""
import argparse
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from openpi.shared import normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path, help="New directory to contain norm_stats.json")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = "local/openarm_unscrew_diagnostic"
    meta = LeRobotDatasetMetadata(repo_id=repo, root=args.dataset)
    ds = LeRobotDataset(repo_id=repo, root=args.dataset,
                         delta_timestamps={"actions": [i / meta.fps for i in range(10)]})
    states, actions = [], []
    for i in range(len(ds)):
        item = ds[i]
        states.append(item["state"].numpy())
        actions.append(item["actions"].numpy())
    stats = {}
    for name, values in (("state", states), ("actions", actions)):
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != 8 or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name}: {values.shape}")
        running = normalize.RunningStats()
        running.update(values)
        stats[name] = running.get_statistics()
    normalize.save(args.output, stats)
    print(f"Saved diagnostic statistics: {ds.num_episodes} episodes, {len(ds)} frames, "
          f"10-step action chunks -> {args.output}")


if __name__ == "__main__":
    main()
