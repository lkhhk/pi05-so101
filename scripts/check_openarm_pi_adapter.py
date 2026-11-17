"""Smoke-check OpenArm's LeRobot -> OpenPI transform without training."""
import argparse
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.models.pi0_config import Pi0Config
from openpi.training.config import DataConfig
from openpi import transforms

from openarm_pi_adapter import ACTION_HIGH, ACTION_LOW, OpenArmOutputs
from openarm_pi_config import OpenArmDataConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    ds = LeRobotDataset(repo_id="local/openarm_unscrew_diagnostic", root=args.dataset)
    source = ds[0]
    source["actions"] = np.stack([ds[i]["actions"].numpy() for i in range(10)])
    source["prompt"] = source["task"]
    cfg = OpenArmDataConfig(repo_id="local/openarm_unscrew_diagnostic",
                            base_config=DataConfig(prompt_from_task=True))
    model = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    pipeline = cfg.create(Path("/nonexistent/openarm_smoke_assets"), model)
    result = transforms.compose([*pipeline.repack_transforms.inputs,
                                 *pipeline.data_transforms.inputs,
                                 *pipeline.model_transforms.inputs])(source)
    assert result["state"].shape == (32,)
    assert result["actions"].shape == (10, 32)
    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert np.isfinite(result["actions"]).all()
    clipped = OpenArmOutputs()({"actions": np.full((10, 32), 100, dtype=np.float32)})["actions"]
    assert clipped.shape == (10, 8) and np.all(clipped <= ACTION_HIGH) and np.all(clipped >= ACTION_LOW)
    try:
        OpenArmOutputs()({"actions": np.full((10, 32), np.nan, dtype=np.float32)})
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite action was accepted")
    print(f"OK episodes={ds.num_episodes} frames={len(ds)} state={result['state'].shape} "
          f"actions={result['actions'].shape} cameras=2 output=(10, 8)")


if __name__ == "__main__":
    main()
