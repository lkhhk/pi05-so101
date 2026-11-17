"""Check stepped simulator with recorded actions and a stationary control."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from openarm_assisted_env import OpenArmAssistedEnv
from openarm_pi_adapter import OpenArmInputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode", type=Path)
    p.add_argument("--hold-initial", action="store_true")
    args = p.parse_args()
    ep = np.load(args.episode, allow_pickle=False)
    meta = json.loads(str(ep["metadata_json"]))
    env = OpenArmAssistedEnv(offset=meta.get("bottle_offset_xy_m", [0, 0]))
    try:
        obs = env.observe()
        converted = OpenArmInputs()(obs)
        assert converted["state"].shape == (8,)
        assert converted["image"]["base_0_rgb"].shape == (224, 224, 3)
        for action in ep["action"]:
            env.step(ep["action"][0] if args.hold_initial else action)
        print(env.metrics())
        print("final_obs", env.observe()["observation/state"].shape)
    finally:
        env.close()


if __name__ == "__main__":
    main()
