"""Compare model actions with recorded expert states (not a rollout)."""
import argparse
import json
from pathlib import Path

import numpy as np

from openarm_pi_adapter import OpenArmOutputs
from run_openarm_pi import PROTOCOL, PolicyClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--stride", type=int, default=5)
    args = p.parse_args()
    if args.stride <= 0 or args.output.exists():
        raise ValueError("Stride must be positive and output must be new")
    episode = np.load(args.episode, allow_pickle=False)
    meta = json.loads(str(episode["metadata_json"]))
    client = PolicyClient(args.port)
    try:
        if client.metadata.get("action_protocol") != PROTOCOL or client.metadata.get("mock"):
            raise ValueError("Expected an OpenArm model server, not a mock")
        predicted, target, indices = [], [], []
        for i in range(0, len(episode["time"]), args.stride):
            obs = {"observation/table_image": episode["table_image"][i],
                   "observation/front_image": episode["front_image"][i],
                   "observation/state": episode["state"][i].astype(np.float32),
                   "prompt": meta["task"]}
            actions = OpenArmOutputs()(client.infer(obs))["actions"]
            if actions.ndim != 2 or len(actions) < 1:
                raise ValueError("Bad action shape")
            predicted.append(actions[0]); target.append(episode["action"][i]); indices.append(i)
    finally:
        client.close()
    predicted, target = np.asarray(predicted), np.asarray(target)
    summary = {"episode": str(args.episode), "mode": "expert-observation-only; not rollout",
               "stride": args.stride, "samples": len(indices),
               "joint_mae_rad": np.mean(np.abs(predicted[:, :7] - target[:, :7]), axis=0).tolist(),
               "joint_mae_mean_rad": float(np.mean(np.abs(predicted[:, :7] - target[:, :7]))),
               "gripper_mae": float(np.mean(np.abs(predicted[:, 7] - target[:, 7]))),
               "gripper_binary_agreement": float(np.mean((predicted[:, 7] > -0.25) == (target[:, 7] > -0.25))),
               "expert_closed_fraction": float(np.mean(target[:, 7] > -0.25)),
               "predicted_closed_fraction": float(np.mean(predicted[:, 7] > -0.25))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
