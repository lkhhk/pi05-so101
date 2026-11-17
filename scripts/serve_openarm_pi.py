"""Serve an OpenArm-trained OpenPI checkpoint on localhost.

Requires an OpenArm-specific checkpoint with matching normalization statistics;
the original LIBERO checkpoint cannot be used here.
"""
import argparse
import dataclasses
from pathlib import Path

from openpi.policies import policy_config
from openpi.models.pi0_config import Pi0Config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config

from openarm_pi_config import OpenArmDataConfig


PROTOCOL = "openarm-absolute-position-8d-v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--lora", action="store_true", help="Checkpoint contains OpenPI LoRA parameters")
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve()
    stats = checkpoint / "assets" / "openarm_unscrew" / "norm_stats.json"
    if not (checkpoint / "params").is_dir() or not stats.is_file():
        raise ValueError("Expected OpenArm checkpoint params and assets/openarm_unscrew/norm_stats.json")
    base = config.get_config("pi05_libero")
    data = OpenArmDataConfig(repo_id="local/openarm_unscrew_diagnostic",
                             assets=config.AssetsConfig(asset_id="openarm_unscrew"),
                             base_config=config.DataConfig(prompt_from_task=True))
    model = (Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                       paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
             if args.lora else base.model)
    train_config = dataclasses.replace(base, name="pi05_openarm_unscrew", data=data, model=model,
                                       policy_metadata={"action_protocol": PROTOCOL})
    policy = policy_config.create_trained_policy(train_config, checkpoint)
    WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port,
                          metadata={"action_protocol": PROTOCOL}).serve_forever()


if __name__ == "__main__":
    main()
