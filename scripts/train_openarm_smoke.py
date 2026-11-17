"""Run an OpenPI LoRA pipeline smoke test, not a trained task policy.

Run in the OpenPI checkout. HF_LEROBOT_HOME must point at a local LeRobot
cache containing local/openarm_unscrew_diagnostic. The checkpoint is saved to
the supplied new directory; existing experiments are never overwritten.
"""
import argparse
import dataclasses
import importlib.util
from pathlib import Path

from openpi.models.pi0_config import Pi0Config
from openpi.training import config, data_loader, optimizer, weight_loaders

from openarm_pi_config import OpenArmDataConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-params", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--exp-name", default="pipeline-smoke")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--loader-only", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    if not args.checkpoint_params.is_dir():
        raise FileNotFoundError(args.checkpoint_params)
    if not (args.assets_root / "openarm_unscrew" / "norm_stats.json").is_file():
        raise FileNotFoundError("Missing OpenArm training-only statistics")
    model = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                      paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    data = OpenArmDataConfig(repo_id="local/openarm_unscrew_diagnostic",
                             assets=config.AssetsConfig(assets_dir=str(args.assets_root), asset_id="openarm_unscrew"),
                             base_config=config.DataConfig(prompt_from_task=True))
    cfg = dataclasses.replace(config.get_config("pi05_libero"), name="pi05_openarm_lora_smoke",
                              exp_name=args.exp_name, model=model, data=data,
                              freeze_filter=model.get_freeze_filter(), ema_decay=None,
                              weight_loader=weight_loaders.CheckpointWeightLoader(str(args.checkpoint_params)),
                              checkpoint_base_dir=str(args.checkpoint_root), num_train_steps=args.steps,
                              batch_size=args.batch_size, num_workers=0, wandb_enabled=False,
                              overwrite=False, resume=False, log_interval=1, save_interval=1000,
                              lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=2.5e-5,
                                                                        decay_steps=1000, decay_lr=2.5e-6))
    if args.loader_only:
        loader = data_loader.create_data_loader(cfg, shuffle=False, num_batches=1)
        obs, actions = next(iter(loader))
        print("loader OK", "state", obs.state.shape, "actions", actions.shape,
              "cameras", list(obs.images), "training-only stats loaded")
        return
    train_path = Path("scripts/train.py").resolve()
    if not train_path.is_file():
        raise FileNotFoundError("Run this script from the OpenPI checkout")
    spec = importlib.util.spec_from_file_location("openpi_training_entry", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(cfg)


if __name__ == "__main__":
    main()
