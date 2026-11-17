"""SO-101 medicine LoRA training and readback for OpenPI's pinned environment."""
import argparse, dataclasses, importlib.util
from pathlib import Path
import numpy as np
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.training import config, data_loader, optimizer, weight_loaders
from openarm_pi_adapter import _image
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# TorchCodec is installed on the cloud image but its FFmpeg libraries are absent.
lerobot_dataset.get_safe_default_codec = lambda: 'pyav'

@dataclasses.dataclass(frozen=True)
class Inputs:
    def __call__(self, d):
        state=np.asarray(d['observation/state'],dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all(): raise ValueError('Invalid SO101 state')
        front=_image(d['observation/front_image']); wrist=_image(d['observation/wrist_image'])
        out={'state':state,'image':{'base_0_rgb':front,'left_wrist_0_rgb':wrist,'right_wrist_0_rgb':np.zeros_like(front)},'image_mask':{'base_0_rgb':np.True_,'left_wrist_0_rgb':np.True_,'right_wrist_0_rgb':np.False_}}
        if 'actions' in d:
            a=np.asarray(d['actions'],dtype=np.float32)
            if a.shape[-1]!=6 or not np.isfinite(a).all(): raise ValueError('Invalid SO101 actions')
            out['actions']=a
        if 'prompt' in d: out['prompt']=d['prompt']
        return out

@dataclasses.dataclass(frozen=True)
class Outputs:
    def __call__(self,d): return {'actions':np.asarray(d['actions'])[..., :6]}

@dataclasses.dataclass(frozen=True)
class SO101Data(config.DataConfigFactory):
    def create(self,assets_dirs,model_config):
        repack=transforms.Group(inputs=[transforms.RepackTransform({'observation/front_image':'front_image','observation/wrist_image':'wrist_image','observation/state':'state','actions':'actions','prompt':'prompt'})])
        return dataclasses.replace(self.create_base_config(assets_dirs,model_config),repack_transforms=repack,data_transforms=transforms.Group(inputs=[Inputs()],outputs=[Outputs()]),model_transforms=config.ModelTransformFactory()(model_config))

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2);p.add_argument('--name',default='smoke');p.add_argument('--loader-only',action='store_true');args=p.parse_args()
    root=Path('/root/autodl-tmp/so101-medicine-run')
    model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=True,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    data=SO101Data(repo_id='local/so101_medicine_train',assets=config.AssetsConfig(assets_dir=str(root/'assets'),asset_id='so101_medicine'),base_config=config.DataConfig(prompt_from_task=True))
    cfg=dataclasses.replace(config.get_config('pi05_libero'),name='pi05_so101_medicine',exp_name=args.name,model=model,data=data,freeze_filter=model.get_freeze_filter(),ema_decay=None,weight_loader=weight_loaders.CheckpointWeightLoader('/root/pi05-base-deploy/checkpoint/params'),checkpoint_base_dir=str(root/'checkpoints'),num_train_steps=args.steps,batch_size=1,num_workers=0,wandb_enabled=False,overwrite=False,resume=False,log_interval=10,save_interval=10000,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10,peak_lr=2.5e-5,decay_steps=max(args.steps,100),decay_lr=2.5e-6))
    if args.loader_only:
        loader=data_loader.create_data_loader(cfg,shuffle=False,num_batches=1);obs,actions=next(iter(loader));print('loader OK',obs.state.shape,actions.shape,list(obs.images));return
    spec=importlib.util.spec_from_file_location('openpi_training_entry',Path('scripts/train.py').resolve());module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main(cfg)

if __name__=='__main__': main()
