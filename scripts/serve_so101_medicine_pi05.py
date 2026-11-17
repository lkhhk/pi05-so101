"""Serve the trained SO-101 model over localhost; never controls motors."""
import dataclasses
from pathlib import Path
from openpi.models.pi0_config import Pi0Config
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config
from so101_pi05_train import SO101Data

checkpoint=Path('/root/autodl-tmp/so101-medicine-run/checkpoints/pi05_so101_medicine/train-500/499')
model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=True,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
data=SO101Data(repo_id='local/so101_medicine_train',assets=config.AssetsConfig(asset_id='so101_medicine'),base_config=config.DataConfig(prompt_from_task=True))
cfg=dataclasses.replace(config.get_config('pi05_libero'),model=model,data=data)
policy=policy_config.create_trained_policy(cfg,checkpoint)
print('MODEL_READY',flush=True)
WebsocketPolicyServer(policy=policy,host='127.0.0.1',port=8002,metadata={'protocol':'so101-medicine-pi05-v1','motor_control':False}).serve_forever()
