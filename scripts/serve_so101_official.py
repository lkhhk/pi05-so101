"""Register the local SO-101 LoRA config and run OpenPI's official serve_policy CLI."""
import dataclasses
import runpy
from openpi.models.pi0_config import Pi0Config
from openpi.training import config
from openpi.serving import websocket_policy_server
from so101_pi05_train import SO101Data

model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=True,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
data=SO101Data(repo_id='local/so101_medicine_train',assets=config.AssetsConfig(asset_id='so101_medicine'),base_config=config.DataConfig(prompt_from_task=True))
config._CONFIGS_DICT['pi05_so101_medicine']=dataclasses.replace(config.get_config('pi05_libero'),name='pi05_so101_medicine',model=model,data=data,policy_metadata={'protocol':'so101-medicine-pi05-v1','motor_control':False})

# The upstream CLI binds 0.0.0.0. Keep the model private and use an SSH tunnel.
original_server=websocket_policy_server.WebsocketPolicyServer
def local_server(*args,**kwargs):
    kwargs['host']='127.0.0.1'
    return original_server(*args,**kwargs)
websocket_policy_server.WebsocketPolicyServer=local_server
runpy.run_path('scripts/serve_policy.py',run_name='__main__')
