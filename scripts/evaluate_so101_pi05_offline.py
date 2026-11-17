"""Teacher-forced held-out action comparison. Does not control robot hardware."""
import glob,json,dataclasses
from pathlib import Path
import cv2,numpy as np,pandas as pd
from openpi.models.pi0_config import Pi0Config
from openpi.training import config
from openpi.policies import policy_config
from so101_pi05_train import SO101Data

root=Path('/root/autodl-tmp/so101-medicine-run');src=root/'source'
checkpoint=root/'checkpoints/pi05_so101_medicine/train-500/499'
model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=True,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
data=SO101Data(repo_id='local/so101_medicine_train',assets=config.AssetsConfig(asset_id='so101_medicine'),base_config=config.DataConfig(prompt_from_task=True))
cfg=dataclasses.replace(config.get_config('pi05_libero'),model=model,data=data)
policy=policy_config.create_trained_policy(cfg,checkpoint)
episodes=pd.concat([pd.read_parquet(x) for x in glob.glob(str(src/'meta/episodes/**/*.parquet'),recursive=True)])
frames=pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(src/'data/**/*.parquet'),recursive=True))],ignore_index=True)
results=[]
for idx in [5,15,28,40,49]:
 e=episodes[episodes.episode_index==idx].iloc[0];rows=frames[frames.episode_index==idx].sort_values('frame_index').reset_index(drop=True)
 for n in np.linspace(0,len(rows)-1,8,dtype=int):
  sample=rows.iloc[n]; obs={}
  for cam in ['front','wrist']:
   k='videos/observation.images.'+cam+'/';path=src/'videos'/('observation.images.'+cam)/f"chunk-{int(e[k+'chunk_index']):03d}"/f"file-{int(e[k+'file_index']):03d}.mp4"
   c=cv2.VideoCapture(str(path));c.set(cv2.CAP_PROP_POS_FRAMES,round(float(e[k+'from_timestamp'])*30)+int(n));ok,im=c.read();c.release()
   if not ok: raise RuntimeError(f'Failed video read {idx}/{n}/{cam}')
   obs['observation/'+cam+'_image']=cv2.cvtColor(cv2.resize(im,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
  state=np.asarray(sample['observation.state'],dtype=np.float32);target=np.asarray(sample['action'],dtype=np.float32)
  obs.update({'observation/state':state,'prompt':'Pick up the white medicine bottle and place it inside the brown cardboard box.'})
  prediction=np.asarray(policy.infer(obs)['actions'])[0,:6]
  if not np.isfinite(prediction).all(): raise ValueError('Nonfinite prediction')
  results.append({'episode':idx,'frame':int(n),'predicted':prediction.tolist(),'target':target.tolist(),'state':state.tolist()})
  print('evaluated',idx,int(n),flush=True)
pred=np.asarray([r['predicted'] for r in results]);target=np.asarray([r['target'] for r in results]);state=np.asarray([r['state'] for r in results])
summary={'evaluation':'teacher-forced offline, not rollout', 'episodes':[5,15,28,40,49], 'samples':len(results),'mae_by_joint':np.abs(pred-target).mean(axis=0).tolist(),'hold_position_mae_by_joint':np.abs(state-target).mean(axis=0).tolist(),'samples_detail':results}
out=root/'offline-eval.json';out.write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='samples_detail'},indent=2))
