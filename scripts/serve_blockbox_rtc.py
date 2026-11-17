"""Guided RTC for the trained adapter; loopback-only, separate from the sync server."""
import argparse
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from blockbox_rtc_protocol import CHECKPOINT,PROTOCOL,TASK,decode_image
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy,make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8092); args=parser.parse_args()
    cfg=PreTrainedConfig.from_pretrained(CHECKPOINT)
    cfg.pretrained_path=CHECKPOINT; cfg.device='cuda'; cfg.use_peft=True; cfg.gradient_checkpointing=False
    cfg.rtc_config=RTCConfig(enabled=True,execution_horizon=20,max_guidance_weight=10.0,
                             prefix_attention_schedule=RTCAttentionSchedule.EXP)
    rename={'observation.images.front':'observation.images.desk_view','observation.images.wrist':'observation.images.wrist_left'}
    metadata=LeRobotDatasetMetadata('local/so101_block_box',root='/root/autodl-tmp/so101-block-box-trainstats-20261117')
    policy=make_policy(cfg,ds_meta=metadata,rename_map=rename).merge_and_unload(safe_merge=True).eval()
    policy.requires_grad_(False)
    if not policy._rtc_enabled() or policy.model.rtc_processor is None: raise RuntimeError('RTC initialization failed')
    pre,post=make_pre_post_processors(policy.config,pretrained_path=CHECKPOINT,
        preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':rename}})
    counts={'guided':0}
    original=policy.model.rtc_processor.denoise_step
    def counted_step(*a,**kw):
        if kw.get('prev_chunk_left_over') is not None: counts['guided']+=1
        return original(*a,**kw)
    policy.model.rtc_processor.denoise_step=counted_step
    gpu_lock=threading.Lock()

    # no_grad allows the official RTC implementation's inner enable_grad block.
    # inference_mode would prevent the guided correction from working.
    @torch.no_grad()
    def infer(state,front,wrist,prefix,delay,horizon):
        batch={'observation.state':torch.from_numpy(state),'task':TASK,
               'observation.images.front':front,'observation.images.wrist':wrist}
        start=time.perf_counter(); counts['guided']=0
        batch=pre(batch)
        previous=None if len(prefix)==0 else torch.from_numpy(prefix.copy()).to('cuda')
        normalized=policy.predict_action_chunk(batch,inference_delay=delay,
                                               prev_chunk_left_over=previous,execution_horizon=horizon)
        actions=post(normalized.clone()).detach().float().cpu().numpy()[0]
        normalized=normalized.detach().float().cpu().numpy()[0]
        if actions.shape!=(50,6) or not np.isfinite(actions).all() or not np.isfinite(normalized).all():
            raise RuntimeError('Non-finite or invalid prediction')
        elapsed=time.perf_counter()-start
        return {'actions':actions.tolist(),'normalized_actions':normalized.tolist(),
                'inference_s':elapsed,'guided_steps':counts['guided']}

    z=torch.zeros(3,224,224); empty=np.empty((0,6),np.float32)
    result=infer(np.zeros(6,np.float32),z,z,empty,0,0)
    for _ in range(2):
        result=infer(np.zeros(6,np.float32),z,z,np.array(result['normalized_actions'][:35],np.float32),10,20)
        if result['guided_steps']!=10: raise RuntimeError('Guided warmup failed')
        print('GUIDED_WARMUP_OK',result['inference_s'],flush=True)
    health={'ready':True,'protocol':PROTOCOL,'checkpoint':CHECKPOINT,'rtc_enabled':True,'rtc_mode':'guided',
            'action_shape':[50,6],'fps':30,'state_units':'body_degrees_gripper_percent',
            'action_mean':np.asarray(metadata.stats['action']['mean']).tolist(),
            'action_std':np.asarray(metadata.stats['action']['std']).tolist()}

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def reply(self,status,data):
            payload=json.dumps(data).encode(); self.send_response(status)
            self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        def do_GET(self):
            self.reply(200 if self.path=='/health' else 404,health if self.path=='/health' else {'error':'Unknown endpoint'})
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length',0))
                if self.path!='/infer' or not 0<size<1_000_000: raise ValueError('Invalid request')
                with np.load(io.BytesIO(self.rfile.read(size)),allow_pickle=False) as d:
                    if str(d['protocol'])!=PROTOCOL: raise ValueError('Protocol mismatch')
                    request_id=str(d['request_id']); state=d['state'].astype(np.float32); prefix=d['prefix'].astype(np.float32)
                    delay,horizon=int(d['delay']),int(d['horizon'])
                    if state.shape!=(6,) or not np.isfinite(state).all(): raise ValueError('Invalid state')
                    if prefix.ndim!=2 or prefix.shape[1]!=6 or len(prefix)>50 or not np.isfinite(prefix).all(): raise ValueError('Invalid prefix')
                    if len(prefix) and not 0<=delay<horizon<=len(prefix): raise ValueError('Invalid overlap')
                    front,wrist=decode_image(d['front_png']),decode_image(d['wrist_png'])
                with gpu_lock: result=infer(state,front,wrist,prefix,delay,horizon)
                result.update(protocol=PROTOCOL,checkpoint=CHECKPOINT,request_id=request_id)
                self.reply(200,result)
            except Exception as error:
                traceback.print_exc(); self.reply(400,{'error':repr(error)})

    print('RTC_READY',health,flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__': main()
