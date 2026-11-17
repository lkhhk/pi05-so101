"""Versioned transport for the pinned block-box deployment; no motor operations."""
import http.client
import io
import json
import uuid
from urllib.parse import urlsplit

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROTOCOL = 'blockbox-guided-rtc-v1'
CHECKPOINT = '/root/autodl-tmp/so101-block-box-lora-run-20261117/checkpoints/020000/pretrained_model'
BASE_CHECKPOINT = '/root/autodl-tmp/ira-pi05-v7-full-v2-008000-blockbox-20261117'
TASK = 'Pick up the block and place it inside the box.'


def resized_png(bgr):
    """Match the model's bilinear resize; transmit only the image, not its border.

    Quantizing the interpolated pixels adds at most 0.5/255 error. The server
    restores the original model's FLOAT padding (-1 before SigLIP normalization).
    """
    if bgr.dtype != np.uint8 or bgr.shape != (480, 640, 3):
        raise ValueError('Expected 480x640 uint8 BGR camera frame')
    tensor = torch.from_numpy(bgr).permute(2,0,1).unsqueeze(0).float() / 255
    resized = F.interpolate(tensor,size=(168,224),mode='bilinear',align_corners=False)
    pixels = resized[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy()
    ok, encoded = cv2.imencode('.png',pixels,[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok: raise RuntimeError('Image encoding failed')
    return encoded


def encode_request(state, images, prefix, delay, horizon):
    state = np.asarray(state,dtype=np.float32)
    prefix = np.empty((0,6),np.float32) if prefix is None else np.asarray(prefix,dtype=np.float32)
    if state.shape != (6,) or not np.isfinite(state).all(): raise ValueError('Invalid state')
    if prefix.ndim != 2 or prefix.shape[1] != 6 or len(prefix)>50 or not np.isfinite(prefix).all():
        raise ValueError('Invalid normalized prefix')
    if len(prefix) and not 0 <= delay < horizon <= len(prefix): raise ValueError('Invalid RTC overlap')
    request_id = uuid.uuid4().hex
    buf = io.BytesIO()
    np.savez(buf,protocol=np.array(PROTOCOL),request_id=np.array(request_id),state=state,
             front_png=resized_png(images[0]),wrist_png=resized_png(images[1]),
             prefix=prefix,delay=np.int64(delay),horizon=np.int64(horizon))
    return request_id,buf.getvalue()


def decode_image(encoded):
    bgr = cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (168,224,3): raise ValueError('Invalid image payload')
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
    return F.pad(image,(0,0,28,28),value=-1)


class Remote:
    def __init__(self,url,expected_checkpoint=CHECKPOINT):
        self.expected_checkpoint=expected_checkpoint
        parsed = urlsplit(url)
        if parsed.scheme!='http' or parsed.path not in ('','/') or parsed.hostname not in ('127.0.0.1','localhost'):
            raise ValueError('Use the local SSH-forwarded HTTP endpoint')
        self.conn = http.client.HTTPConnection(parsed.hostname,parsed.port or 80,timeout=8)

    def health(self):
        self.conn.request('GET','/health')
        response=self.conn.getresponse(); payload=response.read(); data=json.loads(payload)
        if response.status!=200 or data.get('protocol')!=PROTOCOL or data.get('checkpoint')!=self.expected_checkpoint or not data.get('rtc_enabled'):
            raise RuntimeError('Wrong RTC deployment: '+str(data))
        if data.get('fps')!=30 or data.get('action_shape')!=[50,6] or data.get('state_units')!='body_degrees_gripper_percent':
            raise RuntimeError('Deployment action rate, dimensions or units mismatch')
        return data

    def infer(self,state,images,prefix=None,delay=0,horizon=0):
        request_id,payload=encode_request(state,images,prefix,delay,horizon)
        self.conn.request('POST','/infer',body=payload,headers={'Content-Type':'application/octet-stream'})
        response=self.conn.getresponse(); result=json.loads(response.read())
        if response.status!=200: raise RuntimeError(str(result))
        if result.get('protocol')!=PROTOCOL or result.get('checkpoint')!=self.expected_checkpoint or result.get('request_id')!=request_id:
            raise RuntimeError('Mismatched inference response')
        for key in ('actions','normalized_actions'):
            values=np.asarray(result[key],np.float32)
            if values.shape!=(50,6) or not np.isfinite(values).all(): raise RuntimeError('Invalid '+key)
            result[key]=values
        if prefix is not None and len(prefix) and result.get('guided_steps')!=10:
            raise RuntimeError('RTC guidance was not applied to all denoising steps')
        result['request_bytes']=len(payload)
        result['requested_delay_steps']=delay
        result['requested_horizon']=horizon
        return result

    def close(self): self.conn.close()
