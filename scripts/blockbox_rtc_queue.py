"""Continuation validation around LeRobot's official ActionQueue."""
import math

import numpy as np
import torch
from lerobot.policies.rtc import ActionQueue, RTCConfig


class Continuation:
    def __init__(self,health,low,high,fps=30):
        self.queue=ActionQueue(RTCConfig(enabled=True))
        self.mean=np.asarray(health['action_mean'],np.float32)
        self.std=np.asarray(health['action_std'],np.float32)+1e-8
        self.low=np.asarray(low,np.float32); self.high=np.asarray(high,np.float32)
        if self.mean.shape!=(6,) or self.std.shape!=(6,) or not np.all(self.std>0):
            raise ValueError('Invalid normalization metadata')
        self.fps=fps

    def accept(self,result,snapshot,elapsed):
        now=self.queue.snapshot()
        if now.generation!=snapshot.generation: raise RuntimeError('Continuation was cancelled')
        consumed=now.cursor-snapshot.cursor
        prefix_length=0 if snapshot.model_actions is None else len(snapshot.model_actions)
        if consumed<0 or consumed>=50 or elapsed>=50/self.fps:
            raise RuntimeError('Stale RTC response')
        if prefix_length and (consumed>=prefix_length or elapsed>=prefix_length/self.fps):
            raise RuntimeError('RTC response exhausted its supplied overlap')
        canonical=np.asarray(result['actions'],np.float32)
        original=np.asarray(result['normalized_actions'],np.float32)
        if canonical.shape!=(50,6) or original.shape!=(50,6) or not np.isfinite(canonical).all() or not np.isfinite(original).all():
            raise RuntimeError('Invalid prediction shape or values')
        if not np.allclose(canonical,original*self.std+self.mean,atol=5e-4,rtol=1e-4):
            raise RuntimeError('Normalized actions disagree with motor-space actions')
        target=np.clip(canonical,self.low,self.high)
        clipped=int(np.any(target!=canonical,axis=1).sum())
        # Prefix guidance must follow the goals that will actually be sent.
        if clipped: original=(target-self.mean)/self.std
        merged=self.queue.merge(torch.from_numpy(original.copy()),torch.from_numpy(target.copy()),
                                consumed,snapshot=snapshot)
        if not merged: raise RuntimeError('Cancelled RTC continuation')
        return {'consumed_during_request':consumed,'queue_remaining':self.queue.qsize(),
                'clipped_frames':clipped,'overlap_supplied':prefix_length}

    def request_parameters(self,latency_s):
        snapshot=self.queue.snapshot()
        if snapshot.model_actions is None or len(snapshot.model_actions)==0:
            raise RuntimeError('Cannot guide a continuation without remaining actions')
        delay=math.ceil(latency_s*self.fps)+2
        horizon=min(len(snapshot.model_actions),delay+10)
        if delay>=horizon: raise RuntimeError('Insufficient remaining actions for measured latency')
        return snapshot,delay,horizon
