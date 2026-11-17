"""Remote PI05 inference and a bounded SO-101 trial using recording calibration."""
import argparse
import io
import json
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from pynput import keyboard

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


class Camera:
    def __init__(self, index):
        self.index = index
        self.frame = None
        self.stamp = 0
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened(): raise RuntimeError(f'Cannot open camera {self.index}')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_FPS,30)
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok: raise RuntimeError(f'Camera {self.index} disconnected')
                with self.lock: self.frame, self.stamp = frame, time.monotonic()
        except Exception as error: self.error = str(error)
        finally: cap.release()

    def read(self):
        if self.error: raise RuntimeError(self.error)
        with self.lock:
            if self.frame is None or time.monotonic()-self.stamp > 1:
                raise RuntimeError(f'Camera {self.index} has no fresh frame')
            return self.frame.copy()

    def close(self):
        self.stop.set(); self.thread.join(timeout=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--duration', type=float, default=35)
    p.add_argument('--url', default='http://127.0.0.1:8091')
    p.add_argument('--front', type=int, default=1)
    p.add_argument('--wrist', type=int, default=2)
    p.add_argument('--chunk-steps', type=int, default=50)
    args = p.parse_args()
    assert 1 <= args.chunk_steps <= 50 and 0 < args.duration <= 60
    out = (Path(__file__).resolve().parents[1] / 'results/blockbox-remote-20261117') / datetime.now().strftime('%H%M%S')
    out.mkdir(parents=True)
    stop = threading.Event()
    listener = keyboard.Listener(on_press=lambda key: stop.set() if key==keyboard.Key.esc else None)
    listener.start()
    robot = SO101Follower(SO101FollowerConfig(port='COM6',id='official_block_follower',use_degrees=True))
    bus = robot.bus
    cameras = []
    video = None
    enabled = False
    summary = {'execute':args.execute,'calibration':str(robot.calibration_fpath),
               'units':'body_degrees_gripper_percent','frames_sent':0,'chunks':0,'status':'initializing'}
    started = time.monotonic()
    log = open(out/'telemetry.jsonl','w',encoding='utf-8')
    def record(row):
        row['t'] = time.monotonic()-started
        log.write(json.dumps(row)+'\n'); log.flush()
    def state():
        values=bus.sync_read('Present_Position',num_retry=2)
        return np.array([values[k] for k in JOINTS],np.float32)
    def snapshot():
        return [c.read() for c in cameras]
    def infer():
        before = time.monotonic()
        images = snapshot(); values = state()
        buf = io.BytesIO()
        np.savez_compressed(buf,state=values,
                            front=cv2.cvtColor(images[0],cv2.COLOR_BGR2RGB),
                            wrist=cv2.cvtColor(images[1],cv2.COLOR_BGR2RGB))
        request=urllib.request.Request(args.url+'/infer', data=buf.getvalue(), headers={'Content-Type':'application/octet-stream'})
        with urllib.request.urlopen(request,timeout=15) as response: result=json.load(response)
        actions=np.asarray(result['actions'],np.float32)
        assert actions.shape==(50,6) and np.isfinite(actions).all()
        assert '/020000/pretrained_model' in result['checkpoint']
        elapsed=time.monotonic()-before
        record({'event':'inference','state':values.tolist(),'actions':actions.tolist(),
                'inference_s':result['inference_s'],'roundtrip_s':elapsed,'checkpoint':result['checkpoint']})
        print('INFERENCE',json.dumps({'roundtrip_s':round(elapsed,3),'gpu_s':round(result['inference_s'],3),
                                    'first_delta':(actions[0]-values).tolist()}),flush=True)
        return actions
    try:
        with urllib.request.urlopen(args.url+'/health',timeout=5) as response: health=json.load(response)
        assert health['ready'] and '/020000/pretrained_model' in health['checkpoint']
        summary['server'] = health
        bus.connect()
        if not bus.is_calibrated: raise RuntimeError('Recording calibration does not match motors; no calibration will be written')
        if any(bus.sync_read('Operating_Mode',normalize=False).values()): raise RuntimeError('Motor not in position mode')
        initial=state(); summary['initial_state']=initial.tolist()
        cameras=[Camera(args.front),Camera(args.wrist)]
        warmup_deadline=time.monotonic()+15
        while any(c.frame is None for c in cameras):
            if stop.wait(.1): raise InterruptedError('Stopped during camera startup')
            for c in cameras:
                if c.error: raise RuntimeError(c.error)
            if time.monotonic()>warmup_deadline: raise RuntimeError('Camera startup timed out')
        images=snapshot()
        for name,im in zip(['front','wrist'],images):cv2.imwrite(str(out/(name+'-start.jpg')),im)
        first=infer()
        if not args.execute:
            for _ in range(2): infer()
            summary['status']='probe_passed_no_motor_commands'
            return
        low=[]; high=[]
        for k in JOINTS:
            cal=robot.calibration[k]
            extent=(cal.range_max-cal.range_min)*180/4095
            low.append(0 if k=='gripper' else -extent)
            high.append(100 if k=='gripper' else extent)
        print('STARTING_TRIAL in 3 seconds; press ESC anywhere to stop. Duration',args.duration,flush=True)
        for _ in range(30):
            if stop.wait(.1): raise InterruptedError('Operator stopped before motion')
        bus.sync_write('Goal_Position',dict(zip(JOINTS,state().tolist())))
        bus.enable_torque(num_retry=2); enabled=True
        video=cv2.VideoWriter(str(out/'trial.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,480))
        if not video.isOpened(): raise RuntimeError('Could not create video')
        begin=time.monotonic(); deadline=begin+args.duration
        actions=first
        while not stop.is_set() and time.monotonic()<deadline:
            summary['chunks']+=1
            for predicted in actions[:args.chunk_steps]:
                tick=time.monotonic()
                if stop.is_set() or tick>=deadline: break
                measured=state()
                # Keep the calibrated mechanical joint bounds, without adding a speed limiter.
                target=np.clip(predicted,low,high)
                bus.sync_write('Goal_Position',dict(zip(JOINTS,target.tolist())))
                images=snapshot(); mosaic=cv2.hconcat(images)
                video.write(mosaic)
                summary['frames_sent']+=1
                record({'event':'action','state':measured.tolist(),'target':target.tolist(),
                        'clipped':bool(np.any(target!=predicted))})
                if summary['frames_sent']%30==0:
                    cv2.imwrite(str(out/'latest.jpg'),mosaic)
                    print('TRIAL',summary['frames_sent'],'frames',round(tick-begin,1),'seconds',flush=True)
                stop.wait(max(0,1/30-(time.monotonic()-tick)))
            if stop.is_set() or time.monotonic()>=deadline: break
            actions=infer()
        summary['status']='operator_stopped' if stop.is_set() else 'trial_finished'
    except Exception as error:
        summary['status']='error'; summary['error']=repr(error)
        print('ERROR',repr(error),flush=True)
        raise
    finally:
        if bus.is_connected:
            if enabled:
                try:
                    measured=state(); bus.sync_write('Goal_Position',dict(zip(JOINTS,measured.tolist())))
                    summary['final_state']=measured.tolist(); summary['end_mode']='hold_measured_pose'
                except Exception as error:
                    summary['stop_error']=repr(error); bus.disable_torque(num_retry=2)
                    summary['end_mode']='torque_disabled_after_communication_fault'
            bus.disconnect(disable_torque=False)
        if cameras:
            try:
                for name,im in zip(['front','wrist'],snapshot()): cv2.imwrite(str(out/(name+'-end.jpg')),im)
                cv2.imwrite(str(out/'end.jpg'),cv2.hconcat(snapshot()))
            except Exception: pass
        for camera in cameras: camera.close()
        if video is not None: video.release()
        listener.stop(); log.close()
        summary['elapsed_s']=time.monotonic()-started
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('RESULT',str(out),json.dumps(summary),flush=True)


if __name__=='__main__': main()
