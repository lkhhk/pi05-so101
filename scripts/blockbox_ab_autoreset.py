"""Return the SO-101 to the saved evaluation pose through the existing motor driver."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']


def trajectory(current,target,duration=3,fps=30):
    current=np.asarray(current,float);target=np.asarray(target,float)
    if current.shape!=(6,) or target.shape!=(6,) or not np.isfinite([current,target]).all():
        raise ValueError('Expected six finite joint positions')
    if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid reset duration')
    u=np.linspace(0,1,max(2,math.ceil(duration*fps)+1))
    blend=10*u**3-15*u**4+6*u**5
    return current+(target-current)*blend[:,None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--duration',type=float,default=3)
    args=p.parse_args()
    ref=json.loads(args.reference.read_text(encoding='utf-8'))
    target=np.asarray(ref['state'],float)
    from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig
    from pynput import keyboard
    robot=SO101Follower(SO101FollowerConfig(port='COM6',id='official_block_follower',use_degrees=True))
    bus=robot.bus;stop=threading.Event();listener=None;enabled=False
    summary={'execute':args.execute,'reference':str(args.reference),'target':target.tolist(),
             'started':datetime.now().isoformat(),'status':'initializing','trajectory_commands':0}
    out=args.reference.parent/'resets'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    try:
        bus.connect()
        if not bus.is_calibrated:raise RuntimeError('Recording calibration mismatch')
        if any(bus.sync_read('Operating_Mode',normalize=False).values()):raise RuntimeError('Not in position mode')
        calibration=json.loads(robot.calibration_fpath.read_text())
        limits=[]
        for joint in JOINTS:
            cal=calibration[joint];extent=(cal['range_max']-cal['range_min'])*180/4095
            limits.append((0,100) if joint=='gripper' else (-extent,extent))
        if target.shape!=(6,) or not np.isfinite(target).all() or any(not low<=x<=high for x,(low,high) in zip(target,limits)):
            raise ValueError('Reference outside recorded calibration range')
        positions=bus.sync_read('Present_Position',num_retry=2)
        current=np.array([positions[k] for k in JOINTS]);summary['initial_state']=current.tolist()
        path=trajectory(current,target,args.duration)
        print('AUTO_RESET_PLAN',json.dumps({'current':current.tolist(),'target':target.tolist(),'duration_s':args.duration}),flush=True)
        if not args.execute:
            summary['status']='preview_only';return
        listener=keyboard.Listener(on_press=lambda key:stop.set() if key==keyboard.Key.esc else None);listener.start()
        print('AUTO RESET STARTS IN 3 SECONDS. Move hands away; Esc stops and holds.',flush=True)
        if stop.wait(3):raise InterruptedError('Reset cancelled before motion')
        # Refresh after countdown; never enable torque against a stale goal.
        positions=bus.sync_read('Present_Position',num_retry=2)
        current=np.array([positions[k] for k in JOINTS]);path=trajectory(current,target,args.duration)
        bus.sync_write('Goal_Position',positions);enabled=True;bus.enable_torque(num_retry=2)
        start=time.monotonic()
        for i,point in enumerate(path):
            if stop.is_set():break
            bus.sync_write('Goal_Position',dict(zip(JOINTS,point.tolist())))
            summary['trajectory_commands']+=1
            if i<len(path)-1:
                stop.wait(max(0,start+(i+1)*args.duration/(len(path)-1)-time.monotonic()))
        stop.wait(.3)
        measured=bus.sync_read('Present_Position',num_retry=2)
        values=np.array([measured[k] for k in JOINTS])
        summary['final_state']=values.tolist();summary['delta']=(values-target).tolist()
        if stop.is_set():summary['status']='operator_stopped'
        elif np.any(np.abs(values-target)>np.asarray(ref['tolerance'])):
            raise RuntimeError('Did not reach evaluation starting pose')
        else:summary['status']='reset_complete'
    except Exception as error:
        summary['status']='error';summary['error']=repr(error);raise
    finally:
        if bus.is_connected:
            try:
                if enabled:
                    try:
                        measured=bus.sync_read('Present_Position',num_retry=2)
                        bus.sync_write('Goal_Position',measured);summary['end_mode']='hold_measured_pose'
                    except Exception as error:
                        summary['hold_error']=repr(error);bus.disable_torque(num_retry=2)
            finally:bus.disconnect(disable_torque=False)
        if listener is not None:listener.stop()
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('AUTO_RESET_RESULT',json.dumps(summary,ensure_ascii=False),flush=True)
    if summary['status']!='reset_complete':raise SystemExit(1)


if __name__=='__main__':main()
