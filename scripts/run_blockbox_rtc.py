"""Asynchronous guided RTC; default is a no-hardware, timed queue test."""
import argparse
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from blockbox_rtc_protocol import CHECKPOINT, Remote
from blockbox_rtc_queue import Continuation
from blockbox_rtc_admission import preflight
from blockbox_trial_control import TrialControl


def validate_start_state(values,reference):
    if reference is None:return
    target=np.asarray(reference['state'],np.float32)
    tolerance=np.asarray(reference['tolerance'],np.float32)
    current=np.asarray(values,np.float32)
    if any(x.shape!=(6,) or not np.isfinite(x).all() for x in (target,tolerance,current)) or np.any(tolerance<=0):
        raise ValueError('Invalid starting-pose reference or measured state')
    delta=np.abs(current-target)
    if np.any(delta>tolerance):
        raise RuntimeError('Starting pose differs from paired reference; no motion allowed. Delta='+str(delta.tolist()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--live-observation',action='store_true',help='Read live cameras and joints without sending motor commands')
    parser.add_argument('--record-video',action='store_true',help='Include video-writing workload in a no-motion timing test')
    parser.add_argument('--duration',type=float,default=30)
    parser.add_argument('--url',default='http://127.0.0.1:8092')
    parser.add_argument('--expected-checkpoint',default=CHECKPOINT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--reference-state',type=Path)
    parser.add_argument('--front',type=int,default=1); parser.add_argument('--wrist',type=int,default=2)
    parser.add_argument('--refill',type=int,default=35)
    parser.add_argument('--observation',type=Path,default=(Path(__file__).resolve().parents[1] / 'results/blockbox-remote-20261117/155955'))
    args=parser.parse_args()
    reference=None if args.reference_state is None else json.loads(args.reference_state.read_text(encoding='utf-8'))
    def check_start(values):
        validate_start_state(values,reference)
    if not 0<args.duration<=60 or not 5<=args.refill<=40: raise ValueError('Invalid duration or refill threshold')
    torch.set_num_threads(1)
    out=args.output_dir or (Path(__file__).resolve().parents[1] / 'results/blockbox-rtc-20261117')/datetime.now().strftime('%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    log=open(out/'telemetry.jsonl','w',encoding='utf-8')
    stop=threading.Event(); listener=None; bus=None; enabled=False; cameras=[]; video=None
    control=TrialControl(out,args.duration,stop)
    remote=Remote(args.url,args.expected_checkpoint); executor=ThreadPoolExecutor(max_workers=1)
    pending=None; continuation=None; begin=time.monotonic()
    summary={'execute':args.execute,'live_observation':args.live_observation,'status':'initializing','frames':0,'refills':0,'queue_underflows':0,
             'mode':'guided_rtc','result_dir':str(out),'duration_s':args.duration}
    request_latencies=[]; guided_latencies=[]; payload_sizes=[]; max_tick=0.
    def record(event,**values):
        log.write(json.dumps({'event':event,'t':time.monotonic()-begin,**values})+'\n');log.flush()

    def request(state,images,prefix=None,delay=0,horizon=0):
        start=time.monotonic()
        response=remote.infer(state,images,prefix,delay,horizon)
        elapsed=time.monotonic()-start
        return response,elapsed

    def observation():
        if bus is None: return saved_state.copy(),[im.copy() for im in saved_images]
        positions=bus.sync_read('Present_Position',num_retry=2)
        images=[c.read() for c in cameras]
        return np.array([positions[k] for k in joints],np.float32),images

    def note_inference(result,elapsed,kind,**extra):
        request_latencies.append(elapsed);payload_sizes.append(result['request_bytes'])
        if result['guided_steps']: guided_latencies.append(elapsed)
        record('inference',kind=kind,elapsed_s=elapsed,gpu_s=result['inference_s'],
               request_bytes=result['request_bytes'],guided_steps=result['guided_steps'],
               delay_steps=result['requested_delay_steps'],execution_horizon=result['requested_horizon'],
               actions=result['actions'].tolist(),normalized_actions=result['normalized_actions'].tolist(),**extra)
        print('RTC_INFERENCE',kind,'elapsed',round(elapsed,3),'gpu',round(result['inference_s'],3),
              'bytes',result['request_bytes'],'guided_steps',result['guided_steps'],extra,flush=True)

    try:
        health=remote.health(); summary['deployment']=health
        calibration=Path.home()/'.cache/huggingface/lerobot/calibration/robots/so_follower/official_block_follower.json'
        calibration_data=json.loads(calibration.read_text())
        joints=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
        low=[];high=[]
        for joint in joints:
            cal=calibration_data[joint];extent=(cal['range_max']-cal['range_min'])*180/4095
            low.append(0 if joint=='gripper' else -extent);high.append(100 if joint=='gripper' else extent)
        continuation=Continuation(health,low,high)
        if args.execute or args.live_observation:
            from run_blockbox_remote import Camera
            from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig
            from pynput import keyboard
            listener=keyboard.Listener(on_press=lambda key:stop.set() if key==keyboard.Key.esc else None);listener.start()
            robot=SO101Follower(SO101FollowerConfig(port='COM6',id='official_block_follower',use_degrees=True))
            bus=robot.bus;bus.connect()
            if not bus.is_calibrated:raise RuntimeError('Recording calibration mismatch; refusing to calibrate')
            if any(bus.sync_read('Operating_Mode',normalize=False).values()):raise RuntimeError('Not in position mode')
            cameras=[Camera(args.front),Camera(args.wrist)]
            timeout=time.monotonic()+15
            while any(c.frame is None for c in cameras):
                if stop.wait(.1):raise InterruptedError('Operator stopped')
                for c in cameras:
                    if c.error:raise RuntimeError(c.error)
                if time.monotonic()>timeout:raise RuntimeError('Camera startup timed out')
        else:
            saved_state=np.array(json.loads((args.observation/'summary.json').read_text())['initial_state'],np.float32)
            saved_images=[cv2.imread(str(args.observation/(key+'-start.jpg'))) for key in ('front','wrist')]
            if any(im is None for im in saved_images):raise RuntimeError('Missing saved observation images')

        # Benchmark the actual network route and guided code before permitting torque.
        values,images=observation()
        recent=preflight(
            lambda prefix:request(values,images,prefix,0 if prefix is None else 10,0 if prefix is None else 25),
            lambda result,elapsed:note_inference(result,elapsed,'preflight'),
            args.refill,
            lambda attempt,maximum:print('PREFLIGHT_RETRY','guided_probes',attempt,
                                         'recent_max_s',round(maximum,3),flush=True),
        )
        runtime_latencies=deque(recent,maxlen=5)
        latency_budget=max(runtime_latencies)
        # A guided result spends part of its 50-step horizon during inference.
        # Require room for the *next* request too, not merely a sub-1.67s request.
        delay_budget=int(np.ceil(latency_budget*30))+2
        summary['preflight_latency_budget_s']=latency_budget
        summary['admitted_preflight_samples_s']=recent
        print('PREFLIGHT_PASS','delay_budget',delay_budget,'refill',args.refill,flush=True)
        if args.execute:
            print('MOTION STARTS IN 3 SECONDS; ESC stops and holds the current pose',flush=True)
            if stop.wait(3):raise InterruptedError('Stopped before motion')

        # Refresh after preflight/countdown so the first chunk uses a current observation.
        values,images=observation(); snapshot=continuation.queue.snapshot()
        check_start(values)
        summary['initial_state']=values.tolist()
        if reference is not None:summary['reference_state']=reference
        for name,frame in zip(('front','wrist'),images):cv2.imwrite(str(out/(name+'-start.jpg')),frame)
        result,elapsed=request(values,images)
        note_inference(result,elapsed,'bootstrap')
        continuation.accept(result,snapshot,elapsed)
        if args.execute or args.record_video:
            video=cv2.VideoWriter(str(out/'trial.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,480))
            if not video.isOpened():raise RuntimeError('Could not create video')
        if args.execute:
            if stop.is_set():raise InterruptedError('Stopped before torque enable')
            positions=bus.sync_read('Present_Position',num_retry=2)
            check_start([positions[k] for k in joints])
            bus.sync_write('Goal_Position',positions);enabled=True;bus.enable_torque(num_retry=2)
        motion_start_ns=time.monotonic_ns();start=motion_start_ns/1e9
        summary['motion_start_ns']=motion_start_ns
        deadline=start+args.duration; tick=start
        control.start(motion_start_ns)
        while not stop.is_set() and time.monotonic()<deadline:
            tick_start=time.monotonic()
            if pending is not None and pending['future'].done():
                result,elapsed=pending['future'].result()
                age=time.monotonic()-pending['started']
                merged=continuation.accept(result,pending['snapshot'],age)
                note_inference(result,elapsed,'continuation',request_age_s=age,**merged)
                summary['refills']+=1;pending=None
                # Use the admitted window, not rejected startup probes.
                runtime_latencies.append(elapsed)
                latency_budget=max(runtime_latencies)
            if continuation.queue.empty():
                summary['queue_underflows']+=1
                raise RuntimeError('Action queue exhausted; stopping instead of replaying stale actions')
            values,images=observation()
            if pending is None and continuation.queue.qsize()<=args.refill:
                snap,delay,horizon=continuation.request_parameters(latency_budget)
                pending={'snapshot':snap,'started':time.monotonic()}
                pending['future']=executor.submit(request,values.copy(),[im.copy() for im in images],
                                                   snap.model_actions.numpy().copy(),delay,horizon)
            item=continuation.queue.get_with_provenance()
            if item is None:raise RuntimeError('No valid action')
            target=item[0].numpy()
            if stop.is_set():break
            if args.execute:bus.sync_write('Goal_Position',dict(zip(joints,target.tolist())))
            summary['frames']+=1
            record('action' if args.execute else 'simulated_tick',index=summary['frames'],state=values.tolist(),
                   target=target.tolist(),queue_remaining=continuation.queue.qsize())
            if video is not None:video.write(cv2.hconcat(images))
            if summary['frames']%30==0:
                print('RTC_TICK',summary['frames'],'queue',continuation.queue.qsize(),flush=True)
                if args.execute:cv2.imwrite(str(out/'latest.jpg'),cv2.hconcat(images))
            max_tick=max(max_tick,time.monotonic()-tick_start)
            tick+=1/30
            remaining=tick-time.monotonic()
            if remaining<0:tick=time.monotonic()
            else:stop.wait(remaining)
        summary['status']='success_selected' if control.success else ('operator_stopped' if stop.is_set() else ('trial_finished' if args.execute else 'dry_run_passed'))
        if control.success:summary.update(control.success)
        summary['playback_s']=time.monotonic()-start
    except Exception as error:
        if control.success:
            summary['status']='success_selected';summary.update(control.success)
            summary['stop_cleanup_note']=repr(error)
        else:
            summary['status']='error';summary['error']=repr(error)
            print('RTC_ERROR',repr(error),flush=True)
            raise
    finally:
        stop.set()
        if continuation is not None:continuation.queue.clear()
        # Motor hold runs before waiting for network worker cleanup.
        if bus is not None and bus.is_connected:
            try:
                if enabled:
                    try:
                        measured=bus.sync_read('Present_Position',num_retry=2);bus.sync_write('Goal_Position',measured)
                        summary['end_mode']='hold_measured_pose';summary['final_state']=measured
                    except Exception as error:
                        summary['hold_error']=repr(error)
                        try:
                            bus.disable_torque(num_retry=2)
                            summary['end_mode']='torque_disabled_after_communication_fault'
                        except Exception as disable_error:summary['disable_error']=repr(disable_error)
                if cameras:
                    try:cv2.imwrite(str(out/'end.jpg'),cv2.hconcat([c.read() for c in cameras]))
                    except Exception:pass
            finally:bus.disconnect(disable_torque=False)
        control.finish(summary['status'])
        if pending is not None:pending['future'].cancel()
        executor.shutdown(wait=True,cancel_futures=True);remote.close()
        for c in cameras:c.close()
        if video is not None:video.release()
        if listener is not None:listener.stop()
        summary['motor_commands_sent']=summary['frames'] if enabled else 0
        summary['max_tick_work_s']=max_tick
        if guided_latencies:
            summary['guided_roundtrip_mean_s']=float(np.mean(guided_latencies))
            summary['guided_roundtrip_p95_s']=float(np.quantile(guided_latencies,.95))
            summary['guided_roundtrip_max_s']=max(guided_latencies)
        if payload_sizes:summary['request_bytes_mean']=float(np.mean(payload_sizes))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');log.close()
        print('RTC_RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
