"""Supervised Cartesian single-step experiment; --move is required for writes."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"vendor"/"pygame"))
import pygame
import scservo_sdk as sdk
from so101_coordinates import wrist_zero

NAMES=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
CAL=Path.home()/".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_ik.json"

def checked(result, what):
 if result[-2:] != (0,0): raise RuntimeError(f"{what}: {result}")
 return result[0] if len(result)==3 else None

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--move",action="store_true");args=parser.parse_args()
 cal=json.loads(CAL.read_text(encoding="utf-8"));pygame.init();pygame.joystick.init()
 if pygame.joystick.get_count()!=1: raise RuntimeError("Connect exactly one gamepad")
 j=pygame.joystick.Joystick(0);j.init()
 if j.get_numaxes()<4 or j.get_numbuttons()<5: raise RuntimeError("Unexpected gamepad")
 pygame.event.pump()
 if j.get_button(4) or j.get_button(0): raise RuntimeError("Release LB and A before starting")
 p=sdk.PortHandler("COM6");p.baudrate=1000000;h=sdk.PacketHandler(0);worker=None;enabled=False
 try:
  if not p.openPort(): raise RuntimeError("Cannot open COM6")
  for name in NAMES:
   c=cal[name];mid=c["id"]
   vals=[checked(h.read2ByteTxRx(p,mid,a),f"read {name} register {a}") for a in (9,11,31)]
   off=-(vals[2]&2047) if vals[2]&2048 else vals[2]
   if [vals[0],vals[1],off]!=[c["range_min"],c["range_max"],c["homing_offset"]]: raise RuntimeError(f"Calibration mismatch: {name}")
   if checked(h.read1ByteTxRx(p,mid,33),f"mode {name}")!=0: raise RuntimeError(f"Not position mode: {name}")
   if checked(h.read1ByteTxRx(p,mid,40),f"torque {name}")!=0: raise RuntimeError(f"Motor torque already enabled: {name}; support arm and stop other controllers")
  worker=subprocess.Popen(["wsl.exe","-d","Ubuntu-26.04","--","/home/asus/.cache/so101-ik-venv/bin/python","/mnt/e/pi0.5/scripts/so101_ik_check.py"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
  print("MODE:","MOTION" if args.move else "PREVIEW ONLY",flush=True)
  print("Support the arm. Center sticks, hold LB, move a stick, then tap A for ONE 0.3 mm step.",flush=True)
  print("Left stick X/Y, right stick vertical Z. Release LB to prevent more steps. Ctrl+C exits.",flush=True)
  print("In motion mode motors stay enabled and may hold after exit; support before power off.",flush=True)
  armed=False;was_a=False;last=0
  while True:
   events=pygame.event.get()
   if any(e.type==pygame.JOYDEVICEREMOVED and e.instance_id==j.get_instance_id() for e in events):break
   if pygame.joystick.get_count()!=1 or not j.get_init():break
   held=bool(j.get_button(4)); pressed=bool(j.get_button(0)); axes=[j.get_axis(i) for i in range(4)]
   dead=lambda x: x if abs(x)>0.35 else 0.0
   vec=[-dead(axes[1]),-dead(axes[0]),-dead(axes[3])]
   if not held:armed=False
   elif not any(vec) and not pressed:armed=True
   trigger=held and armed and pressed and not was_a and any(vec)
   was_a=pressed
   if trigger:
    started=time.monotonic();raw=[];q=[]
    for name in NAMES:
     c=cal[name];v=checked(h.read2ByteTxRx(p,c["id"],56),f"read {name}");raw.append(v)
     if not c["range_min"]<=v<=c["range_max"]:raise RuntimeError(f"Position outside calibration: {name}")
     q.append(((v-wrist_zero(c)+2048)%4096-2048)*360/4095 if name=="wrist_flex" else (v-(c["range_min"]+c["range_max"])/2)*360/4095)
    norm=math.sqrt(sum(x*x for x in vec)); delta=[v*.0003/norm for v in vec]
    worker.stdin.write(json.dumps({"joints_deg":q,"delta_m":delta})+"\n");worker.stdin.flush()
    line=worker.stdout.readline()
    if not line:raise RuntimeError("IK solver stopped")
    result=json.loads(line)
    if not result.get("ok"):print("\nREJECTED",result.get("error"),flush=True);continue
    goals=[]
    for name,deg,now in zip(NAMES,result["joints_deg"],raw):
     c=cal[name]; goal=round(wrist_zero(c)+deg*4095/360)%4096 if name=="wrist_flex" else round((c["range_min"]+c["range_max"])/2+deg*4095/360)
     if not c["range_min"]+10<=goal<=c["range_max"]-10:raise RuntimeError(f"Target too close to limit: {name}")
     if abs(goal-now)>6:raise RuntimeError(f"Target step too large or crosses encoder wrap: {name}")
     goals.append(goal)
    pygame.event.pump()
    if not j.get_button(4) or not j.get_button(0) or time.monotonic()-started>1.5:
     print("\nCancelled: input released or calculation timed out",flush=True);continue
    changes=[b-a for a,b in zip(raw,goals)]
    print("\nSTEP mm",[round(v*1000,3) for v in delta],"raw_change",changes,flush=True)
    if args.move:
     if not any(changes): print("Step rounds to zero; no motor command sent",flush=True);continue
     # Refuse to energize against stale positions if the supported arm has moved.
     fresh=[checked(h.read2ByteTxRx(p,cal[name]["id"],56),f"recheck {name}") for name in NAMES]
     if any(abs(a-b)>2 for a,b in zip(raw,fresh)):
      print("Cancelled: arm moved during calculation",flush=True);continue
     pygame.event.pump()
     if not j.get_button(4) or not j.get_button(0):print("Cancelled before writes",flush=True);continue
     if not enabled:
      # Motors are initially off. Preload present positions before enabling holding torque.
      for name,v in zip(NAMES,fresh):checked(h.write2ByteTxRx(p,cal[name]["id"],42,v),f"preload {name}")
      fresh2=[checked(h.read2ByteTxRx(p,cal[name]["id"],56),f"verify {name}") for name in NAMES]
      if any(abs(a-b)>2 for a,b in zip(fresh,fresh2)):
       print("Cancelled: arm moved while preloading",flush=True);continue
      pygame.event.pump()
      if not j.get_button(4) or not j.get_button(0):print("Cancelled before enabling torque",flush=True);continue
      for name in NAMES:
       pygame.event.pump()
       if not j.get_button(4) or not j.get_button(0):raise RuntimeError("Input released during torque enable; some motors may hold")
       checked(h.write1ByteTxRx(p,cal[name]["id"],40,1),f"enable {name}")
      enabled=True
     pygame.event.pump()
     if not j.get_button(4) or not j.get_button(0):print("Cancelled before move; motors may hold",flush=True);continue
     current=[checked(h.read2ByteTxRx(p,cal[name]["id"],56),f"final {name}") for name in NAMES]
     if any(abs(a-b)>6 for a,b in zip(current,goals)):
      print("Cancelled: arm moved before step; motors may hold",flush=True);continue
     for name,goal in zip(NAMES,goals):
      pygame.event.pump()
      if not j.get_button(4) or not j.get_button(0):raise RuntimeError("Input released during step; some motors may hold")
      checked(h.write2ByteTxRx(p,cal[name]["id"],42,goal),f"target {name}")
     print("MOTION_WRITTEN: target registers accepted; physical movement not yet verified",flush=True)
   elif time.monotonic()-last>1:
    print(f"LB={int(held)} armed={int(armed)} A={int(pressed)}  ",end="\r",flush=True);last=time.monotonic()
   time.sleep(.05)
  print("\nController disconnected; stopped issuing commands.")
 finally:
  if p.ser:p.closePort()
  if worker:
   if worker.stdin:worker.stdin.close()
   try:worker.wait(timeout=5)
   except subprocess.TimeoutExpired:worker.terminate()
  pygame.quit()

if __name__=="__main__":
 try:main()
 except KeyboardInterrupt:print("\nStopped. Enabled motors may hold; support the arm before power off.")
 except Exception as exc:print(f"\nStopped: {exc}. Enabled motors may hold; support the arm.",file=sys.stderr);sys.exit(1)
