"""Read-only Xbox Cartesian IK preview. Never writes to or enables the motors."""
import json
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

def main():
 cal=json.loads(CAL.read_text(encoding="utf-8")); pygame.init();pygame.joystick.init()
 if pygame.joystick.get_count()!=1: raise RuntimeError("Connect exactly one Xbox controller")
 j=pygame.joystick.Joystick(0);j.init()
 if j.get_numaxes()<4 or j.get_numbuttons()<5: raise RuntimeError("Unexpected gamepad layout")
 pygame.event.pump()
 if j.get_button(4): raise RuntimeError("Release LB before starting")
 port=sdk.PortHandler("COM6");port.baudrate=1000000;h=sdk.PacketHandler(0); worker=None
 try:
  if not port.openPort(): raise RuntimeError("Cannot open COM6")
  worker=subprocess.Popen(["wsl.exe","-d","Ubuntu-26.04","--","/home/asus/.cache/so101-ik-venv/bin/python","/mnt/e/pi0.5/scripts/so101_ik_check.py"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
  print("PREVIEW ONLY - NO MOTOR COMMANDS. Hold LB after centering sticks.",flush=True)
  print("Left stick: X/Y; right stick up/down: Z. Ctrl+C exits.",flush=True)
  armed=False;last_report=0
  while True:
   events=pygame.event.get()
   if any(e.type==pygame.JOYDEVICEREMOVED and e.instance_id==j.get_instance_id() for e in events): break
   if pygame.joystick.get_count()!=1 or not j.get_init(): break
   held=bool(j.get_button(4)); a=[j.get_axis(i) for i in range(4)]
   dead=lambda x: x if abs(x)>0.35 else 0.0
   vec=[-dead(a[1]),-dead(a[0]),-dead(a[3])]
   if not held: armed=False
   elif not any(vec): armed=True
   if held and armed and any(vec):
    q=[]; raw_positions=[]
    for name in NAMES:
     c=cal[name];raw,comm,err=h.read2ByteTxRx(port,c["id"],56)
     if comm or err: raise RuntimeError(f"Read {name} failed: {comm}, {err}")
     if name=="wrist_flex": degrees=((raw-wrist_zero(c)+2048)%4096-2048)*360/4095
     else: degrees=(raw-(c["range_min"]+c["range_max"])/2)*360/4095
     q.append(degrees); raw_positions.append(raw)
    norm=max(1.0,sum(x*x for x in vec)**.5)
    delta=[v*0.001/norm for v in vec]
    worker.stdin.write(json.dumps({"joints_deg":q,"delta_m":delta})+"\n");worker.stdin.flush()
    line=worker.stdout.readline()
    if not line: raise RuntimeError("IK worker stopped")
    result=json.loads(line)
    if result.get("ok"):
     changes=[round(b-a,2) for a,b in zip(q,result["joints_deg"])]
     goals=[];rejection=None
     for name,deg,current in zip(NAMES,result["joints_deg"],raw_positions):
      c=cal[name]
      goal=round(wrist_zero(c)+deg*4095/360)%4096 if name=="wrist_flex" else round((c["range_min"]+c["range_max"])/2+deg*4095/360)
      if not c["range_min"]<=goal<=c["range_max"]:
       rejection=f"{name}: goal outside saved range";break
      if abs(goal-current)>35:
       rejection=f"{name}: raw goal jump or encoder wrap {current}->{goal}";break
      goals.append(goal)
     if rejection: print("\nREJECTED",rejection,flush=True)
     else: print("\ndelta_mm",[round(x*1000,2) for x in delta],"joint_changes_deg",changes,"raw_changes",[v-u for u,v in zip(raw_positions,goals)],"error_mm",round(result["error_mm"],3),flush=True)
    else: print("\nREJECTED",result.get("error"),flush=True)
   elif time.monotonic()-last_report>1:
    print(f"LB={int(held)} armed={int(armed)} stick={[round(x,2) for x in a]}     ",end="\r",flush=True);last_report=time.monotonic()
   time.sleep(.2)
  print("\nController disconnected; preview stopped.")
 finally:
  if port.ser: port.closePort()
  if worker:
   if worker.stdin:worker.stdin.close()
   try:worker.wait(timeout=5)
   except subprocess.TimeoutExpired:worker.terminate()
  pygame.quit()

if __name__=="__main__":
 try:main()
 except KeyboardInterrupt:print("\nPreview stopped.")
 except Exception as exc:print(f"\nPreview stopped: {exc}",file=sys.stderr);sys.exit(1)
