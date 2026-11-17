"""Read current arm and check candidate IK in WSL; no writes or motor control."""
import json
import subprocess
from pathlib import Path
import scservo_sdk as sdk
from so101_coordinates import wrist_zero

NAMES=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CAL=Path.home()/".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_ik.json"
cal=json.loads(CAL.read_text(encoding="utf-8"))
p=sdk.PortHandler("COM6");p.baudrate=1000000;h=sdk.PacketHandler(0)
q=[]
try:
 if not p.openPort(): raise RuntimeError("Cannot open COM6")
 for name in NAMES:
  m=cal[name];v,comm,err=h.read2ByteTxRx(p,m["id"],56)
  if comm or err: raise RuntimeError(f"{name} read failed: {comm}, {err}")
  if name == "wrist_flex":
   val=((v-wrist_zero(m)+2048)%4096-2048)*360/4095
  else:
   val=(v-(m["range_min"]+m["range_max"])/2)*360/4095
  q.append(val)
finally:
 if p.ser:p.closePort()
print("Approximate joints (degrees):",dict(zip(NAMES,[round(x,2) for x in q])))
print("Wrist zero is approximate; other zero points are not independently verified.")
requests=[{"joints_deg":q,"delta_m":d} for d in ([.002,0,0],[0,.002,0],[0,0,.002])]
run=subprocess.run(["wsl.exe","-d","Ubuntu-26.04","--","/home/asus/.cache/so101-ik-venv/bin/python","/mnt/e/pi0.5/scripts/so101_ik_check.py"],input="".join(json.dumps(r)+"\n" for r in requests),capture_output=True,text=True,timeout=30)
if run.returncode: raise RuntimeError(run.stderr)
for axis,line in zip("xyz",run.stdout.splitlines()):print(axis,line)
