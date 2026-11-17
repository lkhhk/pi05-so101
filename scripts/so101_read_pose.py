"""Read SO-101 positions and calibration without enabling torque."""
import json
from pathlib import Path
import scservo_sdk as sdk
from so101_coordinates import wrist_zero
names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
f=Path.home()/".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_ik.json"
c=json.loads(f.read_text(encoding="utf-8"))
p=sdk.PortHandler("COM6"); p.baudrate=1000000; h=sdk.PacketHandler(0)
try:
 if not p.openPort(): raise RuntimeError("Cannot open COM6")
 for name in names:
  m=c[name]; v,comm,err=h.read2ByteTxRx(p,m["id"],56)
  if comm or err: raise RuntimeError(f"Read {name}: comm={comm} error={err}")
  mid=(m["range_min"]+m["range_max"])/2
  degrees=(v-mid)*360/4095
  if name == "wrist_flex":
   degrees=((v-wrist_zero(m)+2048)%4096-2048)*360/4095
  print(f"{name}: raw={v} midpoint={mid:.1f} range={m['range_min']}..{m['range_max']} approx_deg={degrees:.2f}")
finally:
 if p.ser:p.closePort()
