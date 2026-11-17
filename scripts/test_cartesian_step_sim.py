"""Simulated input/servo checks for the experimental step program; no hardware."""
import contextlib
import io
import json
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT=Path(__file__).with_name("so101_cartesian_step.py")
CAL=json.loads((Path.home()/".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_ik.json").read_text())
NAMES=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]

class FakeServo:
 def __init__(self,release_during_recheck=False):
  self.by_id={CAL[n]["id"]:CAL[n] for n in NAMES}
  self.raw={CAL[n]["id"]:(865 if n=="wrist_flex" else int((CAL[n]["range_min"]+CAL[n]["range_max"])/2)) for n in NAMES}
  self.torque={i:0 for i in self.raw};self.writes=[]
 def read2ByteTxRx(self,p,i,a):
  c=self.by_id[i];off=c["homing_offset"];encoded=abs(off)|(2048 if off<0 else 0)
  return ({9:c["range_min"],11:c["range_max"],31:encoded,56:self.raw[i]}[a],0,0)
 def read1ByteTxRx(self,p,i,a):return (0 if a==33 else self.torque[i],0,0)
 def write2ByteTxRx(self,p,i,a,v):self.writes.append(("write2",i,a,v));return (0,0)
 def write1ByteTxRx(self,p,i,a,v):self.writes.append(("write1",i,a,v));self.torque[i]=v;return (0,0)

class FakeInput:
 def __init__(self,release_on_pump=False):self.frame=-1;self.pumps=0;self.release_on_pump=release_on_pump
 def get_count(self):return 1
 def Joystick(self,i):return self
 def init(self):pass
 def get_numaxes(self):return 6
 def get_numbuttons(self):return 11
 def get_init(self):return True
 def get_instance_id(self):return 1
 def get_axis(self,i):return -1.0 if self.frame==1 and i==1 else 0.0
 def get_button(self,i):
  if self.release_on_pump and self.frame==1 and self.pumps>=3:return 0
  return int((i==4 and self.frame in (0,1)) or (i==0 and self.frame==1))
 def get(self):
  self.frame+=1
  return [types.SimpleNamespace(type=99,instance_id=1)] if self.frame>=2 else []
 def pump(self):self.pumps+=1

class Pipe:
 def __init__(self):self.line=""
 def write(self,x):self.line=x
 def flush(self):pass
 def close(self):pass
 def readline(self):
  q=json.loads(self.line)["joints_deg"];q[0]+=0.1
  return json.dumps({"ok":True,"joints_deg":q})+"\n"
class Worker:
 def __init__(self,*a,**kw):self.stdin=Pipe();self.stdout=self.stdin
 def wait(self,timeout=None):return 0
 def terminate(self):pass

class StepTest(unittest.TestCase):
 def run_program(self,move=False,release=False):
  servo=FakeServo();game=FakeInput(release)
  pg=types.ModuleType("pygame");pg.init=lambda:None;pg.quit=lambda:None;pg.joystick=game;pg.event=game;pg.JOYDEVICEREMOVED=99
  sdk=types.ModuleType("scservo_sdk")
  class Port:
   def __init__(self,*a):self.ser=None;self.baudrate=None
   def openPort(self):self.ser=True;return True
   def closePort(self):self.ser=None
  sdk.PortHandler=Port;sdk.PacketHandler=lambda _:servo
  argv=[str(SCRIPT)]+(["--move"] if move else [])
  with mock.patch.dict(sys.modules,{"pygame":pg,"scservo_sdk":sdk}),mock.patch.object(sys,"argv",argv),mock.patch("subprocess.Popen",Worker),mock.patch("time.sleep",lambda _:None),contextlib.redirect_stdout(io.StringIO()):
   runpy.run_path(str(SCRIPT),run_name="__main__")
  return servo.writes
 def test_preview_never_writes(self):self.assertEqual(self.run_program(),[])
 def test_motion_requires_button_and_bounds(self):
  writes=self.run_program(move=True)
  self.assertEqual(len([w for w in writes if w[0]=="write1"]),5)
  self.assertEqual(len([w for w in writes if w[0]=="write2"]),10)
 def test_released_input_never_writes(self):self.assertEqual(self.run_program(move=True,release=True),[])

if __name__=="__main__":unittest.main()
