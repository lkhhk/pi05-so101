"""Read-only camera/servo observation and trained model prediction, no motor writes."""
import json,sys,time
from pathlib import Path
import cv2,numpy as np,scservo_sdk as sdk
from websockets.sync.client import connect

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/openpi-probe-client'))
from openpi_client import msgpack_numpy

names=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
cal_path=Path.home()/'.cache/huggingface/lerobot/calibration/robots/so_follower/official_medicine_follower.json'
cal=json.loads(cal_path.read_text(encoding='utf-8'))
p=sdk.PortHandler('COM6');p.baudrate=1000000
if not p.openPort(): raise RuntimeError('Cannot open COM6')
raw=[];state=[];diagnostics={}
try:
 motor=sdk.PacketHandler(0)
 for i,name in enumerate(names):
  c=cal[name];vals={}
  for addr,size in [(9,2),(11,2),(31,2),(33,1),(40,1),(56,2),(65,1)]:
   fn=motor.read2ByteTxRx if size==2 else motor.read1ByteTxRx
   value,comm,err=fn(p,c['id'],addr)
   if comm or err: raise RuntimeError(f'{name} register {addr}: {comm}, {err}')
   vals[addr]=value
  off=-(vals[31]&2047) if vals[31]&2048 else vals[31]
  if [vals[9],vals[11],off]!=[c['range_min'],c['range_max'],c['homing_offset']]: raise RuntimeError(f'{name} calibration mismatch')
  if vals[33]!=0 or vals[65]!=0: raise RuntimeError(f'{name} invalid operating mode or fault')
  x=vals[56];lo,hi=c['range_min'],c['range_max'];v=100*(x-lo)/(hi-lo)
  raw.append(x);state.append(v if name=='gripper' else 2*v-100);diagnostics[name]={'position':x,'torque':vals[40], 'range':[lo,hi], 'status':vals[65]}
finally:p.closePort()

images=[]
for index in [1,2]:
 cap=cv2.VideoCapture(index,cv2.CAP_DSHOW)
 try:
  if not cap.isOpened(): raise RuntimeError(f'Camera {index} cannot open')
  for _ in range(20):
   ok,img=cap.read()
   if not ok: raise RuntimeError(f'Camera {index} read failed')
  images.append(cv2.cvtColor(cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
 finally:cap.release()
obs={'observation/front_image':images[0], 'observation/wrist_image':images[1], 'observation/state':np.array(state,dtype=np.float32),'prompt':'Pick up the white medicine bottle and place it inside the brown cardboard box.'}
start=time.monotonic()
with connect('ws://127.0.0.1:8003',compression=None,max_size=None,open_timeout=10,ping_interval=None) as ws:
 meta=msgpack_numpy.unpackb(ws.recv(timeout=30))
 if meta.get('protocol')!='so101-medicine-pi05-v1' or meta.get('motor_control') is not False: raise RuntimeError('Unexpected server')
 ws.send(msgpack_numpy.packb(obs)); result=ws.recv(timeout=300)
 if isinstance(result,str): raise RuntimeError(result)
 prediction=np.asarray(msgpack_numpy.unpackb(result)['actions'],dtype=np.float32)
if prediction.shape!=(10,6) or not np.isfinite(prediction).all(): raise ValueError('Invalid prediction')
root=Path(__file__).resolve().parents[1]/'results';root.mkdir(exist_ok=True);out=root/f'so101-medicine-preview-{time.strftime("%Y%m%d-%H%M%S")}.json'
report={'motor_writes':False,'state':dict(zip(names,state)),'diagnostics':diagnostics,'predicted_first':dict(zip(names,prediction[0].tolist())),'prediction':prediction.tolist(),'elapsed_seconds':time.monotonic()-start,'cameras':{'front':1,'wrist':2},'calibration_file':str(cal_path)}
out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));print('saved:',out)
