import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('k','/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'])
q=np.array([0.,-30.,40.,0.,0.])
for i,name in [(1,'shoulder_lift'),(2,'elbow_flex')]:
 for d in (-5,0,5):
  p=q.copy();p[i]+=d;t=k.forward_kinematics(p);print(name,d,'tip_xyz',np.round(t[:3,3],4).tolist())
