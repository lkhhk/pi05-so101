import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('k','/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'])
for a in [-5,0,5]:
 q=np.array([0.,0.,0.,0.,float(a)]); t=k.forward_kinematics(q);print(a,'tool X',np.round(t[:3,0],3),'tool Z',np.round(t[:3,2],3))
