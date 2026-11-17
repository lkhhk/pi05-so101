import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('k','/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'])
for a in [-5,0,5]:
 t=k.forward_kinematics(np.array([float(a),-60.,30.,-30.,0.]));print('pan',a,'x_y',np.round(t[:2,3],5))
