import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('k','/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
n=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=n)
for angle in [-10,0,10]:
 q=np.array([0.,0.,0.,float(angle),0.]); t=k.forward_kinematics(q);print('wrist_flex_deg',angle,'tip_xyz',np.round(t[:3,3],4).tolist())
