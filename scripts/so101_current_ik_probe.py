import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('k','/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'])
q=np.array([-1.19,-105.98,96.48,0.,-2.86]); print('Hypothetical wrist zero test only');t=k.forward_kinematics(q); dest=t.copy();dest[0,3]+=.002
out=k.inverse_kinematics(q,dest);print('joint_changes_deg',np.round(out-q,3)); print('target error mm',round(float(np.linalg.norm(k.forward_kinematics(out)[:3,3]-dest[:3,3])*1000),3))
