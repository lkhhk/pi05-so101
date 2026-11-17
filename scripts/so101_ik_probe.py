import importlib.util
import numpy as np
p='/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py'
s=importlib.util.spec_from_file_location('so_kinematics',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=names)
q=np.array([0.,-20.,25.,30.,0.]); T=k.forward_kinematics(q); print('forward XYZ meters:',np.round(T[:3,3],5))
for i,axis in enumerate('xyz'):
 k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=names)
 k.forward_kinematics(q)
 target=T.copy(); target[i,3]+=0.002
 result=k.inverse_kinematics(q,target,orientation_weight=0.01)
 actual=k.forward_kinematics(result)
 print(axis,'degrees',np.round(result,3),'achieved delta mm',np.round((actual[:3,3]-T[:3,3])*1000,3),'error mm',round(np.linalg.norm(actual[:3,3]-target[:3,3])*1000,4))

# Read-only snapshot obtained from motor registers, converted using LeRobot's
# DEGREES normalization and the saved follower calibration.
physical_q = np.array([-18.46, -84.53, 93.49, 178.68, 6.20])
model_bounds_deg = np.rad2deg(np.array([[-1.91986,1.91986],[-1.74533,1.74533],[-1.69,1.69],[-1.65806,1.65806],[-2.74385,2.84121]]))
print("physical snapshot degrees:", physical_q)
for name, value, (lower, upper) in zip(names, physical_q, model_bounds_deg):
 if not lower <= value <= upper:
  print(f"OUTSIDE MODEL LIMIT: {name} {value:.2f} not in [{lower:.2f}, {upper:.2f}] degrees")
