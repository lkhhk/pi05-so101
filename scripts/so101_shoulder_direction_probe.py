import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('k','/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'])
for d in (-5,0,5):
 k.forward_kinematics(np.array([0.,-80.+d,65.,0.,0.])); print(d,'elbow z',k.robot.get_T_world_frame('lower_arm_link')[2,3])
