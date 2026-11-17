import importlib.util,numpy as np
s=importlib.util.spec_from_file_location('k','/mnt/e/pi0.5/vendor/lerobot043/lerobot/model/kinematics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k=m.RobotKinematics('/mnt/e/pi0.5/vendor/so101_kinematics_only.urdf',joint_names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'])
k.forward_kinematics(np.zeros(5))
for n in ['lower_arm_link','wrist_link','gripper_link','gripper_frame_link']:
 try:
  t=k.robot.get_T_world_frame(n);print(n, np.round(t[:3,3],4), 'z', np.round(t[:3,2],4))
 except Exception as e:print(n,str(e))
