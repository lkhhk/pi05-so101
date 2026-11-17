"""Step-based OpenArm simulator with explicitly privileged grasp assistance.

This is a diagnostic simulator, not a physical robot control interface. Its
helper uses MuJoCo ground-truth cap pose, a grasp weld, and a lift actuator.
"""
import mujoco
import numpy as np

from openarm_control.bimanual import UnscrewTask
from openarm_control.config import RIGHT_ARM_ACTUATORS, RIGHT_ARM_JOINTS, UNSCREW_SCENE
from openarm_pi_adapter import ACTION_HIGH, ACTION_LOW
from openarm_cameras import front_camera


def _id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing scene element: {name}")
    return index


class OpenArmAssistedEnv:
    def __init__(self, offset=(0.0, 0.0), grasp_distance=0.012, lift_gain=1.0):
        self.model = mujoco.MjModel.from_xml_path(UNSCREW_SCENE)
        bottle = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
        self.model.body_pos[bottle, :2] += np.asarray(offset, dtype=float)
        self.data = mujoco.MjData(self.model)
        ready = _id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        mujoco.mj_resetDataKeyframe(self.model, self.data, ready)
        mujoco.mj_forward(self.model, self.data)
        self.task = UnscrewTask(self.model, self.data)
        self.task.bi.sync_ctrl()
        self.cap = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cap")
        self.weld = _id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right_cap")
        self.lift_q = self.model.jnt_qposadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cap_lift")]
        self.turn_q = self.model.jnt_qposadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cap_turn")]
        joints = [*RIGHT_ARM_JOINTS, "openarm_right_finger_joint1"]
        self.state_q = [self.model.jnt_qposadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints]
        self.action_ids = [_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                           for n in [*RIGHT_ARM_ACTUATORS, "right_finger1_ctrl"]]
        self.stride = round(0.1 / self.model.opt.timestep)
        if not np.isclose(self.stride * self.model.opt.timestep, 0.1):
            raise ValueError("Timestep must divide 0.1 seconds")
        self.grasp_distance, self.lift_gain = grasp_distance, lift_gain
        self.start_z = float(self.data.xpos[self.cap, 2])
        self.lift_target = self.attach_lift = self.attach_tool_z = self.peak_turn = 0.0
        self.attaches = self.releases = 0
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)
        self.front_camera = front_camera()

    def observe(self):
        """Return policy-visible data only, using two external fixed cameras."""
        mujoco.mj_forward(self.model, self.data)
        images = {}
        for camera, key in (("tablecam", "observation/table_image"),
                            (self.front_camera, "observation/front_image")):
            self.renderer.update_scene(self.data, camera=camera)
            images[key] = self.renderer.render().copy()
        return {**images, "observation/state": self.data.qpos[self.state_q].astype(np.float32).copy(),
                "prompt": "Unscrew the bottle cap"}

    def step(self, action):
        """Apply one bounded absolute target for 0.1 s of simulated time."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Action must contain exactly 8 finite values")
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)
        d, task = self.data, self.task
        for _ in range(self.stride):
            d.ctrl[self.action_ids] = action
            tool, _ = task.bi.right.king.forward_kinematics()
            knob = d.xpos[self.cap].copy() + np.array([0.0, 0.0, 0.034])
            distance = float(np.linalg.norm(tool - knob))
            closed = action[-1] > -0.25
            if d.eq_active[self.weld] and not closed:
                task.bi.right.detach("cap")
                self.releases += 1
            elif not d.eq_active[self.weld] and closed and distance < self.grasp_distance:
                task._weld_cap()
                self.attach_lift = float(d.qpos[self.lift_q])
                self.attach_tool_z = float(tool[2])
                self.attaches += 1
            if d.eq_active[self.weld]:
                self.peak_turn = max(self.peak_turn, float(d.qpos[self.turn_q]))
                desired = 0.0008 * self.peak_turn
                upward = float(tool[2]) - self.attach_tool_z
                if self.peak_turn >= 5.0 and upward >= 0.015:
                    desired = max(desired, self.attach_lift + self.lift_gain * upward)
                self.lift_target = max(self.lift_target, min(0.12, desired))
            d.ctrl[task.lift_act] = self.lift_target
            task.bi.grav_comp()
            mujoco.mj_step(self.model, d)

    def metrics(self):
        mujoco.mj_forward(self.model, self.data)
        rise = float(self.data.xpos[self.cap, 2] - self.start_z)
        turn = float(self.data.qpos[self.turn_q])
        return {"cap_rise_m": rise, "cap_turn_rad": turn,
                "proxy_success": rise >= 0.045 and turn >= 5.0,
                "attaches": self.attaches, "releases": self.releases}

    def close(self):
        self.renderer.close()
