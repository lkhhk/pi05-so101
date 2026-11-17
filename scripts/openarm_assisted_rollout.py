"""Experimental state-driven assistance for 10 Hz OpenArm action replay.

The helper reads MuJoCo ground-truth cap and gripper positions and creates a
weld. It is an explicitly privileged simulation experiment, not physical
contact, a learned policy, or a faithful replay of the scripted assistance.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from openarm_control.bimanual import UnscrewTask
from openarm_control.config import RIGHT_ARM_ACTUATORS, UNSCREW_SCENE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--grasp-distance", type=float, default=0.012)
    parser.add_argument("--lift-gain", type=float, default=1.0)
    parser.add_argument("--hold-initial", action="store_true", help="Negative control: repeat the first action")
    parser.add_argument("--trace", action="store_true", help="Print attachment events")
    args = parser.parse_args()
    ep = np.load(args.episode, allow_pickle=False)
    meta = json.loads(str(ep["metadata_json"]))
    model = mujoco.MjModel.from_xml_path(UNSCREW_SCENE)
    bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    model.body_pos[bottle, :2] += meta.get("bottle_offset_xy_m", [0.0, 0.0])
    data = mujoco.MjData(model)
    data.qpos[:] = ep["replay_initial_qpos"]
    data.qvel[:] = ep["replay_initial_qvel"]
    mujoco.mj_forward(model, data)
    task = UnscrewTask(model, data)
    task.bi.sync_ctrl()
    cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cap")
    lift_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cap_lift")
    turn_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cap_turn")
    lift_q = model.jnt_qposadr[lift_joint]
    turn_q = model.jnt_qposadr[turn_joint]
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right_cap")
    actuator_names = [*RIGHT_ARM_ACTUATORS, "right_finger1_ctrl"]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
    if min(cap, lift_joint, turn_joint, weld, *ids) < 0:
        raise ValueError("Missing expected scene element")
    stride = round(0.1 / model.opt.timestep)
    start_z = float(data.xpos[cap, 2])
    lift_target, attach_lift, attach_tool_z, peak_turn = 0.0, 0.0, 0.0, 0.0
    attaches = releases = 0
    min_distance = float("inf")
    for step, recorded_action in enumerate(ep["action"]):
        action = ep["action"][0] if args.hold_initial else recorded_action
        for _ in range(stride):
            data.ctrl[ids] = action
            tool, _ = task.bi.right.king.forward_kinematics()
            knob = data.xpos[cap].copy() + np.array([0.0, 0.0, 0.034])
            distance = float(np.linalg.norm(tool - knob))
            min_distance = min(min_distance, distance)
            closed = action[-1] > -0.25
            is_attached = bool(data.eq_active[weld])
            if is_attached and not closed:
                task.bi.right.detach("cap")
                releases += 1
                if args.trace:
                    print(f"release t={data.time:.3f} turn={data.qpos[turn_q]:.4f} distance={distance:.4f} lift={data.qpos[lift_q]:.4f}")
            elif not is_attached and closed and distance < args.grasp_distance:
                task._weld_cap()
                attach_lift = float(data.qpos[lift_q])
                attach_tool_z = float(tool[2])
                attaches += 1
                if args.trace:
                    print(f"attach t={data.time:.3f} turn={data.qpos[turn_q]:.4f} distance={distance:.4f} lift={data.qpos[lift_q]:.4f}")
            if data.eq_active[weld]:
                peak_turn = max(peak_turn, float(data.qpos[turn_q]))
                # Approximate screw rise and subsequent vertical lifting using
                # current simulated state, with no access to recorded helper signals.
                desired = 0.0008 * peak_turn
                upward_motion = float(tool[2]) - attach_tool_z
                # Do not interpret tiny vertical motion during re-grasping as
                # lifting the cap. Only follow a substantial upward movement
                # after significant unscrewing progress.
                if peak_turn >= 5.0 and upward_motion >= 0.015:
                    desired = max(desired, attach_lift + args.lift_gain * upward_motion)
                lift_target = max(lift_target, min(0.12, desired))
            data.ctrl[task.lift_act] = lift_target
            task.bi.grav_comp()
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    rise = float(data.xpos[cap, 2] - start_z)
    turn = float(data.qpos[turn_q])
    # A provisional benchmark for this simplified scene, not physical proof
    # of a detached threaded cap; the cap remains constrained by scene joints.
    proxy_success = rise >= 0.045 and turn >= 5.0
    print(f"cap_rise_m={rise:.6f} cap_turn_rad={float(data.qpos[turn_q]):.6f} "
          f"attaches={attaches} releases={releases} min_distance_m={min_distance:.6f} "
          f"proxy_success={proxy_success}")
    if not np.isfinite(rise):
        raise SystemExit("Non-finite simulation result")


if __name__ == "__main__":
    main()
