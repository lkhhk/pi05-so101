"""Capture one OpenArm scripted demonstration for interface inspection.

Run inside the OpenArm checkout with its MuJoCo environment. This diagnostic
format is not yet a LeRobot dataset or a trained policy.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from openarm_control.bimanual import UnscrewTask
from openarm_control.config import RIGHT_ARM_ACTUATORS, RIGHT_ARM_JOINTS, UNSCREW_SCENE
from openarm_cameras import front_camera


def named_id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Missing {name}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offset-x", type=float, default=0.0, help="Bottle displacement in metres")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Bottle displacement in metres")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-failure", action="store_true", help="Keep failed diagnostic rollouts")
    args = parser.parse_args()
    np.random.seed(args.seed)
    model = mujoco.MjModel.from_xml_path(UNSCREW_SCENE)
    bottle = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    model.body_pos[bottle, :2] += (args.offset_x, args.offset_y)
    data = mujoco.MjData(model)
    ready = named_id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    mujoco.mj_resetDataKeyframe(model, data, ready)
    mujoco.mj_forward(model, data)

    joint_names = [*RIGHT_ARM_JOINTS, "openarm_right_finger_joint1"]
    action_names = [*RIGHT_ARM_ACTUATORS, "right_finger1_ctrl"]
    joints = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    qpos = [model.jnt_qposadr[i] for i in joints]
    actions = [named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in action_names]
    cap_act = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cap_lift_drive")
    weld = named_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right_cap")
    cap = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cap")
    stride = round(0.1 / model.opt.timestep)
    if not np.isclose(stride * model.opt.timestep, 0.1):
        raise ValueError("Simulation timestep must divide 0.1 seconds")

    renderer = mujoco.Renderer(model, height=224, width=224)
    front = front_camera()
    rows = {k: [] for k in ("time", "state", "action", "table_image", "front_image", "cap_lift_command", "weld_active", "cap_height")}
    # Full-rate privileged signals support diagnostic replay, not policy inputs.
    replay = {k: [] for k in ("control", "applied_force", "eq_active", "eq_data")}
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    original_step = mujoco.mj_step
    count = 0

    def record_step(m, d, *step_args, **step_kwargs):
        nonlocal count
        if count % stride == 0:
            # Capture state and the control about to be applied in this step.
            mujoco.mj_forward(m, d)
            rows["time"].append(d.time)
            rows["state"].append(d.qpos[qpos].copy())
            rows["action"].append(d.ctrl[actions].copy())
            rows["cap_lift_command"].append(d.ctrl[cap_act])
            rows["weld_active"].append(d.eq_active[weld])
            rows["cap_height"].append(d.xpos[cap, 2])
            for camera, key in (("tablecam", "table_image"), (front, "front_image")):
                renderer.update_scene(d, camera=camera)
                rows[key].append(renderer.render().copy())
        replay["control"].append(d.ctrl.copy())
        replay["applied_force"].append(d.qfrc_applied.copy())
        replay["eq_active"].append(d.eq_active.copy())
        replay["eq_data"].append(m.eq_data.copy())
        count += 1
        return original_step(m, d, *step_args, **step_kwargs)

    initial_height = float(data.xpos[cap, 2])
    mujoco.mj_step = record_step
    try:
        returned = bool(UnscrewTask(model, data, cap_xy=(0.30 + args.offset_x, -0.08 + args.offset_y)).run())
    finally:
        mujoco.mj_step = original_step
        renderer.close()
    mujoco.mj_forward(model, data)
    rise = float(data.xpos[cap, 2] - initial_height)
    success = returned and rise >= 0.04 and bool(rows["time"])
    if not success and not args.save_failure:
        raise RuntimeError(f"Demonstration did not pass checks: returned={returned}, rise={rise}")
    if not rows["time"]:
        raise RuntimeError("No frames captured")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays.update({f"replay_{k}": np.asarray(v) for k, v in replay.items()})
    arrays["replay_initial_qpos"] = initial_qpos
    arrays["replay_initial_qvel"] = initial_qvel
    arrays["replay_final_qpos"] = data.qpos.copy()
    arrays["metadata_json"] = np.asarray(json.dumps({
        "task": "Unscrew the bottle cap", "source": "OpenArm scripted UnscrewTask",
        "sample_hz": 10, "simulation_dt": model.opt.timestep,
        "state_joint_names": joint_names, "action_actuator_names": action_names,
        "action_semantics": "absolute position actuator commands applied at the recorded step",
        "cameras": ["tablecam", "fixed_showcase_front"], "success_cap_rise_m": rise,
        "bottle_offset_xy_m": [args.offset_x, args.offset_y],
        "seed": args.seed,
        "script_returned": returned, "success": success,
        "limitations": "fixed bottle; weld-assisted grasp; independent cap lift actuator; scripted controller",
        "auxiliary_keys": ["cap_lift_command", "weld_active", "cap_height", "replay_*"] ,
    }))
    np.savez_compressed(args.output, **arrays)
    print(f"saved {len(rows['time'])} samples to {args.output}; cap rise {rise:.4f} m")


if __name__ == "__main__":
    main()
