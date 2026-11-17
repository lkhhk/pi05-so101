"""Measure zero-order-held 10 Hz arm controls against a privileged replay.

Full-rate cap actuation, forces and equality constraints are replayed from the
demonstration. Therefore this is an interface experiment, not policy rollout.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from openarm_control.config import RIGHT_ARM_ACTUATORS, UNSCREW_SCENE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    episode = np.load(args.episode, allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(UNSCREW_SCENE)
    meta = json.loads(str(episode["metadata_json"]))
    bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    model.body_pos[bottle, :2] += meta.get("bottle_offset_xy_m", [0.0, 0.0])
    data = mujoco.MjData(model)
    data.qpos[:] = episode["replay_initial_qpos"]
    data.qvel[:] = episode["replay_initial_qvel"]
    mujoco.mj_forward(model, data)
    names = [*RIGHT_ARM_ACTUATORS, "right_finger1_ctrl"]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
    if min(ids) < 0:
        raise ValueError("Missing arm actuator")
    cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cap")
    initial_height = float(data.xpos[cap, 2])
    stride = round(0.1 / model.opt.timestep)
    controls = episode["replay_control"]
    actions = episode["action"]
    if len(actions) != (len(controls) + stride - 1) // stride:
        raise ValueError("Unexpected action sampling rate")
    for i, control in enumerate(controls):
        data.ctrl[:] = control
        data.ctrl[ids] = actions[i // stride]
        data.qfrc_applied[:] = episode["replay_applied_force"][i]
        data.eq_active[:] = episode["replay_eq_active"][i]
        model.eq_data[:] = episode["replay_eq_data"][i]
        if i % stride == 0:
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    err = float(np.max(np.abs(data.qpos - episode["replay_final_qpos"])))
    rise = float(data.xpos[cap, 2] - initial_height)
    print(f"steps={len(controls)} held_rate_hz={1/(stride*model.opt.timestep):g} "
          f"cap_rise_m={rise:.8f} max_final_qpos_error={err:.8g}")
    if not np.isfinite(err) or not np.isfinite(rise):
        raise SystemExit("Non-finite replay state")


if __name__ == "__main__":
    main()
