"""Verify a recorded scripted episode by replaying all simulation controls."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from openarm_control.config import UNSCREW_SCENE


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
    controls = episode["replay_control"]
    for i, control in enumerate(controls):
        data.ctrl[:] = control
        data.qfrc_applied[:] = episode["replay_applied_force"][i]
        data.eq_active[:] = episode["replay_eq_active"][i]
        model.eq_data[:] = episode["replay_eq_data"][i]
        if i % round(0.1 / model.opt.timestep) == 0:
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
    err = float(np.max(np.abs(data.qpos - episode["replay_final_qpos"])))
    print(f"replayed {len(controls)} physics steps; maximum final qpos error: {err:.8g}")
    if not np.isfinite(err) or err > 1e-4:
        raise SystemExit("Replay failed: final state differs")


if __name__ == "__main__":
    main()
