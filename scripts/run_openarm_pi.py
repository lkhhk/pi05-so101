"""Evaluate an OpenArm-compatible policy server in the assisted simulator."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from openarm_assisted_env import OpenArmAssistedEnv
from openarm_pi_adapter import OpenArmOutputs


PROTOCOL = "openarm-absolute-position-8d-v1"


class PolicyClient:
    """OpenPI protocol client allowing slow first-call model compilation."""

    def __init__(self, port):
        self.ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None,
                          open_timeout=5, ping_interval=None)
        self.metadata = msgpack_numpy.unpackb(self.ws.recv())
        self.packer = msgpack_numpy.Packer()

    def infer(self, observation):
        self.ws.send(self.packer.pack(observation))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server error: {response}")
        return msgpack_numpy.unpackb(response)

    def close(self):
        self.ws.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--steps", type=int, default=154)
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--offset-x", type=float, default=0)
    p.add_argument("--offset-y", type=float, default=0)
    p.add_argument("--video", type=Path, required=True)
    args = p.parse_args()
    if args.steps <= 0 or args.replan <= 0:
        raise ValueError("steps and replan must be positive")
    if args.video.exists():
        raise FileExistsError(args.video)
    policy = PolicyClient(args.port)
    metadata = policy.metadata
    if metadata.get("action_protocol") != PROTOCOL:
        policy.close()
        raise ValueError("Policy server does not declare the OpenArm action protocol")
    if metadata.get("mock"):
        print("MOCK policy: recorded actions, not model inference", flush=True)
    env = OpenArmAssistedEnv(offset=(args.offset_x, args.offset_y))
    images = []
    try:
        action_buffer = []
        for _ in range(args.steps):
            obs = env.observe()
            images.append(Image.fromarray(obs["observation/front_image"]))
            if not action_buffer:
                actions = OpenArmOutputs()(policy.infer(obs))["actions"]
                if actions.ndim != 2 or len(actions) < args.replan:
                    raise ValueError("Policy returned an invalid action chunk")
                action_buffer = list(actions[:args.replan])
            env.step(action_buffer.pop(0))
        metrics = env.metrics()
        args.video.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(args.video, save_all=True, append_images=images[1:],
                       duration=100, loop=0)
        print("Assisted simulator result:", metrics, "video:", args.video)
    finally:
        env.close()
        policy.close()


if __name__ == "__main__":
    main()
