"""Serve recorded actions solely to test the OpenArm client protocol.

This is NOT a learned policy and must not be used to report model performance.
"""
import argparse
from pathlib import Path

import numpy as np
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from openarm_pi_adapter import OpenArmInputs


class RecordedPolicy:
    def __init__(self, episode):
        self.actions = np.load(episode, allow_pickle=False)["action"].astype(np.float32)
        self.index = 0

    def infer(self, obs):
        OpenArmInputs()(obs)  # Validate the live client observation contract.
        index = np.minimum(np.arange(self.index, self.index + 10), len(self.actions) - 1)
        self.index += 5
        return {"actions": self.actions[index]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode", type=Path)
    p.add_argument("--port", type=int, default=8002)
    args = p.parse_args()
    WebsocketPolicyServer(policy=RecordedPolicy(args.episode), host="127.0.0.1", port=args.port,
                          metadata={"action_protocol": "openarm-absolute-position-8d-v1",
                                    "mock": True}).serve_forever()


if __name__ == "__main__":
    main()
