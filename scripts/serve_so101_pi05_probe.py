"""Serve genuine pi05_base inference for observation diagnostics, without actuator commands.

No SO-101 normalization/action head has been learned. Predictions stay in the
base model's latent action coordinates and must not be sent to motor registers.
"""
import argparse
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.policy import Policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client import image_tools

PROTOCOL = 'so101-pi05-base-observation-probe-v1'
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


class ObservationTransform:
    def __init__(self):
        self.tokenizer = PaligemmaTokenizer(max_len=200)

    def __call__(self, obs):
        state = np.asarray(obs['state_normalized'], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all() or np.any(np.abs(state) > 1):
            raise ValueError('Expected six finite diagnostic coordinates in [-1, 1]')
        state = np.pad(state, (0, 26))
        images, masks = {}, {}
        for source, target in [('front_rgb', 'base_0_rgb'), ('wrist_rgb', 'left_wrist_0_rgb')]:
            if source in obs:
                arr = np.asarray(obs[source])
                if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
                    raise ValueError('Expected HWC uint8 RGB: ' + source)
                images[target] = image_tools.resize_with_pad(arr, 224, 224)
                masks[target] = np.bool_(True)
            else:
                if source == 'front_rgb':
                    raise ValueError('front_rgb is required')
                images[target] = np.zeros((224, 224, 3), dtype=np.uint8)
                masks[target] = np.bool_(False)
        images['right_wrist_0_rgb'] = np.zeros((224, 224, 3), dtype=np.uint8)
        masks['right_wrist_0_rgb'] = np.bool_(False)
        prompt = str(obs['prompt']).strip()
        if not prompt:
            raise ValueError('Task prompt is required')
        tokens, token_mask = self.tokenizer.tokenize(prompt, state)
        return {'state': state, 'image': images, 'image_mask': masks,
                'tokenized_prompt': tokens, 'tokenized_prompt_mask': token_mask}


class ProbePolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, obs):
        result = self.policy.infer(obs)
        predictions = np.asarray(result['actions'])
        if predictions.shape != (50, 32) or not np.isfinite(predictions).all():
            raise ValueError('Invalid base-model predictions')
        return {'normalized_predictions': predictions, 'executable': False,
                'policy_timing': result['policy_timing']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--port', type=int, default=8002)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not (args.checkpoint.parent / 'VERIFIED').is_file():
        raise ValueError('Download must pass official CRC32C verification first')
    config = Pi0Config(pi05=True)
    params = model_lib.restore_params(args.checkpoint / 'params', dtype=jnp.bfloat16)
    model = config.load(params)
    policy = Policy(model, transforms=[ObservationTransform()])
    metadata = {'protocol': PROTOCOL, 'checkpoint': 'pi05_base', 'executable': False,
                'state_order': JOINTS,
                'state_scaling': 'diagnostic calibration-range scaling, not learned SO-101 statistics',
                'output': '50x32 base-model normalized coordinates, no SO-101 action mapping'}
    WebsocketPolicyServer(ProbePolicy(policy), host='127.0.0.1', port=args.port,
                          metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
