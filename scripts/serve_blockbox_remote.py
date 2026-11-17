"""Serve the trained adapter through a localhost-only, lossless NumPy transport."""
import argparse
import io
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    checkpoint = str(Path(args.checkpoint).resolve())
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = 'cuda'
    cfg.use_peft = True
    cfg.gradient_checkpointing = False
    rename = {'observation.images.front': 'observation.images.desk_view',
              'observation.images.wrist': 'observation.images.wrist_left'}
    metadata = LeRobotDatasetMetadata('local/so101_block_box', root='/root/autodl-tmp/so101-block-box-trainstats-20261117')
    policy = make_policy(cfg, ds_meta=metadata, rename_map=rename)
    policy = policy.merge_and_unload(safe_merge=True)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'rename_observations_processor': {'rename_map': rename}},
    )

    @torch.inference_mode()
    def infer(state, front, wrist):
        assert state.shape == (6,) and np.isfinite(state).all()
        batch = {'observation.state': torch.from_numpy(state.astype(np.float32)),
                 'task': 'Pick up the block and place it inside the box.'}
        for key, im in [('front', front), ('wrist', wrist)]:
            assert im.dtype == np.uint8 and im.shape == (480, 640, 3)
            batch['observation.images.' + key] = torch.from_numpy(im.copy()).permute(2, 0, 1).float() / 255
        start = time.perf_counter()
        batch = pre(batch)
        actions = policy.predict_action_chunk(batch)
        actions = post(actions).float().cpu().numpy()
        torch.cuda.synchronize()
        assert actions.shape == (1, 50, 6) and np.isfinite(actions).all(), actions.shape
        return actions[0], time.perf_counter() - start

    # Initialize kernels without touching hardware.
    for _ in range(2):
        actions, elapsed = infer(np.zeros(6, np.float32), np.zeros((480,640,3),np.uint8), np.zeros((480,640,3),np.uint8))
        print('WARMUP', elapsed, actions.shape, flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({'ready': True, 'checkpoint': checkpoint, 'action_shape': [50,6],
                                  'state_units': 'body_degrees_gripper_percent', 'fps': 30}).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_POST(self):
            try:
                if self.path != '/infer': raise ValueError('Unknown endpoint')
                length = int(self.headers.get('Content-Length',0))
                if not 0 < length < 8_000_000: raise ValueError('Invalid request size')
                with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as data:
                    actions, elapsed = infer(data['state'], data['front'], data['wrist'])
                payload = json.dumps({'actions': actions.tolist(), 'inference_s': elapsed,
                                      'checkpoint': checkpoint}).encode()
                self.send_response(200)
            except Exception as error:
                traceback.print_exc()
                payload = json.dumps({'error':str(error)}).encode(); self.send_response(500)
            self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(payload)))
            self.end_headers(); self.wfile.write(payload)

    print('READY', checkpoint, '127.0.0.1', args.port, flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__': main()
