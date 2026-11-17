"""Convert checked human demonstrations to the LeRobot v2 format used by pinned OpenPI.

Run on the OpenPI environment; this deliberately requires LeRobot's v2 API.
Held-out episodes get a separate dataset and never contribute training statistics.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from so101_demo_data import JOINTS, validate_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--held-out', type=int, default=5)
    p.add_argument('--pilot-only', action='store_true', help='format check only; not a final training split')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.held_out < 1 and not args.pilot_only:
        raise ValueError('At least one episode must be reserved for evaluation')
    episodes = []
    fingerprint = None
    for path in sorted(args.raw.glob('episode-*')):
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        if meta.get('status') != 'success':
            continue
        meta, rows, state, action = validate_episode(path)
        if fingerprint is not None and fingerprint != meta['calibration_sha256']:
            raise ValueError('Do not mix episodes recorded under different calibrations')
        fingerprint = meta['calibration_sha256']
        episodes.append((path, meta, rows, state, action))
    if not episodes:
        raise ValueError('No valid successful demonstrations')
    if not args.pilot_only and len(episodes) <= args.held_out:
        raise ValueError('Not enough demonstrations for both train and held-out splits')
    # Choose episode-level holdouts once with a fixed seed, before inspecting model results.
    held = set() if args.pilot_only else set(np.random.default_rng(20261115).permutation(len(episodes))[:args.held_out].tolist())
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    features = {
        'front_image': {'dtype': 'image', 'shape': (224, 224, 3), 'names': ['height', 'width', 'channel']},
        'wrist_image': {'dtype': 'image', 'shape': (224, 224, 3), 'names': ['height', 'width', 'channel']},
        'state': {'dtype': 'float32', 'shape': (6,), 'names': JOINTS},
        'actions': {'dtype': 'float32', 'shape': (6,), 'names': JOINTS},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    split_manifest = {'pilot_only': args.pilot_only, 'calibration_sha256': fingerprint,
                      'calibration': episodes[0][1]['calibration'], 'fps': 10,
                      'action_protocol': episodes[0][1]['protocol'], 'train': [], 'held_out': []}
    for split in ('train', 'held_out'):
        selected = [episode for i, episode in enumerate(episodes) if (i in held) == (split == 'held_out')]
        if not selected:
            continue
        ds = LeRobotDataset.create(repo_id=f'local/so101_medicine_{split}', root=args.output / split,
                                  fps=10, robot_type='so101', features=features,
                                  use_videos=False, image_writer_threads=2)
        for path, meta, rows, state, action in selected:
            for i in range(len(rows)):
                ds.add_frame({'front_image': cv2.cvtColor(cv2.imread(str(path / 'front' / f'{i:06d}.jpg')), cv2.COLOR_BGR2RGB),
                              'wrist_image': cv2.cvtColor(cv2.imread(str(path / 'wrist' / f'{i:06d}.jpg')), cv2.COLOR_BGR2RGB),
                              'state': state[i].astype(np.float32), 'actions': action[i].astype(np.float32),
                              'task': meta['task']})
            ds.save_episode()
            split_manifest[split].append(path.name)
    (args.output / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2), encoding='utf-8')
    print(json.dumps({name: len(split_manifest[name]) for name in ('train', 'held_out')}, indent=2))
    print('Compute normalization statistics from train only. No training was run.')


if __name__ == '__main__':
    main()
