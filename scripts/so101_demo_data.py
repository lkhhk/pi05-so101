"""Validate SO-101 demonstration episodes and preserve a single action convention."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from so101_limits import READBACK_TOLERANCE_TICKS

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
PROTOCOL = 'so101-calibrated-range-absolute6-v1'


def calibration_hash(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()


def validate_episode(path, check_images=True):
    path = Path(path)
    meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'success' or meta.get('protocol') != PROTOCOL:
        raise ValueError(f'{path.name}: episode not marked successful or wrong protocol')
    if meta.get('fps') != 10 or meta.get('joints') != JOINTS or not meta.get('task', '').strip():
        raise ValueError(f'{path.name}: invalid sampling rate, joint order or task')
    cal = meta['calibration']
    if meta.get('calibration_sha256') != calibration_hash(cal):
        raise ValueError(f'{path.name}: calibration fingerprint mismatch')
    rows = [json.loads(line) for line in (path / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) < 30 or len(rows) != meta.get('frame_count'):
        raise ValueError(f'{path.name}: incomplete samples or fewer than 30 frames')
    state = np.asarray([row['state_raw'] for row in rows], dtype=np.float32)
    action = np.asarray([row['action_raw'] for row in rows], dtype=np.float32)
    low = np.asarray([cal[name]['range_min'] for name in JOINTS], dtype=np.float32)
    high = np.asarray([cal[name]['range_max'] for name in JOINTS], dtype=np.float32)
    if not np.all(high > low):
        raise ValueError(f'{path.name}: invalid calibration ranges')
    for name, values in [('state', state), ('action', action)]:
        if values.shape != (len(rows), 6) or not np.isfinite(values).all():
            raise ValueError(f'{path.name}: invalid {name} shape or values')
        tolerance = READBACK_TOLERANCE_TICKS if name == 'state' else 0
        if (np.any(values < low - tolerance) or np.any(values > high + tolerance)
                or np.any(values < 0) or np.any(values > 4095)):
            raise ValueError(f'{path.name}: {name} outside saved calibrated limits')
    times = np.asarray([row['observed_at'] for row in rows])
    intervals = np.diff(times)
    # Keep the time meaning of model action chunks close to the declared 10 Hz.
    if not np.isfinite(times).all() or np.any(intervals <= 0) or np.max(intervals) > 0.15:
        raise ValueError(f'{path.name}: sample timestamps contain a gap or run backwards')
    if abs(float(np.mean(intervals)) - 0.1) > 0.015:
        raise ValueError(f'{path.name}: mean sampling interval is not close to 10 Hz')
    for index, row in enumerate(rows):
        if row['index'] != index:
            raise ValueError(f'{path.name}: non-contiguous frame index')
        start, observed, sent = (row[k] for k in ('state_read_started', 'observed_at', 'command_sent_at'))
        if not np.isfinite([start, observed, sent]).all() or not start <= observed <= sent or sent - start > 0.15:
            raise ValueError(f'{path.name}: invalid observation/command timing')
        for name in ('front', 'wrist'):
            camera_time = row['camera_times'][name]
            if not np.isfinite(camera_time) or not -0.03 <= observed - camera_time <= 0.15:
                raise ValueError(f'{path.name}: stale or misaligned {name} image')
            image_path = path / name / f'{index:06d}.jpg'
            if check_images:
                image = cv2.imread(str(image_path))
                if image is None or image.shape != (224, 224, 3):
                    raise ValueError(f'{path.name}: missing/corrupt {name} frame {index}')
    return meta, rows, np.clip(2 * (state - low) / (high - low) - 1, -1, 1), 2 * (action - low) / (high - low) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    success, rejected, frames, invalid = 0, 0, 0, []
    for path in sorted(args.root.glob('episode-*')):
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        if meta.get('status') != 'success':
            rejected += 1
            continue
        try:
            _, rows, _, _ = validate_episode(path)
            success += 1
            frames += len(rows)
            print('PASS', path.name, len(rows), 'frames')
        except Exception as error:
            invalid.append(str(error))
    print(json.dumps({'valid_success_episodes': success, 'valid_frames': frames,
                      'rejected_or_incomplete_episodes': rejected, 'invalid': invalid}, indent=2))
    if invalid or success == 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
