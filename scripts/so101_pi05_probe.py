"""Send synthetic or real SO-101 observations to pi05_base; contains no motor writes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# A minimal copy of the official client's NumPy message codec is bundled locally.
client_package = Path(__file__).resolve().parents[1] / 'vendor' / 'openpi-probe-client'
if client_package.is_dir():
    sys.path.insert(0, str(client_package))

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
PROTOCOL = 'so101-pi05-base-observation-probe-v1'


def read_state(port_name, calibration_path):
    import scservo_sdk as sdk
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    port = sdk.PortHandler(port_name)
    port.baudrate = 1_000_000
    if not port.openPort():
        raise RuntimeError('Cannot open ' + port_name)
    motor = sdk.PacketHandler(0)
    raw, normalized = [], []
    try:
        for name in JOINTS:
            c = calibration[name]
            lo, hi = c['range_min'], c['range_max']
            if not 0 <= lo < hi <= 4095:
                raise ValueError('Invalid calibration range: ' + name)
            registers = []
            for address in (9, 11, 31):
                value, comm, error = motor.read2ByteTxRx(port, c['id'], address)
                if comm or error:
                    raise RuntimeError('Cannot verify saved calibration: ' + name)
                registers.append(value)
            offset = registers[2]
            offset = -(offset & 2047) if offset & 2048 else offset
            if [registers[0], registers[1], offset] != [lo, hi, c['homing_offset']]:
                raise ValueError('Saved calibration does not match motor registers: ' + name)
            pos, comm, error = motor.read2ByteTxRx(port, c['id'], 56)
            if comm or error:
                raise RuntimeError(f'Read {name}: communication={comm}, motor={error}')
            if not lo <= pos <= hi:
                raise ValueError(f'{name} outside saved calibration range: {pos}')
            raw.append(pos)
            normalized.append(2 * (pos - lo) / (hi - lo) - 1)
    finally:
        port.closePort()
    return np.asarray(raw), np.asarray(normalized, dtype=np.float32)


def read_camera(index):
    import cv2
    # Pin the backend: MSMF and DirectShow enumerate USB devices in different orders.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {index}')
        frame = None
        for _ in range(30):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read camera {index}')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', action='store_true', help='test transport with zero state and black image')
    p.add_argument('--robot-port', default='COM6')
    p.add_argument('--calibration', type=Path, default=Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_ik.json')
    p.add_argument('--camera', type=int, default=2)
    p.add_argument('--wrist-camera', type=int, default=0)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8002)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--prompt', default='Pick up the block and place it in the box.')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Use a new output name to preserve previous results')
    if args.synthetic:
        raw, state = np.zeros(6, dtype=np.int32), np.zeros(6, dtype=np.float32)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        raw, state = read_state(args.robot_port, args.calibration)
        image = read_camera(args.camera)
    obs = {'state_normalized': state, 'front_rgb': image, 'prompt': args.prompt}
    if args.wrist_camera is not None and not args.synthetic:
        obs['wrist_rgb'] = read_camera(args.wrist_camera)
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    started = time.monotonic()
    with connect(f'ws://{args.host}:{args.port}', compression=None, max_size=None,
                 open_timeout=10, ping_interval=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
        if metadata.get('protocol') != PROTOCOL or metadata.get('executable') is not False:
            raise RuntimeError('Server is not the non-executable SO-101 base probe')
        ws.send(msgpack_numpy.packb(obs))
        response = ws.recv(timeout=args.timeout)
        if isinstance(response, str):
            raise RuntimeError(response)
        result = msgpack_numpy.unpackb(response)
    predictions = np.asarray(result['normalized_predictions'])
    if result.get('executable') is not False or predictions.shape != (50, 32) or not np.isfinite(predictions).all():
        raise ValueError('Invalid non-executable prediction response')
    summary = {'synthetic': args.synthetic, 'executable': False,
               'prediction_shape': list(predictions.shape),
               'elapsed_seconds': time.monotonic() - started,
               'policy_timing': result.get('policy_timing'), 'metadata': metadata,
               'prompt': args.prompt, 'state_raw': raw.tolist(),
               'cameras': None if args.synthetic else {'front_index': args.camera,
                           'wrist_index': args.wrist_camera,
                           'backend': 'DirectShow' if os.name == 'nt' else 'auto'},
               'calibration_file': None if args.synthetic else str(args.calibration),
               'note': 'No motor targets or torque commands were sent.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        arrays = dict(front_rgb=image, state_raw=raw, state_normalized=state,
                      normalized_predictions=predictions, prompt=args.prompt)
        if 'wrist_rgb' in obs:
            arrays['wrist_rgb'] = obs['wrist_rgb']
        np.savez_compressed(f, **arrays)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
