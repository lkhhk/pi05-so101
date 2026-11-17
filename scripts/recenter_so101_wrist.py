"""Recenter a manually measured wrist range with torque OFF and backup/rollback.

No position goals or torque-enable writes are performed. A Goal_Position write
can auto-enable these servos, so calibration must never preload a goal.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import scservo_sdk as sdk


def decode_offset(value):
    return -(value & 2047) if value & 2048 else value


def plan(calibration, neutral, endpoint_a, endpoint_b):
    wrist = calibration['wrist_flex']
    shift = 2047 - neutral
    offset = wrist['homing_offset'] - shift
    if not -2047 <= offset <= 2047:
        raise ValueError('New offset is outside signed servo register range')
    a = neutral + (endpoint_a - neutral + 2048) % 4096 - 2048
    b = neutral + (endpoint_b - neutral + 2048) % 4096 - 2048
    low, high = sorted((a, b))
    if not low < neutral < high or not 256 <= high - low <= 2800:
        raise ValueError('Measured endpoints do not form a plausible interval around neutral')
    low, high = low + shift, high + shift
    if not 100 < low < 2047 < high < 3995:
        raise ValueError('New measured range still approaches the encoder wrap boundary')
    result = copy.deepcopy(calibration)
    result['wrist_flex'].update(homing_offset=offset, range_min=low, range_max=high)
    return result, shift


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neutral', type=Path, required=True)
    p.add_argument('--endpoint-a', type=Path, required=True)
    p.add_argument('--endpoint-b', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    cal_path = Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_ik.json'
    original = json.loads(cal_path.read_text(encoding='utf-8'))
    poses = [json.loads(path.read_text()) for path in (args.neutral, args.endpoint_a, args.endpoint_b)]
    if any(pose['torque_enable'] != 0 or pose['status'] != 0 for pose in poses):
        raise ValueError('Measurements require torque off and no servo faults')
    updated, shift = plan(original, *(pose['position'] for pose in poses))
    print(json.dumps({'before': original['wrist_flex'], 'after': updated['wrist_flex'], 'encoder_shift': shift}), flush=True)
    if not args.apply:
        return
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup = Path(__file__).resolve().parents[1] / 'backups' / ('so101-wrist-recenter-' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cal_path, backup / cal_path.name)
    (backup / 'measurements.json').write_text(json.dumps(poses, indent=2))
    port = sdk.PortHandler('COM6')
    port.baudrate = 1_000_000
    packet = sdk.PacketHandler(0)
    if not port.openPort():
        raise RuntimeError('Cannot open COM6')

    def read(motor, address, size=1):
        for _ in range(3):
            port.ser.reset_input_buffer()
            request = [0] * 8
            request[2:7] = [motor, 4, 2, address, size]
            response, comm, error = packet.txRxPacket(port, request)
            if (not comm and not error and response is not None
                    and len(response) == size + 6 and response[3] == size + 2):
                return int.from_bytes(bytes(response[5:5 + size]), 'little')
            time.sleep(0.05)
        raise RuntimeError(f'Read {motor}/{address}: {(response, comm, error)}')

    def write(address, value, size=1):
        for _ in range(3):
            port.ser.reset_input_buffer()
            comm = packet.writeTxOnly(port, 4, address, size, list(value.to_bytes(size, 'little')))
            # Do not issue back-to-back EPROM writes. Verify the value even if
            # an acknowledgement was corrupt: the write itself may have landed.
            time.sleep(0.12)
            if not comm:
                try:
                    if read(4, address, size) == value and read(4, 40) == 0:
                        return
                except RuntimeError:
                    pass
        raise RuntimeError(f'Write/readback wrist/{address}: communication={comm}')

    def write_calibration(wrist):
        value = wrist['homing_offset']
        write(31, abs(value) | (2048 if value < 0 else 0), 2)
        # Widen temporarily while torque is off, then set the new validated limits.
        write(9, 0, 2)
        write(11, 4095, 2)
        write(9, wrist['range_min'], 2)
        write(11, wrist['range_max'], 2)

    changed = False
    lock = None
    try:
        for name, c in original.items():
            if read(c['id'], 40) != 0 or read(c['id'], 65) != 0:
                raise RuntimeError('All motors must be torque off and fault free')
            if [read(c['id'], 9, 2), read(c['id'], 11, 2), decode_offset(read(c['id'], 31, 2))] != [c['range_min'], c['range_max'], c['homing_offset']]:
                raise RuntimeError('Hardware calibration changed: ' + name)
        before_position = read(4, 56, 2)
        old_relative = (before_position - poses[0]['position'] + 2048) % 4096 - 2048
        measured_relative = [(pose['position'] - poses[0]['position'] + 2048) % 4096 - 2048 for pose in poses[1:]]
        if not min(measured_relative) - 16 <= old_relative <= max(measured_relative) + 16:
            raise RuntimeError('Wrist is outside the manually confirmed range; no registers changed')
        lock = read(4, 55)
        (backup / 'hardware-before.json').write_text(json.dumps({'position': before_position, 'lock': lock,
                                                               'calibration': original['wrist_flex']}, indent=2))
        write(55, 0)
        changed = True
        write_calibration(updated['wrist_flex'])
        current = read(4, 56, 2)
        if abs(current - ((before_position + shift) % 4096)) > 16:
            raise RuntimeError('Unexpected coordinate shift or wrist moved; rolling back')
        wrist = updated['wrist_flex']
        if [read(4, 9, 2), read(4, 11, 2), decode_offset(read(4, 31, 2))] != [wrist['range_min'], wrist['range_max'], wrist['homing_offset']]:
            raise RuntimeError('Calibration readback mismatch')
        for c in original.values():
            if read(c['id'], 40) != 0:
                raise RuntimeError('Unexpected torque state')
        temporary = cal_path.with_name(cal_path.name + '.' + stamp + '.tmp')
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=4), encoding='utf-8')
        temporary.replace(cal_path)
        write(55, lock)
        record = {'status': 'applied', 'backup': str(backup), 'before': original['wrist_flex'],
                  'after': wrist, 'encoder_shift': shift, 'before_position': before_position,
                  'after_position': current, 'all_torque_off': True,
                  'position_goal_writes': 0, 'power_cycle_required_before_control': True}
        (backup / 'result.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    except BaseException:
        if changed:
            try:
                write(55, 0)
                write_calibration(original['wrist_flex'])
                cal_path.write_text(json.dumps(original, ensure_ascii=False, indent=4), encoding='utf-8')
                if lock is not None:
                    write(55, lock)
                print('Rolled back original wrist calibration; torque stayed OFF', flush=True)
            except Exception as rollback_error:
                print('ROLLBACK NEEDS INSPECTION:', rollback_error, flush=True)
        raise
    finally:
        port.closePort()


if __name__ == '__main__':
    main()
