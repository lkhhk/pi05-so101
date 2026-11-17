"""Hardware-free checks for recording integrity and paused teleoperation behavior."""
from pathlib import Path
import contextlib
import io
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from collect_so101_medicine import EpisodeWriter
from so101_demo_data import JOINTS, validate_episode
from so101_leader_teleop import Teleop
from recenter_so101_wrist import plan
import recenter_so101_wrist as recenter
from so101_coordinates import wrist_zero


CALIBRATION = {name: {'id': i + 1, 'range_min': 0, 'range_max': 4095,
                      'homing_offset': 0, 'drive_mode': 0} for i, name in enumerate(JOINTS)}


class CameraFixture:
    def snapshot(self, observed):
        return observed - 0.01, np.zeros((224, 224, 3), dtype=np.uint8)


class ArmFixture:
    def __init__(self, value):
        self.value, self.cal, self.writes = value, CALIBRATION, []
        self.torques = dict.fromkeys(JOINTS, 0)
        self.faults = set()

    def positions(self):
        return dict.fromkeys(JOINTS, self.value)

    def write(self, name, address, value):
        self.writes.append((name, address, value))

    def read(self, name, address, size):
        if name in self.faults:
            raise RuntimeError('fixture overload')
        return self.torques[name] if address == 40 else self.value


class CollectionTests(unittest.TestCase):
    def test_recenter_measured_wrapped_wrist_range_preserves_angles(self):
        calibration = {name: dict(value) for name, value in CALIBRATION.items()}
        calibration['wrist_flex'].update(homing_offset=-413, range_min=1, range_max=4095)
        updated, shift = plan(calibration, 745, 1865, 3909)
        self.assertEqual(shift, 1302)
        self.assertEqual(updated['wrist_flex']['homing_offset'], -1715)
        self.assertEqual((updated['wrist_flex']['range_min'], updated['wrist_flex']['range_max']), (1115, 3167))
        for raw in (745, 1865, 3909):
            before = (raw - wrist_zero(calibration['wrist_flex']) + 2048) % 4096 - 2048
            after = ((raw + shift) % 4096 - wrist_zero(updated['wrist_flex']) + 2048) % 4096 - 2048
            self.assertEqual(before, after)
        for name in JOINTS:
            if name != 'wrist_flex':
                self.assertEqual(updated[name], calibration[name])

    def test_calibration_entry_never_writes_goals_or_torque_enable(self):
        calibration = {name: dict(value) for name, value in CALIBRATION.items()}
        calibration['wrist_flex'].update(homing_offset=-413, range_min=1, range_max=4095)
        home = self.root / 'home'
        cal_path = home / '.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_ik.json'
        cal_path.parent.mkdir(parents=True)
        cal_path.write_text(json.dumps(calibration))
        pose_paths = []
        for label, position in [('neutral', 745), ('a', 1865), ('b', 3909)]:
            path = self.root / (label + '.json')
            path.write_text(json.dumps({'position': position, 'torque_enable': 0, 'status': 0}))
            pose_paths.append(str(path))
        registers, writes = {}, []
        for c in calibration.values():
            offset = c['homing_offset']
            registers[c['id']] = {9: c['range_min'], 11: c['range_max'],
                                  31: abs(offset) | (2048 if offset < 0 else 0),
                                  40: 0, 55: 1, 56: 745 if c['id'] == 4 else 2000, 65: 0}

        class Port:
            def __init__(self, name):
                self.ser = self
            def openPort(self):
                return True
            def closePort(self):
                pass
            def reset_input_buffer(self):
                pass

        class Packet:
            def txRxPacket(self, port, request):
                mid, address, size = request[2], request[5], request[6]
                data = list(registers[mid][address].to_bytes(size, 'little'))
                return [255, 255, mid, size + 2, 0, *data, 0], 0, 0
            def writeTxOnly(self, port, mid, address, size, data):
                value = int.from_bytes(bytes(data), 'little')
                writes.append((mid, address, value))
                if address == 31:
                    old = recenter.decode_offset(registers[mid][31])
                    new = recenter.decode_offset(value)
                    registers[mid][56] = (registers[mid][56] + old - new) % 4096
                registers[mid][address] = value
                return 0

        with (patch.object(recenter.Path, 'home', return_value=home),
              patch.object(recenter, '__file__', str(self.root / 'scripts/recenter_so101_wrist.py')),
              patch.object(recenter.sdk, 'PortHandler', Port),
              patch.object(recenter.sdk, 'PacketHandler', lambda protocol: Packet()),
              patch.object(recenter.time, 'sleep', lambda seconds: None),
              patch('sys.argv', ['recenter', '--neutral', pose_paths[0], '--endpoint-a', pose_paths[1],
                                 '--endpoint-b', pose_paths[2], '--apply']),
              contextlib.redirect_stdout(io.StringIO())):
            recenter.main()
        self.assertNotIn(42, [address for _, address, _ in writes])
        self.assertNotIn(40, [address for _, address, _ in writes])
        self.assertTrue(all(mid == 4 for mid, _, _ in writes))
        self.assertEqual(registers[4][56], 2047)
        self.assertEqual(json.loads(cal_path.read_text())['wrist_flex']['range_min'], 1115)
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / 'results/collection-tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='fixture-', dir=base)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(base.resolve())

    def tearDown(self):
        self.temporary.cleanup()

    def episode(self, success=True, wrist_overshoot=False):
        calibration = {name: dict(value) for name, value in CALIBRATION.items()}
        if wrist_overshoot:
            calibration['wrist_flex']['range_max'] = 2100
        writer = EpisodeWriter(self.root, calibration, 'TEST FIXTURE ONLY')
        cameras = {name: CameraFixture() for name in ('front', 'wrist')}
        for i in range(30):
            t = 100 + i * 0.1
            state, action = dict.fromkeys(JOINTS, 2000), dict.fromkeys(JOINTS, 2010)
            if wrist_overshoot:
                state['wrist_flex'], action['wrist_flex'] = 2103, 2100
            writer.add(state, action,
                       t - 0.01, t, t + 0.01, cameras)
        writer.finish(success, 'test fixture')
        return writer.path

    def test_saved_episode_pairs_precommand_state_with_sent_goal(self):
        meta, rows, state, action = validate_episode(self.episode())
        self.assertEqual(meta['frame_count'], 30)
        self.assertEqual(state.shape, (30, 6))
        self.assertTrue(np.all(action > state))
        self.assertLess(rows[0]['observed_at'], rows[0]['command_sent_at'])

    def test_minor_readback_deviation_preserves_raw_data_and_clips_only_training_state(self):
        _, rows, state, action = validate_episode(self.episode(wrist_overshoot=True))
        self.assertEqual(rows[0]['state_raw'][3], 2103)
        self.assertEqual(rows[0]['action_raw'][3], 2100)
        self.assertEqual(state[0, 3], 1)
        self.assertEqual(action[0, 3], 1)

    def test_missing_image_is_rejected(self):
        path = self.episode()
        (path / 'front/000010.jpg').rename(path / 'front/missing-frame.jpg')
        with self.assertRaisesRegex(ValueError, 'missing/corrupt'):
            validate_episode(path)

    def test_operator_discarded_episode_cannot_enter_training(self):
        with self.assertRaisesRegex(ValueError, 'not marked successful'):
            validate_episode(self.episode(False))

    def teleop(self):
        obj = Teleop.__new__(Teleop)
        obj.leader, obj.follower = ArmFixture(2100), ArmFixture(2000)
        obj.root = type('Root', (), {'after': lambda *args: None})()
        obj.active, obj.failed, obj.interval_ms = True, False, 100
        obj.last_time = time.monotonic() - 0.1
        obj.previous = dict.fromkeys(JOINTS, 2000)
        obj.target = dict.fromkeys(JOINTS, 2000.0)
        obj.command = dict(obj.target)
        obj.fail = lambda error: self.fail(str(error))
        return obj

    def test_callback_contains_direct_commands_without_22_degree_speed_cap(self):
        obj = self.teleop()
        captured = []
        obj.step_callback = lambda *args: captured.append(args)
        obj.tick()
        state, action, _, observed, sent = captured[0]
        self.assertEqual(state, dict.fromkeys(JOINTS, 2000))
        self.assertEqual(action, {name: goal for name, address, goal in obj.follower.writes})
        self.assertEqual(action, dict.fromkeys(JOINTS, 2100))
        self.assertLessEqual(observed, sent)

    def test_paused_loop_does_not_write_or_record(self):
        obj = self.teleop()
        obj.active = False
        obj.step_callback = lambda *args: self.fail('Paused loop recorded a frame')
        obj.tick()
        self.assertEqual(obj.follower.writes, [])

    def test_delayed_loop_stops_before_any_goal_writes(self):
        obj = self.teleop()
        errors = []
        obj.last_time = time.monotonic() - 0.5
        obj.fail = lambda error: (errors.append(str(error)), setattr(obj, 'active', False))
        obj.step_callback = lambda *args: self.fail('Stale sample recorded')
        obj.tick()
        self.assertFalse(obj.active)
        self.assertEqual(len(errors), 1)
        self.assertEqual(obj.follower.writes, [])

    def test_old_fault_snapshot_outside_corrected_wrist_range_stops_before_writes(self):
        obj = self.teleop()
        # The old snapshot is outside the corrected physical wrist range.
        obj.follower.cal = {name: dict(value) for name, value in CALIBRATION.items()}
        obj.follower.cal['wrist_flex'].update(range_min=1115, range_max=3167)
        obj.leader.value = 2000
        obj.follower.positions = lambda: {**dict.fromkeys(JOINTS, 2000), 'wrist_flex': 3877}
        obj.command['wrist_flex'] = 982
        obj.target['wrist_flex'] = 982
        errors = []
        obj.fail = lambda error: (errors.append(str(error)), setattr(obj, 'active', False))
        obj.step_callback = lambda *args: self.fail('A rejected control step must not be recorded')
        obj.tick()
        self.assertFalse(obj.active, 'Recorded fault snapshot must stop following')
        self.assertEqual(obj.follower.writes, [], 'Plan all joints before sending any goals')
        self.assertTrue(errors)

    def test_ordinary_tracking_lag_over_64_ticks_does_not_stop_following(self):
        obj = self.teleop()
        obj.leader.value = 2000
        obj.command['wrist_flex'] = 2100
        obj.target['wrist_flex'] = 2100
        captured = []
        obj.step_callback = lambda *args: captured.append(args)
        obj.tick()
        self.assertTrue(obj.active)
        self.assertEqual(captured[0][1]['wrist_flex'], 2100)
        self.assertIn(('wrist_flex', 42, 2100), obj.follower.writes)

    def test_small_readback_overshoot_does_not_stop_or_expand_goal_range(self):
        obj = self.teleop()
        obj.leader.value = 2000
        obj.follower.cal = {name: dict(value) for name, value in CALIBRATION.items()}
        obj.follower.cal['wrist_flex'].update(range_min=1115, range_max=3167)
        obj.follower.positions = lambda: {**dict.fromkeys(JOINTS, 2000), 'wrist_flex': 3170}
        obj.target['wrist_flex'] = 3167
        obj.command['wrist_flex'] = 3167
        captured = []
        obj.step_callback = lambda *args: captured.append(args)
        obj.tick()
        self.assertTrue(obj.active)
        self.assertEqual(captured[0][0]['wrist_flex'], 3170)
        self.assertEqual(captured[0][1]['wrist_flex'], 3167)

    def test_fault_snapshot_preserves_actual_position_and_calibration(self):
        obj = self.teleop()
        obj.last_observation = {**dict.fromkeys(JOINTS, 2000), 'wrist_flex': 3185}
        path = obj.save_fault_snapshot('test range failure', self.root)
        record = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(record['state_raw']['wrist_flex'], 3185)
        self.assertEqual(record['calibration'], obj.follower.cal)

    def test_direct_following_still_respects_measured_joint_range(self):
        obj = self.teleop()
        obj.leader.value = 2500
        obj.follower.cal = {name: dict(value) for name, value in CALIBRATION.items()}
        obj.follower.cal['wrist_flex'].update(range_min=1900, range_max=2200)
        captured = []
        obj.step_callback = lambda *args: captured.append(args)
        obj.tick()
        self.assertEqual(captured[0][1]['wrist_flex'], 2200)

    def test_motor_fault_stops_before_any_direct_following_writes(self):
        obj = self.teleop()
        def fault():
            raise RuntimeError('wrist_flex overload')
        obj.follower.positions = fault
        errors = []
        obj.fail = lambda error: (errors.append(str(error)), setattr(obj, 'active', False))
        obj.step_callback = lambda *args: self.fail('A faulted step cannot be recorded')
        obj.tick()
        self.assertFalse(obj.active)
        self.assertEqual(obj.follower.writes, [])
        self.assertIn('overload', errors[0])

    def test_start_checks_last_joint_before_autoenabling_goal_writes(self):
        obj = self.teleop()
        obj.follower.cal = {name: dict(value) for name, value in CALIBRATION.items()}
        obj.follower.cal['wrist_flex'].update(range_min=1900, range_max=2200)
        obj.follower.cal['gripper']['range_max'] = 1999
        errors = []
        obj.fail = lambda error: errors.append(str(error))
        obj.start()
        self.assertEqual(obj.follower.writes, [])
        self.assertEqual(len(errors), 1)
        self.assertIn('gripper', errors[0])

    def test_hold_does_not_autoenable_a_disabled_motor(self):
        obj = self.teleop()
        obj.enabled = True
        obj.follower.torques['wrist_roll'] = 1
        obj.hold()
        self.assertEqual(obj.follower.writes, [('wrist_roll', 42, 2000)])

    def test_one_fault_does_not_prevent_other_enabled_motors_holding(self):
        obj = self.teleop()
        obj.enabled = True
        obj.follower.torques = dict.fromkeys(JOINTS, 1)
        obj.follower.faults = {'wrist_flex'}
        with self.assertRaisesRegex(RuntimeError, 'overload'):
            obj.hold()
        self.assertEqual(len(obj.follower.writes), 5)
        self.assertNotIn('wrist_flex', [name for name, _, _ in obj.follower.writes])


if __name__ == '__main__':
    unittest.main()
