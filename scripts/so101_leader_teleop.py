"""Local SO-101 relative leader teleoperation; opens paused and preserves calibration."""
import argparse
import json
import time
from pathlib import Path
import tkinter as tk

import scservo_sdk as sdk
from so101_limits import check_readback, READBACK_TOLERANCE_TICKS

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
CAL_ROOT = Path.home() / '.cache/huggingface/lerobot/calibration'


def checked(result, context):
    if result[-2:] != (0, 0):
        raise RuntimeError(f'{context}: {result}')
    return result[0] if len(result) == 3 else None


class Arm:
    def __init__(self, port, calibration):
        self.cal = json.loads(calibration.read_text(encoding='utf-8'))
        self.port = sdk.PortHandler(port)
        self.port.baudrate = 1_000_000
        self.packet = sdk.PacketHandler(0)
        if not self.port.openPort():
            raise RuntimeError(f'Cannot open {port}')
        try:
            for name in NAMES:
                c = self.cal[name]
                values = [self.read(name, a, 2) for a in (9, 11, 31)]
                values[2] = -(values[2] & 2047) if values[2] & 2048 else values[2]
                if values != [c['range_min'], c['range_max'], c['homing_offset']]:
                    raise RuntimeError(f'{port} {name}: calibration mismatch')
                if self.read(name, 33, 1) != 0:
                    raise RuntimeError(f'{port} {name}: not in position mode')
        except BaseException:
            self.close()
            raise

    def read(self, name, address, size):
        method = self.packet.read2ByteTxRx if size == 2 else self.packet.read1ByteTxRx
        return checked(method(self.port, self.cal[name]['id'], address), f'read {name}/{address}')

    def write(self, name, address, value, size=2):
        method = self.packet.write2ByteTxRx if size == 2 else self.packet.write1ByteTxRx
        checked(method(self.port, self.cal[name]['id'], address, int(value)), f'write {name}/{address}')

    def positions(self):
        return {name: self.read(name, 56, 2) for name in NAMES}

    def close(self):
        if self.port.ser:
            self.port.closePort()

    def disable_torque(self, name):
        motor_id = self.cal[name]['id']
        comm, error = self.packet.write1ByteTxRx(self.port, motor_id, 40, 0)
        if comm:
            raise RuntimeError(f'{name}: 关闭力矩通信失败 {comm}')
        # A fault bit may accompany the disable acknowledgement; verify the register.
        value, comm, error = self.packet.read1ByteTxRx(self.port, motor_id, 40)
        if comm or value != 0:
            raise RuntimeError(f'{name}: 未确认力矩关闭 ({value}, {comm}, {error})')


class Teleop:
    def __init__(self, root, leader, follower, *, step_callback=None, interval_ms=20):
        self.root, self.leader, self.follower = root, leader, follower
        self.step_callback = step_callback
        self.interval_ms = interval_ms
        self.active = False
        self.enabled = False
        self.failed = False
        self.last_time = time.monotonic()
        self.status = tk.StringVar(value='已连接，尚未启动。请托住主臂，再点击开始跟随。')
        root.title('SO-101 主臂遥控从臂 — COM5 → COM6')
        root.geometry('680x320')
        tk.Label(root, text='主臂 COM5 → 从臂 COM6', font=('Microsoft YaHei', 18)).pack(pady=12)
        tk.Label(root, text='相对跟随：从启动时两臂的姿态开始，保留初始姿态差。\n无额外软件限速；保留实际校准范围与电机故障检查。',
                 font=('Microsoft YaHei', 11)).pack()
        tk.Label(root, textvariable=self.status, wraplength=640, font=('Microsoft YaHei', 11)).pack(pady=14)
        row = tk.Frame(root)
        row.pack()
        self.start_button = tk.Button(row, text='开始跟随 / 重新对齐', command=self.start, width=23, height=2)
        self.start_button.pack(side='left', padx=8)
        tk.Button(row, text='暂停（空格）', command=self.pause, width=18, height=2).pack(side='left', padx=8)
        tk.Button(row, text='退出（Esc）', command=self.close, width=16, height=2).pack(side='left', padx=8)
        tk.Button(root, text='已托稳从臂：解除电机保持', command=self.release, height=2).pack(pady=6)
        tk.Label(root, text='暂停和退出后从臂保持；断电前请托住从臂。', font=('Microsoft YaHei', 10)).pack(pady=14)
        root.bind('<space>', lambda event: self.pause())
        root.bind('<Escape>', lambda event: self.close())
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.tick()

    def start(self):
        if self.failed:
            return
        try:
            self.active = False
            # Establish the current follower goal BEFORE enabling torque.
            current = self.follower.positions()
            self.last_observation = dict(current)
            self.previous = self.leader.positions()
            wrist_cal = self.follower.cal['wrist_flex']
            if wrist_cal['range_max'] - wrist_cal['range_min'] > 3000:
                raise RuntimeError('腕部范围覆盖接近整圈，须先完成跨零点校准检查')
            if not wrist_cal['range_min'] + 32 <= current['wrist_flex'] <= wrist_cal['range_max'] - 32:
                self.status.set('腕部接近活动边界。电机尚未启动，请托稳并将腕部放回中间姿态后再开始。')
                return
            for name in NAMES:
                if self.leader.read(name, 40, 1):
                    raise RuntimeError('主臂电机已锁定；请先检查其他控制程序。')
                c = self.follower.cal[name]
                if not c['range_min'] <= current[name] <= c['range_max']:
                    raise RuntimeError(f'{name}: 启动位置 {current[name]} 不在校准范围 '
                                       f'{c["range_min"]}..{c["range_max"]}，未启动跟随')
            # Validate the whole pose before any Goal_Position write: these
            # servos can auto-enable torque when a position goal is written.
            fresh = self.follower.positions()
            if any(abs(fresh[name] - current[name]) > 8 for name in NAMES):
                raise RuntimeError('从臂在启动检查期间移动，请放稳后再开始')
            self.enabled = True
            for name in NAMES:
                self.follower.write(name, 42, current[name])
            for name in NAMES:
                self.follower.write(name, 40, 1, 1)
            self.target = {name: float(current[name]) for name in NAMES}
            self.command = dict(self.target)
            self.last_time = time.monotonic()
            self.active = True
            self.status.set('正在跟随。缓慢移动主臂；按空格暂停，Esc 退出。')
        except Exception as error:
            self.fail(error)

    def hold(self):
        if self.enabled:
            failures = []
            for name in NAMES:
                try:
                    if self.follower.read(name, 40, 1) == 0:
                        continue
                    value = self.follower.read(name, 56, 2)
                    c = self.follower.cal[name]
                    if not c['range_min'] <= value <= c['range_max']:
                        raise RuntimeError('实际位置超出范围，不写入保持目标')
                    self.follower.write(name, 42, value)
                except Exception as error:
                    failures.append(f'{name}: {error}')
            if failures:
                raise RuntimeError('; '.join(failures))

    def release(self):
        self.active = False
        failures = []
        for name in NAMES:
            try:
                self.follower.disable_torque(name)
            except Exception as error:
                failures.append(str(error))
        self.enabled = bool(failures)
        if failures:
            self.failed = True
            self.start_button.config(state='disabled')
            self.status.set('部分电机未确认解除，请继续托稳：' + '; '.join(failures))
        else:
            self.status.set('六个电机保持已解除。请支撑好从臂，再决定是否重新启动跟随。')
        print('TORQUE_RELEASE', 'FAILED ' + '; '.join(failures) if failures else 'all motors confirmed off', flush=True)

    def pause(self):
        self.active = False
        try:
            self.hold()
            if not self.failed:
                self.status.set('已暂停，从臂保持。再次开始将从当前姿态重新跟随。')
        except Exception as error:
            self.fail(error)

    def fail(self, error):
        self.active = False
        self.failed = True
        try:
            self.save_fault_snapshot(error)
        except Exception as log_error:
            print(f'Could not save fault snapshot: {log_error}', flush=True)
        try:
            self.hold()
        except Exception:
            pass
        self.start_button.config(state='disabled')
        self.status.set(f'已停止跟随：{error}。电机可能仍保持；托稳后可点击“解除电机保持”。')
        print(f'ERROR: {error}', flush=True)

    def save_fault_snapshot(self, error, directory=None):
        if getattr(self, 'last_observation', None) is None:
            return None
        directory = Path(directory) if directory is not None else Path(__file__).resolve().parents[1] / 'results'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'so101-teleop-fault-{time.time_ns()}.json'
        record = {'error': str(error), 'state_raw': self.last_observation,
                  'previous_command': getattr(self, 'command', None),
                  'previous_target': getattr(self, 'target', None),
                  'calibration': self.follower.cal,
                  'readback_tolerance_ticks': READBACK_TOLERANCE_TICKS}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('FAULT_SNAPSHOT:', path, flush=True)
        return path

    def tick(self):
        tick_start = time.monotonic()
        if self.active:
            try:
                now = time.monotonic()
                elapsed = now - self.last_time
                if elapsed > 0.3:
                    raise RuntimeError('控制循环延迟超过 300ms')
                self.last_time = now
                positions = self.leader.positions()
                # Read both buses every cycle; a follower read failure stops commands.
                follower_positions = self.follower.positions()
                self.last_observation = dict(follower_positions)
                observed_at = time.monotonic()
                if self.step_callback is not None:
                    if observed_at - tick_start > 0.15:
                        raise RuntimeError('读取关节状态超过 150ms，本次不发送动作')
                for name in NAMES:
                    c = self.follower.cal[name]
                    check_readback(name, follower_positions[name], c)
                planned_target, planned_command = {}, {}
                for name in NAMES:
                    delta = (positions[name] - self.previous[name] + 2048) % 4096 - 2048
                    if abs(delta) > 512:
                        raise RuntimeError(f'{name}: 主臂位置跳变过大')
                    c = self.follower.cal[name]
                    scale = 1.0
                    if name == 'gripper':
                        lc = self.leader.cal[name]
                        scale = (c['range_max'] - c['range_min']) / (lc['range_max'] - lc['range_min'])
                    planned_target[name] = max(c['range_min'], min(c['range_max'], self.target[name] + delta * scale))
                    planned_command[name] = planned_target[name]
                # Validate every joint before sending the first command in this step.
                if time.monotonic() - observed_at > 0.15:
                    raise RuntimeError('关节状态已过期，本次不发送动作')
                for name in NAMES:
                    self.follower.write(name, 42, round(planned_command[name]))
                self.previous = positions
                self.target, self.command = planned_target, planned_command
                if self.step_callback is not None:
                    self.step_callback(follower_positions,
                                       {name: round(self.command[name]) for name in NAMES},
                                       tick_start, observed_at, time.monotonic())
            except Exception as error:
                self.fail(error)
        delay = self.interval_ms
        if self.step_callback is not None:
            delay = max(1, round(self.interval_ms - (time.monotonic() - tick_start) * 1000))
        self.root.after(delay, self.tick)

    def close(self):
        self.active = False
        try:
            self.hold()
        except Exception as error:
            print(f'Could not update hold position: {error}', flush=True)
        finally:
            self.leader.close()
            self.follower.close()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='read-only connection and calibration check')
    args = parser.parse_args()
    arms = []
    try:
        leader = Arm('COM5', CAL_ROOT / 'teleoperators/so_leader/my_leader_arm.json')
        arms.append(leader)
        follower = Arm('COM6', CAL_ROOT / 'robots/so_follower/my_follower_ik.json')
        arms.append(follower)
        print('Calibration verified. Leader:', leader.positions(), 'Follower:', follower.positions(), flush=True)
        if not args.check:
            root = tk.Tk()
            Teleop(root, leader, follower)
            root.mainloop()
    finally:
        for arm in arms:
            arm.close()


if __name__ == '__main__':
    main()
