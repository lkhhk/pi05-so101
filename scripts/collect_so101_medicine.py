"""Human-operated SO-101 demonstrations, 10 Hz, with paired camera/state/command data.

Opens paused. A human must start following and start recording separately.
Successful episodes are marked only after the operator chooses Save success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

from so101_leader_teleop import Arm, CAL_ROOT, NAMES, Teleop
from so101_limits import READBACK_TOLERANCE_TICKS

TASK = 'Pick up the white medicine bottle and place it inside the brown cardboard box.'
PROTOCOL = 'so101-calibrated-range-absolute6-v1'


def calibration_hash(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()


def pad_image(image):
    h, w = image.shape[:2]
    scale = 224 / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    result = np.zeros((224, 224, 3), dtype=np.uint8)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (224 - nh) // 2, (224 - nw) // 2
    result[y:y + nh, x:x + nw] = resized
    return result


class Camera:
    def __init__(self, index):
        self.index, self.latest, self.error = index, None, None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera {self.index}')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while not self.stop.is_set():
                ok, image = cap.read()
                stamp = time.monotonic()
                if not ok:
                    raise RuntimeError(f'Camera {self.index} stopped producing frames')
                image = pad_image(image)
                with self.lock:
                    self.latest = stamp, image
        except Exception as error:
            self.error = str(error)
        finally:
            cap.release()

    def snapshot(self, observed_at=None):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            if self.latest is None:
                raise RuntimeError(f'Camera {self.index} warming up')
            stamp, image = self.latest
            image = image.copy()
        if observed_at is not None and not -0.03 <= observed_at - stamp <= 0.15:
            raise RuntimeError(f'Camera {self.index} frame is not aligned with current state')
        return stamp, image

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


class EpisodeWriter:
    def __init__(self, root, calibration, task, camera_indices=None):
        self.path = root / ('episode-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        self.path.mkdir(parents=True, exist_ok=False)
        for name in ('front', 'wrist'):
            (self.path / name).mkdir()
        self.metadata = {'schema_version': 1, 'protocol': PROTOCOL, 'fps': 10,
                         'task': task, 'joints': list(NAMES), 'calibration': calibration,
                         'calibration_sha256': calibration_hash(calibration),
                         'camera_indices': camera_indices or {'front': 2, 'wrist': 0},
                         'image_shape': [224, 224, 3], 'status': 'recording',
                         'state': 'pre-command follower encoder positions',
                         'state_readback_tolerance_ticks': READBACK_TOLERANCE_TICKS,
                         'action': 'absolute follower goals actually sent after calibrated-range clamping',
                         'time_base': 'time.monotonic seconds on collection computer'}
        self.write_metadata()
        self.queue = queue.Queue(maxsize=30)
        self.count, self.error = 0, None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def write_metadata(self):
        temporary = self.path / 'metadata.tmp'
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path / 'metadata.json')

    def run(self):
        try:
            with (self.path / 'samples.jsonl').open('x', encoding='utf-8') as output:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    row, images = item
                    for key, image in images.items():
                        target = self.path / key / f'{row["index"]:06d}.jpg'
                        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise RuntimeError('Could not save ' + str(target))
                    output.write(json.dumps(row) + '\n')
                    output.flush()
        except Exception as error:
            self.error = str(error)

    def add(self, state, action, start, observed, sent, cameras):
        if self.error:
            raise RuntimeError(self.error)
        frames = {name: camera.snapshot(observed) for name, camera in cameras.items()}
        row = {'index': self.count, 'state_read_started': start, 'observed_at': observed,
               'command_sent_at': sent, 'state_raw': [state[n] for n in NAMES],
               'action_raw': [action[n] for n in NAMES],
               'camera_times': {name: value[0] for name, value in frames.items()}}
        try:
            self.queue.put_nowait((row, {name: value[1] for name, value in frames.items()}))
        except queue.Full:
            raise RuntimeError('Image writer cannot keep up; episode must be discarded')
        self.count += 1

    def finish(self, success, reason):
        # Do not hang a control window if the writer has failed with a full queue.
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                if self.error:
                    break
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error = 'Image writer did not finish within 10 seconds'
        success = bool(success and not self.error and self.count >= 30)
        self.metadata.update(status='success' if success else 'rejected',
                             frame_count=self.count, reason=self.error or reason)
        self.write_metadata()
        return success


class Recorder(Teleop):
    def __init__(self, root, leader, follower, output, cameras, task):
        self.writer = None
        self.output, self.cameras, self.task = output, cameras, task
        super().__init__(root, leader, follower, step_callback=self.record_step, interval_ms=100)
        root.title('SO-101 药瓶入盒示范采集 — 启动后暂停')
        root.geometry('760x780')
        self.record_status = tk.StringVar(value='未录制。先开始跟随，确认遥操作正常，再开始录制。')
        tk.Label(root, textvariable=self.record_status, wraplength=720,
                 font=('Microsoft YaHei', 11)).pack(pady=8)
        row = tk.Frame(root)
        row.pack()
        tk.Button(row, text='开始录制（R）', command=self.begin, width=20, height=2).pack(side='left', padx=5)
        tk.Button(row, text='保存成功示范（S）', command=lambda: self.finish(True), width=20, height=2).pack(side='left', padx=5)
        tk.Button(row, text='丢弃本条（D）', command=lambda: self.finish(False), width=20, height=2).pack(side='left', padx=5)
        tk.Label(root, text='完成抓取、放入盒中、松开夹爪并撤开后，停留约 1 秒再保存。\n'
                 '录制期间暂停、断线或超时会将本条标记为无效；原始文件保留。',
                 font=('Microsoft YaHei', 10)).pack(pady=8)
        view = tk.Frame(root)
        view.pack()
        self.previews = {}
        for name, title in [('front', f'主视角：相机 {cameras["front"].index}（Web Camera）'),
                            ('wrist', f'腕部：相机 {cameras["wrist"].index}（2M）')]:
            column = tk.Frame(view)
            column.pack(side='left', padx=12)
            tk.Label(column, text=title).pack()
            self.previews[name] = tk.Label(column)
            self.previews[name].pack()
        root.bind('r', lambda event: self.begin())
        root.bind('s', lambda event: self.finish(True))
        root.bind('d', lambda event: self.finish(False))
        self.refresh()

    def start(self):
        if self.writer:
            self.record_status.set('录制期间不能重新对齐；先保存或丢弃本条。')
            return
        super().start()

    def begin(self):
        if self.writer or not self.active or self.failed:
            self.record_status.set('请先开始跟随；每次只能录制一条。')
            return
        try:
            for camera in self.cameras.values():
                camera.snapshot(time.monotonic())
            self.writer = EpisodeWriter(self.output, self.follower.cal, self.task,
                                        {name: camera.index for name, camera in self.cameras.items()})
            self.record_status.set('正在录制：完成完整任务后按 S；失败按 D。')
        except Exception as error:
            self.record_status.set(str(error))

    def record_step(self, state, action, start, observed, sent):
        if self.writer:
            if self.writer.count >= 1200:
                raise RuntimeError('本条超过 120 秒，请缩短示范并重录')
            self.writer.add(state, action, start, observed, sent, self.cameras)

    def finish(self, success):
        writer, self.writer = self.writer, None
        if writer is None:
            return
        super().pause()
        accepted = writer.finish(success and not self.failed, 'operator marked success' if success else 'operator discarded')
        self.record_status.set(f'{"已保存成功示范" if accepted else "已标记无效"}：{writer.count} 帧。\n{writer.path.name}')
        print('EPISODE', writer.path, 'success=', accepted, 'frames=', writer.count, flush=True)

    def pause(self):
        if self.writer:
            self.finish(False)
        else:
            super().pause()

    def release(self):
        if self.writer:
            self.finish(False)
        super().release()

    def fail(self, error):
        super().fail(error)
        if self.writer:
            self.finish(False)

    def refresh(self):
        for name, camera in self.cameras.items():
            try:
                _, image = camera.snapshot()
                photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                self.previews[name].configure(image=photo)
                self.previews[name].image = photo
            except Exception:
                pass
        if self.writer:
            self.record_status.set(f'正在录制：{self.writer.count} 帧 / 约 {self.writer.count / 10:.1f} 秒。S 保存成功，D 丢弃。')
        self.root.after(100, self.refresh)

    def close(self):
        if self.writer:
            self.finish(False)
        for camera in self.cameras.values():
            camera.close()
        super().close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--leader-port', default='COM5')
    p.add_argument('--follower-port', default='COM6')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/so101-medicine/raw')
    p.add_argument('--task', default=TASK)
    p.add_argument('--check', action='store_true', help='verify arms/cameras without enabling motors')
    args = p.parse_args()
    arms, cameras = [], {}
    try:
        leader = Arm(args.leader_port, CAL_ROOT / 'teleoperators/so_leader/my_leader_arm.json')
        arms.append(leader)
        follower = Arm(args.follower_port, CAL_ROOT / 'robots/so_follower/my_follower_ik.json')
        arms.append(follower)
        print('Calibration verified. Leader:', leader.positions(), 'Follower:', follower.positions(), flush=True)
        from cv2_enumerate_cameras import enumerate_cameras
        devices = list(enumerate_cameras(cv2.CAP_DSHOW))
        for role, device_name in [('front', 'Web Camera'), ('wrist', '2M')]:
            matches = [device for device in devices if device.name == device_name]
            if len(matches) != 1:
                raise RuntimeError(f'Expected exactly one {device_name}: {[(d.index, d.name) for d in devices]}')
            cameras[role] = Camera(matches[0].index)
            # Open the devices sequentially; avoid simultaneous DirectShow initialization.
            deadline = time.monotonic() + 10
            while True:
                try:
                    cameras[role].snapshot(time.monotonic())
                    break
                except RuntimeError:
                    if cameras[role].error or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            print(f'Camera {role}: {device_name}, index={matches[0].index}', flush=True)
        deadline = time.monotonic() + 10
        while True:
            try:
                for camera in cameras.values():
                    camera.snapshot(time.monotonic())
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        if args.check:
            print('CHECK PASSED: both arms and both cameras; no motor writes', flush=True)
            return
        args.output.mkdir(parents=True, exist_ok=True)
        root = tk.Tk()
        Recorder(root, leader, follower, args.output, cameras, args.task)
        root.mainloop()
    finally:
        for camera in cameras.values():
            camera.close()
        for arm in arms:
            arm.close()


if __name__ == '__main__':
    main()
