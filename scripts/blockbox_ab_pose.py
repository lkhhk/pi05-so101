"""Operator-driven pose capture/alignment. Writes only the current measured position."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig

JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
NAMES=['底座旋转','肩关节','肘关节','腕部俯仰','腕部旋转','夹爪']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=(Path(__file__).resolve().parents[1] / 'results/blockbox-ab-formal-20261117/reference.json'))
    args=p.parse_args()
    reference=json.loads(args.reference.read_text(encoding='utf-8')) if args.reference.exists() else None
    robot=SO101Follower(SO101FollowerConfig(port='COM6',id='official_block_follower',use_degrees=True))
    bus=robot.bus;bus.connect()
    try:
        if not bus.is_calibrated:raise RuntimeError('Recording calibration mismatch')
        if any(bus.sync_read('Operating_Mode',normalize=False).values()):raise RuntimeError('Not in position mode')
        if any(bus.sync_read('Torque_Enable',normalize=False).values()):
            raise RuntimeError('Support the arm and release torque before manual alignment')
        window=tk.Tk();window.title('方块入盒：统一起始姿态');window.geometry('680x470')
        tk.Label(window,text='请托住从臂，摆到录制时的起始姿态。\n点击锁定后保持当前姿态；确认锁定成功再移开手。',font=('Microsoft YaHei',13)).pack(pady=15)
        tk.Label(window,text='首次：保存共同起点' if reference is None else '复位：对齐已保存起点（关节±5°，夹爪±5个百分点）',font=('Microsoft YaHei',12)).pack()
        table=tk.Label(window,font=('Consolas',14),justify='left');table.pack(pady=15)
        status=tk.Label(window,font=('Microsoft YaHei',12));status.pack(pady=5)
        current=None;closed=False
        def close():
            nonlocal closed
            closed=True;window.destroy()
        def lock():
            try:
                positions=bus.sync_read('Present_Position',num_retry=2)
                values=[positions[k] for k in JOINTS]
                if reference is not None and any(abs(a-b)>t for a,b,t in zip(values,reference['state'],reference['tolerance'])):
                    messagebox.showinfo('尚未对齐','请继续手动调整，直到所有关节偏差都在允许范围内。');return
                bus.sync_write('Goal_Position',positions)
                bus.enable_torque(num_retry=2)
                if not all(bus.sync_read('Torque_Enable',normalize=False,num_retry=2).values()):
                    raise RuntimeError('Could not hold all joints; keep supporting the arm')
                data=reference or {'state':values,'tolerance':[5]*6,'joints':JOINTS,'created':datetime.now().isoformat()}
                args.reference.parent.mkdir(parents=True,exist_ok=True)
                if reference is None:args.reference.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                locked={'reference':data,'held_state':values,'time':datetime.now().isoformat(),'torque_enabled':True}
                args.reference.with_name('pose-locked.json').write_text(json.dumps(locked,ensure_ascii=False,indent=2),encoding='utf-8')
                print('POSE_LOCKED',json.dumps(locked,ensure_ascii=False),flush=True)
                button.config(state='disabled');status.config(text='已锁定，可以移开手。请回到 Codex 等待测试。',fg='green')
                window.after(2500,close)
            except Exception as error:
                status.config(text=str(error),fg='red')
        button=tk.Button(window,text='锁定当前姿态',font=('Microsoft YaHei',14),command=lock);button.pack(pady=8)
        def update():
            nonlocal current
            if closed:return
            try:
                positions=bus.sync_read('Present_Position',num_retry=2);current=[positions[k] for k in JOINTS]
                lines=[]
                for i,(name,value) in enumerate(zip(NAMES,current)):
                    target='' if reference is None else f"  目标 {reference['state'][i]:7.2f}  差值 {value-reference['state'][i]:+7.2f}"
                    lines.append(f'{name:6s} {value:7.2f}'+target)
                table.config(text='\n'.join(lines))
            except Exception as error:status.config(text=str(error),fg='red')
            window.after(200,update)
        window.protocol('WM_DELETE_WINDOW',close);update();window.mainloop()
    finally:
        if bus.is_connected:bus.disconnect(disable_torque=False)


if __name__=='__main__':main()
