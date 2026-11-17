"""Interactive, deliberately tiny model-driven pilot; STOP leaves holding torque on."""
import json, subprocess, sys, threading, time, tkinter as tk
from pathlib import Path
from tkinter import messagebox
from so101_leader_teleop import Arm, NAMES, CAL_ROOT

ROOT=Path(__file__).resolve().parents[1]
CAL=CAL_ROOT/'robots/so_follower/official_medicine_follower.json'
PREVIEW=ROOT/'scripts/so101_medicine_live_preview.py'
MAX_STEP=3
MAX_STEPS=120 if '--task' in sys.argv else 5

def limits(name, c):
    # The wrist had an earlier measured overload event and changed calibration.
    # This central band is intentionally narrower than either calibration.
    if name=='wrist_flex': return (max(c['range_min']+100,1100),min(c['range_max']-100,2700))
    return c['range_min']+100,c['range_max']-100

def targets(cal, pos, prediction):
    result={}
    for i,name in enumerate(NAMES):
        c=cal[name];lo,hi=limits(name,c);v=float(prediction[i])
        if not (-0.5<=v<=100.5 if name=='gripper' else -100.5<=v<=100.5): raise ValueError(name+': model output outside calibrated range')
        if not lo<=pos[name]<=hi: raise ValueError(name+': current position outside conservative pilot band')
        target=c['range_min']+(v/100 if name=='gripper' else (v+100)/200)*(c['range_max']-c['range_min'])
        if not lo<=target<=hi: raise ValueError(name+': model target outside conservative pilot band')
        delta=target-pos[name]
        result[name]=round(pos[name]+max(-MAX_STEP,min(MAX_STEP,delta)))
    return result

def main():
    ui=tk.Tk();ui.title('SO-101 云端模型执行');ui.geometry('680x300')
    status=tk.StringVar(value='尚未启动。清空机械臂周围，确认随时能切断电源，再点击开始。')
    stop=threading.Event();running=False
    def show(s): ui.after(0,lambda:status.set(s))
    def run():
        nonlocal running
        armed=False
        first_observation=True
        try:
            for step in range(MAX_STEPS):
                if stop.is_set():break
                show(f'步骤 {step+1}/{MAX_STEPS}：读取画面并请求预测；停止按钮随时可用。')
                proc=subprocess.Popen([sys.executable,'-u',str(PREVIEW)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
                deadline=time.monotonic()+30
                while proc.poll() is None:
                    if stop.wait(.1): proc.terminate();proc.communicate(timeout=5);return
                    if time.monotonic()>deadline:
                        proc.terminate();proc.communicate(timeout=5);raise RuntimeError('观察或推理超过 30 秒')
                stdout,stderr=proc.communicate()
                if proc.returncode: raise RuntimeError('读取或预测失败：'+stderr[-700:])
                fileline=next((x for x in stdout.splitlines() if x.startswith('saved: ')),None)
                if fileline is None: raise RuntimeError('预测结果文件缺失')
                report=json.loads(Path(fileline[7:]).read_text(encoding='utf-8'))
                if report['elapsed_seconds']>5: raise RuntimeError('推理延迟超过 5 秒')
                if first_observation:
                    torques=[report['diagnostics'][n]['torque'] for n in NAMES]
                    if all(t == 1 for t in torques): armed=True
                    elif any(t != 0 for t in torques): raise RuntimeError('电机力矩状态不一致')
                    first_observation=False
                with_arm=Arm('COM6',CAL)
                try:
                    pos=with_arm.positions()
                    for name in NAMES:
                        if abs(pos[name]-report['diagnostics'][name]['position'])>8: raise RuntimeError(name+': 观测期间发生移动')
                        if with_arm.read(name,65,1): raise RuntimeError(name+': 电机故障')
                        if with_arm.read(name,40,1)!=(1 if armed else 0): raise RuntimeError(name+': 力矩状态不符合预期')
                    goal=targets(with_arm.cal,pos,[report['predicted_first'][n] for n in NAMES])
                    if stop.is_set(): break
                    if not armed:
                        fresh=with_arm.positions()
                        if any(abs(fresh[n]-pos[n])>3 for n in NAMES): raise RuntimeError('启动期间姿态不稳定')
                        for name in NAMES:
                            if stop.is_set(): return
                            with_arm.write(name,42,fresh[name])
                        for name in NAMES:
                            if with_arm.read(name,40,1)!=1: raise RuntimeError(name+': 未确认力矩开启')
                        armed=True
                    for name in NAMES:
                        if stop.is_set():return
                        with_arm.write(name,42,goal[name])
                    time.sleep(.3)
                    actual=with_arm.positions()
                    for name in NAMES:
                        if abs(actual[name]-goal[name])>20 or with_arm.read(name,65,1): raise RuntimeError(name+': 跟踪异常或故障')
                    show(f'完成 {step+1}/{MAX_STEPS} 次小幅闭环动作。')
                finally:with_arm.close()
                if stop.wait(.7):break
            show('执行已停止。电机可能仍保持力矩；托稳从臂后再断电。')
        except Exception as e:
            show('已停止：'+str(e)+'。电机可能仍保持力矩；托稳后处理。')
        finally:running=False
    def start():
        nonlocal running
        if running:return
        stop.clear();running=True;threading.Thread(target=run,daemon=True).start()
    def halt():
        stop.set();status.set('已请求停止；不再发送新动作。电机可能仍保持力矩。')
    def close():
        halt()
        if running: ui.after(250,close)
        else:ui.destroy()
    tk.Label(ui,text='云端模型控制 SO-101',font=('Microsoft YaHei',17)).pack(pady=10)
    tk.Label(ui,text=f'最多 {MAX_STEPS} 步，每关节每步最多 3 个编码器刻度。停止后仍可能保持力矩。',wraplength=640).pack(pady=8)
    tk.Label(ui,textvariable=status,wraplength=640).pack(pady=15)
    f=tk.Frame(ui);f.pack();tk.Button(f,text='开始执行',command=start,width=20,height=2).pack(side='left',padx=12);tk.Button(f,text='立即停止（Esc）',command=halt,width=20,height=2).pack(side='left',padx=12)
    ui.bind('<Escape>',lambda _:halt());ui.protocol('WM_DELETE_WINDOW',close);ui.mainloop()

if __name__=='__main__':main()
