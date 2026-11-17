"""Local operator panel for the paired evaluation; each motion needs a button press."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox,simpledialog

from evaluate_blockbox_ab import FORMAL,ROOT,TRIAL_DURATION,SUCCESS_DEFINITION,configure_protocol,pending,plan,reference_for,report,save
from blockbox_trial_control import request_success


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ready-after-reset',action='store_true')
    args=parser.parse_args()
    session=FORMAL;manifest=session/'manifest.json'
    session.mkdir(parents=True,exist_ok=True)
    if not manifest.exists():
        save(manifest,{'created':datetime.now().isoformat(),'duration_s':TRIAL_DURATION,
             'success_definition':SUCCESS_DEFINITION,
             'trials':plan()})
    data=json.loads(manifest.read_text(encoding='utf-8'));configure_protocol(data);save(manifest,data)
    window=tk.Tk();window.title('方块入盒：底座与 LoRA 配对测试');window.geometry('780x620')
    label=tk.Label(window,font=('Microsoft YaHei',15),justify='left');label.pack(pady=18)
    text=tk.Label(window,font=('Microsoft YaHei',11),justify='left',wraplength=740);text.pack(pady=8)
    stats=tk.Label(window,font=('Microsoft YaHei',12),justify='left');stats.pack(pady=8)
    tk.Label(window,text='每轮最多 30 秒。执行中点击“成功入盒并停止”，自动记录点击时的成功用时。\n预检和倒计时不计入用时；Esc 用于停止或接管，人工帮助计失败。\n机械臂自动复位，方块和盒子手动摆回；运动时请移开手。',
             font=('Microsoft YaHei',11),justify='left').pack(pady=8)
    success_feedback=tk.Label(window,font=('Microsoft YaHei',12),fg='green');success_feedback.pack()
    process=None;kind=None;opened_log=None;pose_started=0;ready_id=None
    success_requested=None
    cache_stamp=None;cache_data=None;cache_report=None
    def load():return json.loads(manifest.read_text(encoding='utf-8'))
    if args.ready_after_reset:
        row=pending(load()['trials'])
        if row:ready_id=row['id']
    def busy():return process is not None and process.poll() is None
    def prepare():
        nonlocal process,kind,pose_started,ready_id
        if busy():return
        data=load();row=pending(data['trials'])
        if row is None:return
        if row['attempts'] and 'outcome' not in row['attempts'][-1]:
            messagebox.showinfo('先标注结果','请先标注上一轮的实际结果。');return
        try:
            ref=reference_for(data,row,session);pose_started=time.time();ready_id=None
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/blockbox_ab_autoreset.py'),
                                      '--reference',str(ref),'--execute'],cwd=ROOT)
            kind='reset'
        except Exception as error:messagebox.showerror('复位未完成',str(error))
    def start():
        nonlocal process,kind,opened_log,ready_id,success_requested
        if busy():return
        data=load();row=pending(data['trials'])
        if row is None:return
        if row['attempts'] and 'outcome' not in row['attempts'][-1]:
            messagebox.showinfo('先标注结果','请先标注上一轮的实际结果。');return
        if ready_id!=row['id']:
            messagebox.showinfo('先自动复位','请先点击“自动复位机械臂”，复位完成后摆回方块与盒子，再启动本轮。');return
        if row['scene'] not in data.get('scenes',{}):
            description=simpledialog.askstring('记录场景',f"{row['scene']}：请先标记方块和盒子的位置，并简述摆放。\n同场景四轮保持这些位置，每轮重新摆回。",parent=window)
            if not description:return
            data.setdefault('scenes',{})[row['scene']]=description;save(manifest,data)
        opened_log=open(session/(row['id']+'-console-'+datetime.now().strftime('%H%M%S')+'.log'),'w',encoding='utf-8')
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluate_blockbox_ab.py'),
                                  '--session',str(session),'--run-next','--execute'],cwd=ROOT,
                                 stdout=opened_log,stderr=subprocess.STDOUT)
        kind='trial';ready_id=None
        success_requested=None;success_feedback.config(text='')
    def mark(outcome):
        nonlocal ready_id,success_requested
        clicked_ns=time.monotonic_ns()
        if outcome=='success':
            if not busy() or kind!='trial':
                messagebox.showinfo('执行中选择成功','请在本轮执行过程中点击“成功入盒并停止”，用时将按点击时刻自动记录。');return
            if success_requested is not None:return
            try:
                row=pending(load()['trials'])
                directory=Path(row['attempts'][-1]['directory'])
                seconds=request_success(directory,clicked_ns)
                success_requested=(str(directory),seconds)
                success_button.config(state='disabled')
                success_feedback.config(text=f'已点击成功：{seconds:.3f} 秒；正在停止并保持当前位置。')
            except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
                messagebox.showinfo('尚不能记录成功',str(error))
            return
        if busy():return
        data=load();row=pending(data['trials'])
        if row is None or not row['attempts'] or 'outcome' in row['attempts'][-1]:return
        path=Path(row['attempts'][-1]['directory'])/'summary.json'
        summary=json.loads(path.read_text()) if path.exists() else {}
        reason=''
        if outcome=='infrastructure':
            reason=simpledialog.askstring('记录故障原因','仅记录设备、通信或启动条件问题。\n模型错误动作或人工接管应记失败。',
                                          initialvalue=summary.get('error',''),parent=window)
            if not reason:return
        elif outcome=='failure':
            reason=simpledialog.askstring('失败现象','请简述：未抓起、抓起后掉落、未放入盒、人工接管等。',parent=window)
            if reason is None:return
        grasped=messagebox.askyesnocancel('抓取记录','本轮是否曾将方块成功抓起离开桌面？\n不确定可选择取消，仅不记录抓取指标。',parent=window)
        command=[sys.executable,str(ROOT/'scripts/evaluate_blockbox_ab.py'),'--session',str(session),
                 '--label',outcome,'--reason',reason]
        if grasped is not None:command+=['--grasped','yes' if grasped else 'no']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode:messagebox.showerror('结果未保存',result.stderr or result.stdout)
        else:ready_id=None;print('OBSERVER_LABEL',row['id'],outcome,flush=True)
    tk.Button(window,text='① 自动复位机械臂（先移开手）',font=('Microsoft YaHei',12),command=prepare).pack(pady=5)
    tk.Button(window,text='② 已摆好方块和盒子，启动本轮',font=('Microsoft YaHei',12),command=start).pack(pady=5)
    buttons=tk.Frame(window);buttons.pack(pady=15)
    success_button=tk.Button(buttons,text='成功入盒并停止（执行中点击）',font=('Microsoft YaHei',11),bg='#d5f0d8',
                             state='disabled',command=lambda:mark('success'))
    success_button.pack(pady=5)
    for title,outcome in [('未完成 / 人工接管','failure'),('设备或通信故障','infrastructure')]:
        tk.Button(buttons,text=title,font=('Microsoft YaHei',10),command=lambda x=outcome:mark(x)).pack(side='left',padx=5)
    def close():
        if busy():
            messagebox.showinfo('仍在运行','请先按 Esc 停止机械臂，等待本轮退出后再关闭面板。');return
        print('PANEL_CLOSED',flush=True);window.destroy()
    def refresh():
        nonlocal process,opened_log,ready_id,cache_stamp,cache_data,cache_report
        try:
            if process is not None and process.poll() is not None:
                if opened_log is not None:opened_log.close();opened_log=None
                if kind=='reset' and process.returncode==0:
                    row=pending(load()['trials'])
                    if row:ready_id=row['id']
                if kind=='trial' and success_requested is not None:
                    recorded=[a for row in load()['trials'] for a in row['attempts']
                              if a['directory']==success_requested[0] and a.get('outcome')=='success']
                    if recorded:success_feedback.config(text=f"已记录成功用时：{recorded[0]['completion_s']:.3f} 秒。",fg='green')
                    else:success_feedback.config(text='停止已请求，但成功结果未确认，请检查本轮日志后标注。',fg='#a65b00')
                print('PANEL_STEP_FINISHED',kind,process.returncode,flush=True);process=None
            stamp=manifest.stat().st_mtime_ns
            if stamp!=cache_stamp:
                cache_data=load();cache_report=report(cache_data,session);cache_stamp=stamp
            data=cache_data;r=cache_report;row=pending(data['trials'])
            live=None
            if busy() and kind=='trial' and row and row['attempts']:
                runtime=Path(row['attempts'][-1]['directory'])/'runtime.json'
                if runtime.exists():live=json.loads(runtime.read_text(encoding='utf-8'))
            elapsed=None if not live or live.get('phase')!='running' else (time.monotonic_ns()-live['motion_start_ns'])/1e9
            can_select=elapsed is not None and 0<elapsed<=TRIAL_DURATION and success_requested is None
            success_button.config(state='normal' if can_select else 'disabled')
            a=r['base'];b=r['lora']
            historical='累计（含历史60秒记录）\n' if '60' in r.get('by_duration_s',{}) else ''
            stats.config(text=historical+f"底座：{a['successes']}/{a['scored_trials']} 成功    LoRA：{b['successes']}/{b['scored_trials']} 成功\n有效配对：{r['paired']['complete_pairs']}/20    通信/设备中断：{a['infrastructure_attempts']+b['infrastructure_attempts']}")
            if row is None:
                label.config(text='40 轮已评分，测试完成');text.config(text='结果已保存。关闭窗口后，Codex 将汇总结果。')
            else:
                label.config(text=f"本轮 {row['id']} / 40    场景 {row['scene']}    第 {row['repeat']} 次\n模型：{'未做本次 LoRA 微调的底座' if row['variant']=='base' else 'LoRA 微调版本'}")
                if busy():
                    if kind=='reset':message='机械臂自动复位中，请移开手；Esc 停止并保持。'
                    elif elapsed is not None:message=f'执行用时 {min(elapsed,TRIAL_DURATION):.2f} / 30.00 秒。成功入盒后点击绿色按钮，立即停止并记录用时。'
                    else:message='预检 / 倒计时 / 停止处理中。动作开始后启用绿色成功按钮；预检时间不计入用时。'
                elif row['attempts'] and 'outcome' not in row['attempts'][-1]:
                    s=Path(row['attempts'][-1]['directory'])/'summary.json'
                    summary=json.loads(s.read_text()) if s.exists() else {}
                    message='本轮已停止，请标注实际结果。'+('\n中断原因：'+summary['error'] if summary.get('error') else '')
                elif ready_id==row['id']:message='已锁定共同起点。摆好方块和盒子，移开手后点击启动。'
                else:message='点击自动复位，等待机械臂回到共同起点，再摆回方块与盒子。新的场景请使用示范范围内的标记位置。'
                text.config(text=message)
        except Exception as error:text.config(text=str(error))
        window.after(100,refresh)
    window.protocol('WM_DELETE_WINDOW',close);refresh();window.mainloop()


if __name__=='__main__':main()
