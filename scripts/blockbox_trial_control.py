"""Local success-button signal and a shared monotonic trial clock. No hardware I/O."""
import json
from pathlib import Path
import threading
import time


def write_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def request_success(directory,clicked_ns=None):
    # Capture before disk reads; GUI refresh and process shutdown never enter this time.
    clicked_ns=time.monotonic_ns() if clicked_ns is None else clicked_ns
    directory=Path(directory)
    live=json.loads((directory/'runtime.json').read_text(encoding='utf-8'))
    if live.get('phase')!='running':raise ValueError('本轮尚未开始动作或已经停止')
    elapsed=(clicked_ns-live['motion_start_ns'])/1e9
    if not 0<elapsed<=live['duration_s']:raise ValueError('本轮已超时或尚未开始，不能记录成功')
    command={'outcome':'success','motion_start_ns':live['motion_start_ns'],
             'clicked_ns':clicked_ns,'completion_s':elapsed}
    write_json(directory/'control.json',command)
    return elapsed


class TrialControl:
    def __init__(self,directory,duration,stop):
        self.directory=Path(directory);self.duration=duration;self.stop=stop
        self.start_ns=None;self.success=None;self.thread=None
        write_json(self.directory/'runtime.json',{'phase':'preparing','duration_s':duration})

    def start(self,start_ns=None,watch=True):
        self.start_ns=time.monotonic_ns() if start_ns is None else start_ns
        write_json(self.directory/'runtime.json',{'phase':'running','duration_s':self.duration,
                                                 'motion_start_ns':self.start_ns})
        if watch:
            self.thread=threading.Thread(target=self.watch,daemon=True,name='trial-success-button')
            self.thread.start()

    def poll(self):
        path=self.directory/'control.json'
        if self.start_ns is None or self.stop.is_set() or not path.exists():return
        command=json.loads(path.read_text(encoding='utf-8'))
        if command.get('outcome')!='success' or command.get('motion_start_ns')!=self.start_ns:return
        clicked=command.get('clicked_ns')
        if not isinstance(clicked,int):return
        elapsed=(clicked-self.start_ns)/1e9
        if not 0<elapsed<=self.duration or clicked>time.monotonic_ns():return
        self.success={'completion_s':elapsed,'success_clicked_ns':clicked,
                      'completion_time_source':'success_button_monotonic_clock'}
        self.stop.set()

    def watch(self):
        while not self.stop.is_set():
            try:self.poll()
            except (OSError,ValueError):pass
            self.stop.wait(.01)

    def finish(self,status):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=1)
        write_json(self.directory/'runtime.json',{'phase':'stopped','status':status,
                   'duration_s':self.duration,'motion_start_ns':self.start_ns,**(self.success or {})})
