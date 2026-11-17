"""Resumable paired real-robot evaluation. Outcomes are supplied by the observer."""
import argparse
import http.client
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from urllib.parse import urlsplit

from blockbox_rtc_protocol import BASE_CHECKPOINT,CHECKPOINT

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/blockbox-ab-20261117'
FORMAL=ROOT/'results/blockbox-ab-formal-20261117'
VARIANTS={'base':('http://127.0.0.1:8094',BASE_CHECKPOINT),
          'lora':('http://127.0.0.1:8092',CHECKPOINT)}
TRIAL_DURATION=30
SUCCESS_DEFINITION='Within 30 seconds, without assistance, place the whole block inside the box and release the gripper; select success during execution to stop and timestamp completion.'


def save(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def configure_protocol(data):
    changes=data.get('protocol_changes',[])
    original=changes[0].get('previous_duration_s') if changes else data.get('duration_s',TRIAL_DURATION)
    for row in data.get('trials',[]):
        for attempt in row['attempts']:
            if 'duration_s' in attempt:continue
            duration=original or TRIAL_DURATION
            started=attempt.get('started_at')
            if started:
                when=datetime.strptime(started,'%Y%m%d-%H%M%S-%f')
                for change in changes:
                    if when>=datetime.fromisoformat(change['at']):duration=change['duration_s']
            attempt['duration_s']=duration
    if data.get('duration_s')!=TRIAL_DURATION:
        data.setdefault('protocol_changes',[]).append({'at':datetime.now().isoformat(),
            'previous_duration_s':data.get('duration_s'),'duration_s':TRIAL_DURATION,
            'success_timing':'operator success-button click during motion'})
    data['duration_s']=TRIAL_DURATION;data['success_definition']=SUCCESS_DEFINITION


def selected_success(summary,duration):
    seconds=summary.get('completion_s')
    if summary.get('status')!='success_selected' or not summary.get('motor_commands_sent'):return None
    if not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or not 0<seconds<=duration:
        raise ValueError('Invalid success-button completion time')
    return {'outcome':'success','reason':'Operator selected success during execution',
            'completion_s':seconds,'completion_time_source':summary['completion_time_source'],
            'success_clicked_ns':summary['success_clicked_ns'],'motion_start_ns':summary['motion_start_ns'],
            'grasped':None,'labelled_at':datetime.now().isoformat()}


def plan():
    rows=[]
    for scene in range(1,11):
        for repeat in range(1,3):
            order=('base','lora') if (scene+repeat)%2==0 else ('lora','base')
            for variant in order:
                rows.append({'id':f'T{len(rows)+1:02d}','scene':f'S{scene:02d}',
                             'repeat':repeat,'variant':variant,'attempts':[]})
    return rows


def health_check():
    health={}
    for variant,(url,checkpoint) in VARIANTS.items():
        parsed=urlsplit(url)
        connection=http.client.HTTPConnection(parsed.hostname,parsed.port,timeout=8)
        try:
            connection.request('GET','/health')
            response=connection.getresponse()
            if response.status!=200:raise RuntimeError(f'{variant} health HTTP {response.status}')
            health[variant]=json.loads(response.read())
        finally:connection.close()
        d=health[variant]
        if not d.get('ready') or d.get('checkpoint')!=checkpoint or d.get('rtc_mode')!='guided':
            raise RuntimeError(f'Wrong {variant} deployment')
    for key in ('protocol','fps','action_shape','state_units','action_mean','action_std'):
        if health['base'][key]!=health['lora'][key]:raise RuntimeError('A/B mismatch: '+key)
    return health


def pending(rows):
    for row in rows:
        if not row['attempts'] or row['attempts'][-1].get('outcome')=='infrastructure':return row
        if row['attempts'][-1].get('outcome') not in ('success','failure'):return row
    return None


def reference_for(data,row,directory):
    source=directory/'reference.json'
    if not source.exists():return source
    ref=json.loads(source.read_text(encoding='utf-8'))
    for other in data['trials']:
        if (other['scene'],other['repeat'])!=(row['scene'],row['repeat']) or other['variant']==row['variant']:
            continue
        if not other['attempts'] or other['attempts'][-1].get('outcome') not in ('success','failure'):continue
        path=Path(other['attempts'][-1]['directory'])/'summary.json'
        previous=json.loads(path.read_text(encoding='utf-8'))
        ref={**ref,'state':previous['initial_state'],'paired_with':other['id']}
    target=directory/'next-reference.json';save(target,ref)
    return target


def wilson(k,n):
    if not n:return None
    z=1.96;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [c-h,c+h]


def report(data,directory):
    results={}
    for variant in VARIANTS:
        attempts=[a for row in data['trials'] if row['variant']==variant for a in row['attempts']]
        valid=[a for a in attempts if a.get('outcome') in ('success','failure')]
        successes=[a for a in valid if a['outcome']=='success']
        times=[a['completion_s'] for a in successes if a.get('completion_s') is not None]
        grasps=[a['grasped'] for a in valid if a.get('grasped') is not None]
        results[variant]={'successes':len(successes),'scored_trials':len(valid),
            'success_rate':len(successes)/len(valid) if valid else None,
            'wilson95_descriptive':wilson(len(successes),len(valid)),
            'successful_completion_median_s':statistics.median(times) if times else None,
            'successful_trials_with_completion_time':len(times),
            'grasped_count':sum(grasps),'grasp_labelled_trials':len(grasps),
            'infrastructure_attempts':sum(a.get('outcome')=='infrastructure' for a in attempts),
            'unlabelled_attempts':sum('outcome' not in a for a in attempts)}
    pairs={}
    for row in data['trials']:
        if row['attempts'] and row['attempts'][-1].get('outcome') in ('success','failure'):
            pairs.setdefault((row['scene'],row['repeat']),{})[row['variant']]=row['attempts'][-1]['outcome']=='success'
    initial={}
    for row in data['trials']:
        if row['attempts']:
            path=Path(row['attempts'][-1]['directory'])/'summary.json' if row['attempts'][-1].get('directory') else None
            if path and path.exists():
                initial[(row['scene'],row['repeat'],row['variant'])]=json.loads(path.read_text()).get('initial_state')
    complete=[];rejected=[]
    for (scene,repeat),pair in pairs.items():
        if len(pair)!=2:continue
        a=initial.get((scene,repeat,'base'));b=initial.get((scene,repeat,'lora'))
        if a is None or b is None:
            rejected.append({'scene':scene,'repeat':repeat,'reason':'missing_initial_state'});continue
        delta=[abs(x-y) for x,y in zip(a,b)]
        if len(delta)!=6 or any(x>5 for x in delta):
            rejected.append({'scene':scene,'repeat':repeat,'reason':'starting_pose_mismatch','delta':delta});continue
        complete.append(pair)
    wins=sum(x['lora'] and not x['base'] for x in complete)
    losses=sum(x['base'] and not x['lora'] for x in complete)
    results['paired']={'complete_pairs':len(complete),'excluded_pairs':rejected,'lora_only_success':wins,'base_only_success':losses,
        'both_success':sum(all(x.values()) for x in complete),
        'both_failure':sum(not any(x.values()) for x in complete),
        'success_gain_percentage_points':100*(wins-losses)/len(complete) if complete else None}
    results['note']='Pilot study; repeated trials within a scene are correlated. Wilson intervals describe individual rates, not confidence in the paired gain. All infrastructure exclusions remain in the manifest.'
    results['by_duration_s']={}
    for duration in sorted({a['duration_s'] for row in data['trials'] for a in row['attempts'] if 'duration_s' in a}):
        group={}
        for variant in VARIANTS:
            scored=[a for row in data['trials'] if row['variant']==variant for a in row['attempts']
                    if a.get('duration_s')==duration and a.get('outcome') in ('success','failure')]
            group[variant]={'successes':sum(a['outcome']=='success' for a in scored),'scored_trials':len(scored)}
        results['by_duration_s'][str(duration)]=group
    save(directory/'report.json',results)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,default=FORMAL)
    action=p.add_mutually_exclusive_group()
    action.add_argument('--run-next',action='store_true')
    action.add_argument('--label',choices=('success','failure','infrastructure'))
    action.add_argument('--report',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--reason',default='')
    p.add_argument('--grasped',choices=('yes','no'))
    p.add_argument('--completion-s',type=float)
    args=p.parse_args()
    directory=args.session.resolve();directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'manifest.json'
    if not manifest.exists():
        save(manifest,{'created':datetime.now().isoformat(),'duration_s':TRIAL_DURATION,
                      'success_definition':SUCCESS_DEFINITION,
                      'trials':plan()})
    data=json.loads(manifest.read_text(encoding='utf-8'))
    configure_protocol(data);save(manifest,data)
    row=pending(data['trials'])
    if args.label:
        if row is None or not row['attempts'] or 'outcome' in row['attempts'][-1]:raise ValueError('No unlabelled attempt')
        attempt=row['attempts'][-1]
        summary_path=Path(attempt['directory'])/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if args.label=='success' and (not summary.get('motor_commands_sent') or summary.get('status')=='error'):
            raise ValueError('An interrupted/error/no-motion attempt cannot be labelled successful')
        if args.label=='infrastructure' and not args.reason.strip():raise ValueError('Infrastructure exclusion requires a reason')
        limit=summary.get('duration_s',attempt.get('duration_s',data['duration_s']))
        if args.completion_s is not None and (args.label!='success' or not 0<args.completion_s<=limit):
            raise ValueError(f'Completion time must be in (0,{limit}] and only for success')
        attempt.update(outcome=args.label,reason=args.reason,
                       grasped=None if args.grasped is None else args.grasped=='yes',
                       completion_s=args.completion_s,labelled_at=datetime.now().isoformat())
        save(manifest,data);row=pending(data['trials'])
    if args.run_next:
        if not args.execute:raise ValueError('Real-robot execution requires --execute')
        if row is None:raise ValueError('All 40 trials already scored')
        if row['attempts'] and 'outcome' not in row['attempts'][-1]:raise ValueError('Label the previous attempt before continuing')
        reference=reference_for(data,row,directory)
        if not reference.exists():raise ValueError('Capture and hold a common starting pose with blockbox_ab_pose.py first')
        health=health_check()
        timestamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        out=directory/f"{row['id']}-{row['variant']}-attempt{len(row['attempts'])+1}-{timestamp}"
        attempt={'directory':str(out),'started_at':timestamp,'health':health,'duration_s':data['duration_s']}
        row['attempts'].append(attempt);save(manifest,data)
        url,checkpoint=VARIANTS[row['variant']]
        command=[sys.executable,'-u',str(ROOT/'scripts/run_blockbox_rtc.py'),'--execute',
                 '--duration',str(data['duration_s']),'--url',url,'--expected-checkpoint',checkpoint,'--output-dir',str(out)]
        command+=['--reference-state',str(reference)]
        print('RUNNING',row['id'],row['scene'],'repeat',row['repeat'],row['variant'],flush=True)
        status=subprocess.run(command,cwd=ROOT).returncode
        attempt['process_exit_code']=status
        summary_path=out/'summary.json'
        automatic=None
        if status==0 and summary_path.exists():
            automatic=selected_success(json.loads(summary_path.read_text()),data['duration_s'])
            if automatic:attempt.update(automatic)
        save(manifest,data)
        print('SUCCESS_RECORDED' if automatic else 'AWAITING_OBSERVER_LABEL',str(out),flush=True)
    print(json.dumps(report(data,directory),ensure_ascii=False,indent=2))
    row=pending(data['trials'])
    if row:
        print('NEXT',json.dumps({k:v for k,v in row.items() if k!='attempts'},ensure_ascii=False),
              'NEEDS_LABEL' if row['attempts'] and 'outcome' not in row['attempts'][-1] else 'RESET_SCENE_BEFORE_RUN')
        if (directory/'reference.json').exists():print('ALIGNMENT_REFERENCE',reference_for(data,row,directory))
    else:print('ALL_40_TRIALS_SCORED')


if __name__=='__main__':main()
