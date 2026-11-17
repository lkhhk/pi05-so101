"""No-hardware checks for evaluation identity, starting poses and paired scoring."""
import json
from pathlib import Path
import tempfile
import threading
import time
import subprocess
import sys
import unittest
from unittest.mock import Mock

from blockbox_rtc_protocol import BASE_CHECKPOINT,CHECKPOINT,PROTOCOL,Remote
from evaluate_blockbox_ab import configure_protocol,pending,plan,report,selected_success
from blockbox_trial_control import TrialControl,request_success,write_json
from run_blockbox_rtc import validate_start_state


class ABTests(unittest.TestCase):
    def test_protocol_changes_only_future_duration(self):
        historical={'outcome':'failure','duration_s':60}
        data={'duration_s':60,'trials':[{'attempts':[historical]}]}
        configure_protocol(data)
        self.assertEqual(data['duration_s'],30)
        self.assertEqual(historical['duration_s'],60)
        configure_protocol(data)
        self.assertEqual(len(data['protocol_changes']),1)

    def test_live_success_stops_and_uses_click_time_not_cleanup_time(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stop=threading.Event();control=TrialControl(out,30,stop)
            with self.assertRaises(ValueError):request_success(out)
            clicked=time.monotonic_ns();start=clicked-1_250_000_000
            control.start(start)
            seconds=request_success(out,clicked)
            self.assertTrue(stop.wait(.5),'success button did not stop the execution event')
            time.sleep(.03)  # Cleanup latency must not change the measured success time.
            control.finish('success_selected')
            self.assertEqual(seconds,1.25)
            self.assertEqual(control.success['completion_s'],1.25)
            runtime=json.loads((out/'runtime.json').read_text())
            summary={'status':'success_selected','motor_commands_sent':20,
                     'motion_start_ns':start,**control.success}
            result=selected_success(summary,30)
            self.assertEqual(result['completion_s'],1.25)
            self.assertEqual(runtime['phase'],'stopped')
            with self.assertRaises(ValueError):request_success(out)

    def test_timeout_stale_commands_and_escape_cannot_score_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stop=threading.Event();control=TrialControl(out,30,stop)
            start=time.monotonic_ns()-31_000_000_000;control.start(start,watch=False)
            with self.assertRaises(ValueError):request_success(out)
            write_json(out/'control.json',{'outcome':'success','motion_start_ns':start,
                                         'clicked_ns':start+30_000_000_001})
            control.poll();self.assertIsNone(control.success)
            start=time.monotonic_ns()-1_000_000_000;control.start(start,watch=False)
            write_json(out/'control.json',{'outcome':'success','motion_start_ns':start-1,
                                         'clicked_ns':time.monotonic_ns()})
            control.poll();self.assertIsNone(control.success)
            request_success(out);stop.set();control.poll()
            self.assertIsNone(control.success)
            control.finish('operator_stopped')
            self.assertIsNone(selected_success({'status':'operator_stopped','motor_commands_sent':5},30))

    def test_separate_ui_process_signals_execution_clock(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);stop=threading.Event();control=TrialControl(out,30,stop)
            start=time.monotonic_ns()-500_000_000;control.start(start)
            try:
                source="import sys;sys.path.insert(0,sys.argv[1]);from blockbox_trial_control import request_success;request_success(sys.argv[2])"
                subprocess.run([sys.executable,'-c',source,str(Path(__file__).resolve().parent),temp],check=True)
                self.assertTrue(stop.wait(.5))
                observed=(time.monotonic_ns()-start)/1e9
                self.assertGreaterEqual(control.success['completion_s'],.5)
                self.assertLessEqual(control.success['completion_s'],observed)
                self.assertLess(observed-control.success['completion_s'],.5)
            finally:control.finish('success_selected')

    def test_wrong_model_cannot_pass_health(self):
        health={'protocol':PROTOCOL,'checkpoint':BASE_CHECKPOINT,'rtc_enabled':True,
                'fps':30,'action_shape':[50,6],'state_units':'body_degrees_gripper_percent'}
        response=Mock(status=200);response.read.return_value=json.dumps(health).encode()
        remote=Remote('http://127.0.0.1:8093');remote.conn=Mock()
        remote.conn.getresponse.return_value=response
        with self.assertRaisesRegex(RuntimeError,'Wrong RTC deployment'):remote.health()
        remote.expected_checkpoint=BASE_CHECKPOINT
        self.assertEqual(remote.health()['checkpoint'],BASE_CHECKPOINT)

    def test_pose_mismatch_and_nan_rejected_before_motion(self):
        ref={'state':[0]*6,'tolerance':[5]*6}
        validate_start_state([5]*6,ref)
        with self.assertRaisesRegex(RuntimeError,'no motion'):validate_start_state([0,56,0,0,0,0],ref)
        with self.assertRaises(ValueError):validate_start_state([float('nan')]*6,ref)

    def test_plan_balanced_and_unlabelled_attempt_blocks_progress(self):
        rows=plan();self.assertEqual(len(rows),40)
        self.assertEqual(sum(x['variant']=='base' for x in rows),20)
        rows[0]['attempts']=[{}];self.assertEqual(pending(rows)['id'],'T01')
        rows[0]['attempts'][0]['outcome']='infrastructure'
        self.assertEqual(pending(rows)['id'],'T01')

    def test_observed_mismatch_keeps_failures_but_excludes_paired_gain(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=plan()
            for row,state in zip(rows[:2],([0]*6,[0,56,36,0,0,0])):
                out=root/row['id'];out.mkdir()
                (out/'summary.json').write_text(json.dumps({'initial_state':state}))
                row['attempts']=[{'directory':str(out),'outcome':'failure'}]
            result=report({'trials':rows},root)
            self.assertEqual(result['base']['scored_trials'],1)
            self.assertEqual(result['lora']['scored_trials'],1)
            self.assertEqual(result['paired']['complete_pairs'],0)
            self.assertIsNone(result['paired']['success_gain_percentage_points'])
            (root/'T02/summary.json').write_text(json.dumps({'initial_state':[1]*6}))
            result=report({'trials':rows},root)
            self.assertEqual(result['paired']['complete_pairs'],1)
            self.assertEqual(result['paired']['both_failure'],1)


if __name__=='__main__':unittest.main()
