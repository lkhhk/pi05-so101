"""Replay the reported preflight failure without networking or motor access."""
import unittest
import numpy as np
from blockbox_rtc_admission import preflight


class AdmissionTests(unittest.TestCase):
    def sampler(self,latencies):
        calls=[]
        def sample(prefix):
            elapsed=latencies[min(len(calls),len(latencies)-1)]
            calls.append(elapsed)
            return {'normalized_actions':np.zeros((50,6),np.float32)},elapsed
        return sample,calls

    def test_reported_transient_recovers_after_new_good_sample(self):
        sample,calls=self.sampler([.406,.688,.390,.328,.391,.328,.360])
        recent=preflight(sample,lambda *_:None)
        self.assertLessEqual(max(recent),.391)
        self.assertEqual(len(calls),7)

    def test_sustained_slow_route_remains_blocked(self):
        sample,calls=self.sampler([.406,.8])
        with self.assertRaisesRegex(RuntimeError,'RTC latency admission failed'):
            preflight(sample,lambda *_:None)
        self.assertEqual(len(calls),16)

    def test_recovered_window_excludes_rejected_startup_latency(self):
        sample,calls=self.sampler([.4,1.2,.35,.35,.35,.35,.35])
        recent=preflight(sample,lambda *_:None)
        self.assertEqual(recent,[.35]*5)
        self.assertEqual(len(calls),7)

    def test_protocol_error_does_not_retry(self):
        def failed(_):raise RuntimeError('Wrong RTC deployment')
        with self.assertRaisesRegex(RuntimeError,'Wrong RTC deployment'):
            preflight(failed,lambda *_:None)

    def test_good_route_keeps_six_request_startup(self):
        sample,calls=self.sampler([1.1,.375,.328,.312,.312,.375])
        recent=preflight(sample,lambda *_:None)
        self.assertEqual(len(calls),6)
        self.assertEqual(len(recent),5)


if __name__=='__main__':unittest.main()
