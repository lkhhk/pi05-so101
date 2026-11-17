"""No-hardware regression tests for the RTC continuation boundary."""
import io
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from blockbox_rtc_protocol import encode_request,decode_image
from blockbox_rtc_queue import Continuation


class RTCContinuationTests(unittest.TestCase):
    def setUp(self):
        self.health={'action_mean':[1,2,3,4,5,6],'action_std':[2]*6}
        self.q=Continuation(self.health,[-100]*6,[100]*6)
        normal=np.repeat(np.arange(50,dtype=np.float32)[:,None],6,axis=1)/10
        self.result={'normalized_actions':normal,'actions':normal*2+np.arange(1,7)}

    def bootstrap(self):
        self.q.accept(self.result,self.q.queue.snapshot(),.2)

    def test_actual_consumed_steps_trim_both_spaces(self):
        self.bootstrap()
        snap=self.q.queue.snapshot()
        for _ in range(7):self.q.queue.get()
        self.q.accept(self.result,snap,.25)
        self.assertEqual(self.q.queue.qsize(),43)
        np.testing.assert_allclose(self.q.queue.get().numpy(),self.result['actions'][7])
        np.testing.assert_allclose(self.q.queue.get_left_over()[0].numpy(),self.result['normalized_actions'][8])

    def test_cancelled_response_cannot_restart_queue(self):
        self.bootstrap();snap=self.q.queue.snapshot();self.q.queue.clear()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):self.q.accept(self.result,snap,.2)
        self.assertTrue(self.q.queue.empty())

    def test_wall_clock_expiry_rejects_even_if_playback_stalled(self):
        self.bootstrap()
        for _ in range(30):self.q.queue.get()
        snap=self.q.queue.snapshot()
        with self.assertRaisesRegex(RuntimeError,'overlap'):self.q.accept(self.result,snap,.7)

    def test_exhausted_prefix_rejects_response(self):
        self.bootstrap()
        for _ in range(40):self.q.queue.get()
        snap=self.q.queue.snapshot()
        for _ in range(10):self.q.queue.get()
        with self.assertRaisesRegex(RuntimeError,'overlap'):self.q.accept(self.result,snap,.2)

    def test_normalization_mismatch_and_nan_rejected(self):
        broken={**self.result,'actions':self.result['actions']+1}
        with self.assertRaisesRegex(RuntimeError,'disagree'):self.q.accept(broken,self.q.queue.snapshot(),.2)
        broken={**self.result,'actions':np.full((50,6),np.nan)}
        with self.assertRaisesRegex(RuntimeError,'Invalid'):self.q.accept(broken,self.q.queue.snapshot(),.2)

    def test_clipped_prefix_matches_actual_goal_units(self):
        normal=np.full((50,6),100,np.float32)
        result={'normalized_actions':normal,'actions':normal*2+np.arange(1,7)}
        self.q.accept(result,self.q.queue.snapshot(),.2)
        motor=self.q.queue.get().numpy();prefix=self.q.queue.get_left_over()[0].numpy()
        np.testing.assert_allclose(motor,100)
        np.testing.assert_allclose(prefix*self.q.std+self.q.mean,motor)

    def test_insufficient_overlap_rejected_before_request(self):
        self.bootstrap()
        for _ in range(40):self.q.queue.get()
        with self.assertRaisesRegex(RuntimeError,'Insufficient'):self.q.request_parameters(.3)

    def test_transport_preserves_color_resize_and_model_padding(self):
        bgr=np.zeros((480,640,3),np.uint8)
        bgr[:,:,0]=17;bgr[:,:,1]=99;bgr[:,:,2]=233
        bgr[::5,:,0]=155
        _,payload=encode_request(np.zeros(6),[bgr,bgr],None,0,0)
        with np.load(io.BytesIO(payload),allow_pickle=False) as data:
            decoded=decode_image(data['front_png'])
        original=torch.from_numpy(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
        expected=F.interpolate(original[None],size=(168,224),mode='bilinear',align_corners=False)[0]
        expected=F.pad(expected,(0,0,28,28),value=-1)
        self.assertLessEqual(float((decoded-expected).abs().max()),.5/255+1e-6)
        self.assertTrue(torch.all(decoded[:,:28]==-1))
        self.assertTrue(torch.all(decoded[:,-28:]==-1))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
