"""Video hooks and rendering; encoder I/O is exercised by the CLI smoke test."""
import unittest
from unittest.mock import Mock, patch
import numpy as np
import continuation
from curved_linesearch import visualization
import test_continuation


class RecordingVideo:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append(self, uv, norm, label):
        self.frames.append((np.asarray(uv).reshape(-1, 2).copy(), float(norm), label))

    def close(self):
        self.closed = True


class ContinuationVideoChecks(unittest.TestCase):
    make_problem = test_continuation.ContinuationChecks.make_problem

    def test_recording_preserves_solution_and_callback_and_uses_true_norms(self):
        p0, _, opt0, uv = self.make_problem()
        reference = continuation.run_continuation(p0, opt0, uv, gtol=.4, final_gtol=1e-5,
                                                 final_niter=50, verbose=False)
        param, prob, opt, uv = self.make_problem()
        callback_calls = []
        prob.setCustomIterationCallback(lambda problem, i: callback_calls.append(i) or False)
        recorder = RecordingVideo()
        with patch.object(visualization, 'ContinuationVideoWriter', return_value=recorder):
            result = continuation.run_continuation(param, opt, uv, gtol=.4, final_gtol=1e-5,
                                                  final_niter=50, verbose=False, video_path='test.mp4')
        self.assertTrue(recorder.closed)
        np.testing.assert_allclose(result.uv, reference.uv, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(result.steps), len(reference.steps))
        self.assertEqual(recorder.frames[0][2], 'Initial state')
        self.assertEqual(sum(label.startswith('Continuation ') for _, _, label in recorder.frames), len(result.steps))
        self.assertGreater(sum(label.startswith('Final optimization:') for _, _, label in recorder.frames), 1)
        np.testing.assert_array_equal(recorder.frames[-1][0], result.uv)
        true_prob = continuation.original_rest_problem(param, prob)
        for frame_uv, norm, _ in recorder.frames:
            true_prob.setVars(frame_uv.ravel())
            self.assertAlmostEqual(norm, np.linalg.norm(true_prob.gradient()), places=10)
        previous_count = len(callback_calls)
        previous_frames = len(recorder.frames)
        self.assertFalse(prob.getCustomIterationCallback()(prob, 123))
        self.assertEqual(len(callback_calls), previous_count + 1)
        self.assertEqual(callback_calls[-1], 123)
        self.assertEqual(len(recorder.frames), previous_frames)

    def test_writer_and_callback_restored_after_optimizer_failure(self):
        param, prob, opt, uv = self.make_problem()
        calls = []
        def callback(problem, iteration):
            calls.append(iteration)
            raise RuntimeError('callback failure')
        prob.setCustomIterationCallback(callback)
        recorder = RecordingVideo()
        with patch.object(visualization, 'ContinuationVideoWriter', return_value=recorder):
            with self.assertRaisesRegex(RuntimeError, 'callback failure'):
                # Skip continuation so the failure occurs in the wrapped final callback.
                continuation.run_continuation(param, opt, uv, gtol=1e10, final_gtol=1e-5,
                                              final_niter=50, verbose=False, video_path='test.mp4')
        self.assertTrue(recorder.closed)
        with self.assertRaisesRegex(RuntimeError, 'callback failure'):
            prob.getCustomIterationCallback()(prob, 123)
        self.assertEqual(calls[-1], 123)
        np.testing.assert_array_equal(prob.getVars(), uv.ravel())

    def test_absent_callback_remains_absent(self):
        param, prob, opt, uv = self.make_problem()
        self.assertIsNone(prob.getCustomIterationCallback())
        recorder = RecordingVideo()
        with patch.object(visualization, 'ContinuationVideoWriter', return_value=recorder):
            continuation.run_continuation(param, opt, uv, gtol=1e10, final_niter=50,
                                          final_gtol=1e-5, verbose=False, video_path='test.mp4')
        self.assertIsNone(prob.getCustomIterationCallback())
        self.assertTrue(recorder.closed)

    def test_renderer_has_requested_panels_and_skips_duplicate_frames(self):
        param, _, _, uv = self.make_problem()
        encoder = Mock(ffmpegProc=None)
        encoder.writeFrame.side_effect = lambda fig: fig.canvas.draw()
        with patch.object(visualization.video_writer, 'PlotVideoWriter', return_value=encoder), \
             patch('shutil.which', return_value='/fake/ffmpeg'):
            movie = visualization.ContinuationVideoWriter('test.mp4', param.mesh, dpi=60)
            try:
                movie.append(uv, 10., 'Initial state')
                movie.append(uv, 10., 'Duplicate')
                movie.append(uv * 1.1, 1., 'Continuation 1')
                self.assertEqual(encoder.writeFrame.call_count, 2)
                self.assertEqual(movie.gradient_norms, [10., 1.])
                self.assertEqual(len(movie.figure.axes), 2)
                self.assertEqual(len(movie.ax_mesh.collections), 2)
                self.assertEqual(movie.ax_norm.get_yscale(), 'log')
                np.testing.assert_array_equal(movie.ax_norm.lines[0].get_ydata(), [10., 1.])
            finally:
                movie.close()
            movie.close()
            encoder.finish.assert_called_once()


if __name__ == '__main__':
    unittest.main()
