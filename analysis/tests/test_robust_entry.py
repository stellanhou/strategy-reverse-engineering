"""Fit isolation, causal inputs and opening-alert accounting checks."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_joint_entry import decode
from robust_entry import M, FitClipper, calibrate_threshold, decode_refractory, timing_inputs


class RobustEntryChecks(unittest.TestCase):
    def test_timing_windows_and_symbol_boundaries(self):
        close = np.array([np.arange(1, 21), np.arange(101, 121)], dtype=float)
        x, names = timing_inputs(close.ravel(), 2, 20)
        x = x.reshape(2, 20, 3)
        self.assertEqual(names, ['ret1', 'ret3', 'pullback15'])
        for column, width in enumerate((2, 4, 15)):
            self.assertTrue(np.isnan(x[:, :width - 1, column]).all())
        self.assertAlmostEqual(x[0, 3, 0], 4 / 3 - 1)
        self.assertAlmostEqual(x[0, 3, 1], 4 / 1 - 1)
        self.assertAlmostEqual(x[1, 3, 1], 104 / 101 - 1)
        np.testing.assert_array_equal(x[:, 14:, 2], 0.)
        declining, _ = timing_inputs(np.arange(20, 0, -1), 1, 20)
        self.assertAlmostEqual(declining[19, 2], 1 / 15 - 1)

    def test_timing_requires_contiguous_positive_finite_windows(self):
        for invalid in (np.nan, np.inf, 0., -1.):
            close = np.arange(1, 26, dtype=float)
            close[5] = invalid
            x, _ = timing_inputs(close, 1, 25)
            self.assertTrue(np.isnan(x[5:7, 0]).all())
            self.assertTrue(np.isnan(x[5:9, 1]).all())
            self.assertTrue(np.isnan(x[14:20, 2]).all())
            self.assertTrue(np.isfinite(x[7, 0]))
            self.assertTrue(np.isfinite(x[9, 1]))
            self.assertTrue(np.isfinite(x[20, 2]))

    def test_timing_prefix_and_future_mutation(self):
        close = np.arange(1, 61, dtype=float).reshape(2, 30)
        x, _ = timing_inputs(close, 2, 30)
        prefix, _ = timing_inputs(close[:, :18], 2, 18)
        np.testing.assert_array_equal(x.reshape(2, 30, 3)[:, :18], prefix.reshape(2, 18, 3))
        close[:, 18:] = -999
        changed, _ = timing_inputs(close, 2, 30)
        np.testing.assert_array_equal(x.reshape(2, 30, 3)[:, :18], changed.reshape(2, 30, 3)[:, :18])

    def test_fit_bounds_unchanged_by_future_outlier_and_nan_preserved(self):
        train = np.arange(20, dtype=float).reshape(10, 2)
        clipper = FitClipper((.1, .9)).fit(train)
        lower, upper = np.quantile(train, [.1, .9], axis=0)
        transformed = clipper.transform([[1e8, -1e8], [np.nan, 5.]])
        np.testing.assert_array_equal(transformed[0], [upper[0], lower[1]])
        self.assertTrue(np.isnan(transformed[1, 0]))
        self.assertEqual(transformed[1, 1], 5.)
        np.testing.assert_array_equal(clipper.lower_, lower)
        np.testing.assert_array_equal(clipper.upper_, upper)
        np.testing.assert_array_equal(train, np.arange(20).reshape(10, 2))

    def test_fit_rejects_empty_and_nonfinite_rows(self):
        for bad in (np.empty((0, 2)), [[1., np.nan]], [[1., np.inf]]):
            with self.assertRaises(ValueError):
                FitClipper().fit(bad)

    def test_cooldown_selects_second_coin_and_reopens_at_boundary(self):
        times = np.arange(5) * M
        scores = np.array([[.9] * 5, [.8] * 5])
        alerts, pending = decode_refractory(times, scores, ['A', 'B'], .5, 10, 0, 5 * M, 0, 3)
        self.assertEqual([(a['symbol'], a['minute_index']) for a in alerts],
                         [('A', 0), ('B', 1), ('A', 3), ('B', 4)])
        self.assertEqual(pending, [])

    def test_cooldown_crosses_utc_day_and_cap_resets(self):
        times = np.array([1439, 1440, 1441, 1442]) * M
        scores = np.array([[.9] * 4, [.8] * 4])
        alerts, _ = decode_refractory(times, scores, ['A', 'B'], .5, 1,
                                      1439 * M, 1443 * M, 0, 3)
        self.assertEqual([(a['symbol'], a['issue_ms']) for a in alerts], [('A', 1439 * M), ('B', 1440 * M)])

    def test_pending_consumes_budget_and_window_is_half_open(self):
        times = np.arange(5) * M
        alerts, pending = decode_refractory(times, np.ones((1, 5)), ['A'], .5, 1,
                                            M, 3 * M, 2, 0)
        self.assertEqual(alerts, [])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['issue_ms'], M)
        self.assertEqual(pending[0]['time_ms'], 3 * M)

    def test_future_scores_cannot_change_prior_issues(self):
        times = np.arange(60) * M
        scores = np.full((2, 60), .6)
        before = decode_refractory(times, scores, ['A', 'B'], .5, 10, 0, 60 * M, 1, 7)
        scores[:, 25:] = 1.
        after = decode_refractory(times, scores, ['A', 'B'], .5, 10, 0, 60 * M, 1, 7)
        prior = lambda decoded: [a for group in decoded for a in group if a['issue_ms'] < 25 * M]
        self.assertEqual(prior(before), prior(after))

    def test_zero_cooldown_matches_old_decoder_including_ties(self):
        times = np.arange(1436, 1452) * M
        scores = np.random.default_rng(7).choice([.1, .5, .9, np.nan, np.inf], size=(3, len(times)))
        scores[:, 2] = .5
        scores[:, 3] = .9
        for cap in (1, 3, 20):
            for delay in (0, 1, 5):
                args = times, scores, ['A', 'B', 'C'], .5, cap, 1437 * M, 1450 * M, delay
                self.assertEqual(decode_refractory(*args, cooldown_minutes=0), decode(*args))

    def test_calibration_excludes_future_and_counts_only_valid_minutes(self):
        times = np.arange(20) * M
        scores = np.array([np.linspace(.01, .2, 20), np.linspace(.2, .01, 20)])
        scores[:, 2] = np.nan
        before = calibrate_threshold(times, scores, ['A', 'B'], M, 10 * M, .25, 20, 0, 1)
        self.assertEqual(before['target_alerts'], 2.)
        scores[:, 10:] = 1e8
        scores[:, 0] = 1e9
        after = calibrate_threshold(times, scores, ['A', 'B'], M, 10 * M, .25, 20, 0, 1)
        self.assertEqual(before, after)
        self.assertTrue(np.isfinite(before['threshold']))

    def test_calibration_counts_pending_and_chooses_higher_threshold_on_tie(self):
        times = np.arange(10) * M
        scores = np.arange(10, dtype=float).reshape(1, 10)
        result = calibrate_threshold(times, scores, ['A'], 0, 10 * M, .1, 10, 0, 1)
        self.assertEqual(result['realized_alerts'], 1)
        self.assertEqual(result['threshold'], max(t['threshold'] for t in result['trials']))
        alerts, pending = decode_refractory(times, scores, ['A'], result['threshold'], 10, 0, 10 * M, 1, 0)
        self.assertEqual((len(alerts), len(pending)), (0, 1))

    def test_calibration_rejects_empty_or_nonfinite_window(self):
        times = np.arange(5) * M
        for start, stop, scores in ((0, 5 * M, np.full((1, 5), np.nan)),
                                    (6 * M, 7 * M, np.ones((1, 5)))):
            with self.assertRaises(ValueError):
                calibrate_threshold(times, scores, ['A'], start, stop, .1, 10)


if __name__ == '__main__':
    unittest.main()
