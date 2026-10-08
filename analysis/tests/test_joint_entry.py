"""Causality and accounting checks required by the frozen opening protocol."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_joint_entry import M, public_inputs, sample_fit, decode, event_result


class JointEntryChecks(unittest.TestCase):
    def test_future_market_mutation_preserves_prior_inputs_and_alerts(self):
        times = np.arange(12)*M
        market = np.arange(24*14, dtype=float).reshape(24, 14)
        x = public_inputs(market, times, 2)
        changed = market.reshape(2, 12, 14).copy()
        changed[:, 7:] = -999.
        other = public_inputs(changed.reshape(24, 14), times, 2)
        for coin in range(2):
            np.testing.assert_array_equal(x[coin*12:coin*12+7], other[coin*12:coin*12+7])
        scores = np.full((2, 12), .4)
        a, _ = decode(times, scores, ['A', 'B'], .3, 4, 0, 12*M, 1)
        scores[:, 7:] = 1.
        b, _ = decode(times, scores, ['A', 'B'], .3, 4, 0, 12*M, 1)
        self.assertEqual(a, b)

    def test_sample_weights_retain_natural_frequency(self):
        y = np.zeros(120, int)
        y[[3, 30, 60]] = 1
        eligible = np.arange(100)
        rows, weights = sample_fit(y, eligible, 10, 1)
        self.assertEqual(set(rows[y[rows] == 1]), {3, 30, 60})
        self.assertAlmostEqual(weights.sum(), 100.)
        self.assertAlmostEqual(weights[y[rows] == 0].sum(), 97.)
        self.assertTrue(np.all(rows < 100))

    def test_non_event_minute_can_emit_without_real_state_or_truth(self):
        times = np.arange(3)*M
        scores = np.array([[.01, .8, .01], [.01, .2, .01]])
        alerts, _ = decode(times, scores, ['A', 'B'], .5, 1, 0, 3*M, 1)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['symbol'], 'A')
        self.assertEqual(event_result(alerts, [])['primary']['fp'], 1)

    def test_same_minute_second_symbol_is_a_miss(self):
        times = np.arange(3)*M
        alerts, _ = decode(times, np.array([[.9, 0, 0], [.8, 0, 0]]), ['A', 'B'], .5, 1, 0, 3*M, 1)
        truth = [dict(symbol=s, action='open', time_ms=10000) for s in ['A', 'B']]
        card = event_result(alerts, truth)['primary']
        self.assertEqual((card['tp'], card['fp'], card['fn']), (1, 0, 1))

    def test_cannot_retroactively_match_past_event(self):
        alert = dict(symbol='A', action='open', issue_ms=2*M, time_ms=3*M)
        truth = [dict(symbol='A', action='open', time_ms=M)]
        self.assertEqual(event_result([alert], truth)['sensitivity_5m']['tp'], 0)

    def test_final_pending_alarm_cannot_get_match_credit(self):
        times = np.arange(3)*M
        alerts, pending = decode(times, np.array([[0., 0., 1.]]), ['A'], .5, 1, 0, 3*M, 1)
        self.assertEqual(len(alerts), 0)
        self.assertEqual(len(pending), 1)

    def test_missing_public_feature_coin_stays_in_truth_denominator(self):
        times = np.arange(3)*M
        alerts, _ = decode(times, np.array([[.8, 0, 0], [np.nan]*3]), ['A', 'B'], .5, 1, 0, 3*M, 1)
        truth = [dict(symbol='B', action='open', time_ms=10000)]
        card = event_result(alerts, truth)['primary']
        self.assertEqual((card['tp'], card['fp'], card['fn']), (0, 1, 1))


if __name__ == '__main__':
    unittest.main()
