"""Independently verify saved v7 fits, probabilities, alerts and progression."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

from run_entry_management import load_view
from verify_entry_management import matching

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'analysis/results/joint_entry_v6'
DEFAULT_OUT = ROOT / 'analysis/results/robust_entry_v7'
M = 60000
BASELINES = ('clock_uniform', 'v5_factorized')
OLD = BASELINES + ('joint_linear', 'joint_tree')
LEARNED = ('bounded_linear', 'bounded_timing')
VARIANTS = tuple(model + '__' + decoder for model in OLD + LEARNED
                 for decoder in (('original',) if model in BASELINES else ('original', 'dispersed')))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)


def equal(actual, expected):
    """Compare reconstructed numeric cards without rounding their contents."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    else:
        assert actual == expected, (actual, expected)


def short_inputs(close, ns, n):
    """Rebuild own-symbol complete lag windows by explicit lagged slices."""
    close = np.asarray(close, float).reshape(ns, n)
    result = np.full((ns, n, 3), np.nan)
    for column, lag in enumerate((1, 3, 14)):
        pieces = [close[:, lag - back:n - back] for back in range(lag + 1)]
        valid = np.logical_and.reduce([np.isfinite(p) & (p > 0) for p in pieces])
        base = np.maximum.reduce(pieces) if column == 2 else pieces[-1]
        values = np.full(base.shape, np.nan)
        values[valid] = pieces[0][valid] / base[valid] - 1
        result[:, lag:, column] = values
    return result.reshape(ns * n, 3)


def probabilities(y, values):
    p = np.clip(values, 1e-12, 1 - 1e-12)
    count = int(y.sum())
    bins = [0., .0001, .001, .01, .05, .2, 1.000001]
    calibration = []
    for low, high in zip(bins[:-1], bins[1:]):
        at = (p >= low) & (p < high)
        if at.any():
            calibration.append(dict(lower=low, upper=high, minutes=int(at.sum()),
                                    predicted=float(p[at].mean()), observed=float(y[at].mean())))
    return dict(minutes=len(y), events=count, expected_events=float(p.sum()),
                event_mass_ratio=float(p.sum() / count),
                log_loss=float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                brier=float(np.square(p - y).mean()),
                average_precision=float(average_precision_score(y, p)), calibration=calibration)


def replay(times, scores, symbols, threshold, cap, start, stop, delay, cooldown):
    """Independent chronological issue ledger, including end-window pending."""
    used, last = Counter(), {}
    alerts, pending = [], []
    for minute in np.flatnonzero((times >= start) & (times < stop)):
        now = int(times[minute])
        day = now // (1440 * M)
        if used[day] >= cap:
            continue
        choices = [j for j, score in enumerate(scores[:, minute])
                   if np.isfinite(score) and score >= threshold
                   and (j not in last or now - last[j] >= cooldown * M)]
        if not choices:
            continue
        coin = sorted(choices, key=lambda j: (
            -scores[j, minute], hashlib.sha256(f'{symbols[j]}|open|{now}'.encode()).hexdigest()))[0]
        used[day] += 1
        last[coin] = now
        item = dict(symbol=symbols[coin], action='open', issue_ms=now, time_ms=now + delay * M,
                    minute_index=int(minute), score=float(scores[coin, minute]))
        (alerts if item['time_ms'] < stop else pending).append(item)
    return dict(alerts=alerts, pending=pending)


def threshold_trials(times, scores, symbols, start, stop, rate, cap, cooldown):
    window = scores[:, (times >= start) & (times < stop)]
    usable = np.isfinite(window).any(axis=0)
    maxima = np.max(np.where(np.isfinite(window[:, usable]), window[:, usable], -np.inf), axis=0)
    quantiles = sorted(set([.9, .95, .97, .98, .99, .995, .997, .999, .9995, 1 - rate]))
    target, trials = float(rate * usable.sum()), []
    for value in np.unique(np.quantile(maxima, quantiles)):
        decoded = replay(times, scores, symbols, float(value), cap, start, stop, 1, cooldown)
        trials.append(dict(threshold=float(value), realized_alerts=sum(map(len, decoded.values()))))
    best = sorted(trials, key=lambda t: (abs(t['realized_alerts'] - target), -t['threshold']))[0]
    return dict(**best, target_alerts=target, trials=trials)


def check_match(predicted, actual, saved, tolerance):
    result = matching(predicted, actual, tolerance)
    tp, fp, fn = (result[key] for key in ('tp', 'fp', 'fn'))
    result.update(precision=tp / (tp + fp) if tp + fp else 0.,
                  recall=tp / (tp + fn) if tp + fn else 0.,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    equal({key: saved[key] for key in result}, result)
    assert saved['tolerance_ms'] == tolerance and saved['actual_cannot_precede_issue'] is True
    return result


def check_events(alerts, truth, saved):
    assert saved['alerts'] == len(alerts) and saved['actual'] == len(truth)
    card = {name: check_match(alerts, truth, saved[name], tolerance)
            for name, tolerance in (('primary', M), ('sensitivity_5m', 5 * M))}
    check_match([{**e, 'symbol': 'ALL'} for e in alerts], [{**e, 'symbol': 'ALL'} for e in truth],
                saved['timing_only'], M)
    return card


def gains(y, candidate, baseline, times, seed):
    losses = []
    days, indices = np.unique(times // (1440 * M), return_inverse=True)
    for values in (candidate, baseline):
        p = np.clip(values, 1e-12, 1 - 1e-12)
        losses.append(-(y * np.log(p) + (1 - y) * np.log1p(-p)))
    sums = [np.bincount(indices, weights=loss) for loss in losses]
    picks = np.random.default_rng(seed).integers(len(days), size=(500, len(days)))
    replicas = 1 - sums[0][picks].sum(axis=1) / sums[1][picks].sum(axis=1)
    return dict(days=len(days), replicates=500,
                relative_gain=float(1 - losses[0].sum() / losses[1].sum()),
                interval_95=np.quantile(replicas, [.025, .975]).tolist())


def main(out=DEFAULT_OUT):
    plan, result, models, archive, protected = [read(out / name) for name in
        ('protocol.json', 'results.json', 'models.json', 'predictions.json', 'source_hashes.json')]
    old_plan, old_result, old_archive = [read(SOURCE / name) for name in
        ('protocol.json', 'results.json', 'predictions.json')]
    assert sha(out / 'protocol.json') == result['protocol_sha256']
    assert sha(SOURCE / 'protocol.json') == old_result['protocol_sha256']
    required = set(read(SOURCE / 'source_hashes.json'))
    required.update(str(p.relative_to(ROOT)) for p in SOURCE.iterdir() if p.is_file())
    required.update('analysis/' + name for name in ('robust_entry.py', 'research_robust_entry.py',
                    'research_joint_entry.py', 'tests/test_robust_entry.py', 'tests/test_joint_entry.py'))
    assert required == set(protected)
    changed = [name for name, digest in protected.items() if sha(ROOT / name) != digest]
    assert not changed, changed
    assert plan['version'] == result['version'] == 'robust-entry-v7'
    for field in ('folds', 'market_features', 'clock_features', 'target', 'progression'):
        assert plan[field] == old_plan[field]
    assert plan['model'] == old_plan['candidates']['joint_linear']
    assert plan['clipping']['quantiles'] == [.005, .995]
    assert plan['additional_timing_features'] == ['ret1', 'ret3', 'pullback15']
    assert plan['learned_candidates'] == list(LEARNED)
    assert plan['latency_minutes'] == old_plan['decoder']['latency_minutes'] == [1, 2]
    assert plan['dispersed_decoder']['cooldown_minutes'] == 30
    assert plan['dispersed_decoder']['daily_cap'] == old_plan['decoder']['daily_cap']
    assert plan['dispersed_decoder']['threshold_quantiles'] == [.9, .95, .97, .98, .99, .995, .997, .999, .9995, '1-fit_any_open_rate']
    for field in ('seed', 'negative_limit', 'sampling', 'calibration'):
        assert plan['fit'][field] == old_plan['fit'][field]
    assert len(plan['folds']) == len(result['folds']) == 3
    assert len(list(out.glob('*.joblib'))) == plan['budget']['learned_heads'] == 6

    view = load_view()
    times, symbols = view['data']['timeline_ms'], view['meta']['symbols']
    n, ns = len(times), len(symbols)
    row_times = np.tile(times, ns)
    angle = times % (1440 * M) / (1440 * M) * (2 * np.pi)
    clock = np.tile(np.column_stack([np.sin(angle), np.cos(angle)]), (ns, 1))
    inputs = dict(bounded_linear=view['market'], bounded_timing=np.column_stack([
        view['market'], short_inputs(view['features']['close'], ns, n)]))
    ready = np.isfinite(view['market']).all(axis=1) & np.tile(view['timing_ready'], ns)
    label_ok = ready & np.tile(view['timing_ok'], ns)
    y = view['rank_y']
    saved, old_scores = np.load(out / 'scores.npz', allow_pickle=False), np.load(SOURCE / 'scores.npz', allow_pickle=False)
    for key, value in dict(times=times, symbols=np.asarray(symbols), labels=y, public_ready=ready, label_ok=label_ok).items():
        np.testing.assert_array_equal(old_scores[key], value)
    checked, all_truth, fit_checks = [], [], []
    counts = dict.fromkeys([key for key in VARIANTS if key.split('__')[0] not in BASELINES], 0)
    pooled = {key: [] for key in VARIANTS}
    probability_rows, calibration_rows, alerts_checked = 0, 0, 0
    for fold, fc, old_fc in zip(plan['folds'], result['folds'], old_result['folds']):
        name = fold['name']
        assert fc['fold'] == old_fc['fold'] == name and set(fc['variants']) == set(VARIANTS)
        lo, mid, hi, end = [stamp(fold[key]) for key in ('train_start', 'calibration_start', 'train_end', 'eval_end')]
        rows = {part: np.flatnonzero(label_ok & (row_times >= start) & (row_times < stop))
                for part, start, stop in (('fit', lo, mid), ('cal', mid, hi), ('eval', hi, end))}
        for part, value in rows.items():
            np.testing.assert_array_equal(old_scores[name + '_' + part + '_rows'], value)
        truth = [e for e in view['opens'] if hi <= e['time_ms'] < end]
        assert truth == archive[name + '_truth'] == old_archive[name + '_truth']
        all_truth.extend(truth)
        assert fc['actual_openings'] == len(truth) == old_fc['actual_openings']
        assert fc['actual_cycles'] == len({e['cycle_id'] for e in truth}) == old_fc['actual_cycles']
        largest = Counter(e['symbol'] for e in truth).most_common(1)[0][0]
        assert fc['largest_symbol'] == largest
        fit_min = (times >= lo) & (times < mid) & view['timing_ok']
        cal_min = (times >= mid) & (times < hi) & view['timing_ok']
        rate = float(view['timing_y'][fit_min].mean())
        cap = int(np.ceil(2 * view['timing_y'][fit_min].sum() / ((mid - lo) / (1440 * M))))
        assert fc['fit_any_open_rate'] == rate and fc['daily_cap'] == cap
        last, rapid = {}, 0
        for event in sorted(truth, key=lambda e: e['time_ms']):
            coin, now = event['symbol'], event['time_ms']
            rapid += coin in last and now - last[coin] < 30 * M
            last[coin] = now
        assert fc['rapid_reentry_events'] == rapid
        score_map = {model: old_scores[name + '_' + model + '_scores'] for model in OLD}
        for model, market in inputs.items():
            key = name + '_' + model
            available = ready & np.isfinite(market).all(axis=1)
            assert available[rows['cal']].all() and available[rows['eval']].all()
            fit = np.flatnonzero(label_ok & available & (row_times >= lo) & (row_times < mid))
            np.testing.assert_array_equal(saved[key + '_fit_rows'], fit)
            chosen, weights = saved[key + '_sample_rows'], saved[key + '_sample_weights']
            positive, negative = fit[y[fit] == 1], fit[y[fit] == 0]
            selected = np.random.default_rng(plan['fit']['seed']).choice(
                negative, min(len(negative), plan['fit']['negative_limit']), replace=False)
            np.testing.assert_array_equal(chosen, np.sort(np.r_[positive, selected]))
            np.testing.assert_array_equal(weights, np.where(y[chosen] == 1, 1., len(negative) / len(selected)))
            np.testing.assert_allclose([weights.sum(), weights[y[chosen] == 1].sum()], [len(fit), len(positive)])
            lower, upper = np.quantile(market[fit], [.005, .995], axis=0)
            bundle = joblib.load(out / (key + '.joblib'))
            np.testing.assert_array_equal(bundle['clipper'].lower_, lower)
            np.testing.assert_array_equal(bundle['clipper'].upper_, upper)
            assert tuple(bundle['clipper'].quantiles) == (.005, .995)
            scaler, fitted = bundle['scaler'], bundle['model']
            x = np.column_stack([np.minimum(np.maximum(market, lower), upper), clock])
            mean = np.average(x[chosen], axis=0, weights=weights)
            variance = np.average(np.square(x[chosen] - mean), axis=0, weights=weights)
            np.testing.assert_allclose(scaler.mean_, mean, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(scaler.var_, variance, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(scaler.scale_, np.sqrt(variance), rtol=1e-10, atol=1e-12)
            assert fitted.n_features_in_ == market.shape[1] + 2
            np.testing.assert_array_equal(fitted.classes_, [0, 1])
            for parameter, value in plan['model'].items():
                if parameter not in ('family', 'weighted_standard_scaler'):
                    assert fitted.get_params()[parameter] == value
            assert fitted.get_params()['random_state'] == plan['fit']['seed']
            score_map[model] = saved[key + '_scores']
            np.testing.assert_array_equal(np.isfinite(score_map[model]).ravel(), available)
            verify_rows = np.flatnonzero(available & (row_times >= mid) & (row_times < end))
            z = ((x[verify_rows] - scaler.mean_) / scaler.scale_) @ fitted.coef_[0] + fitted.intercept_[0]
            raw = np.clip(expit(z), 1e-12, 1 - 1e-12)
            expected = expit(np.log(raw / (1 - raw)) + bundle['offset'])
            np.testing.assert_allclose(score_map[model].ravel()[verify_rows], expected, rtol=1e-10, atol=1e-12)
            cal = score_map[model].ravel()[rows['cal']]
            np.testing.assert_allclose(cal.mean(), (y[rows['cal']].sum() + .5) / (len(cal) + 1), rtol=1e-10)
            details = dict(natural_rows=len(fit), positives=len(positive), sampled_rows=len(chosen), weight_sum=float(weights.sum()),
                clipping_lower=lower.tolist(), clipping_upper=upper.tolist(),
                features=plan['market_features'] + (plan['additional_timing_features'] if model == 'bounded_timing' else []) + plan['clock_features'],
                calibration_offset=bundle['offset'], calibration_rows=len(rows['cal']), calibration_positives=int(y[rows['cal']].sum()),
                evaluation_clipped_row_fraction=float(np.any(market[rows['eval']] != x[rows['eval'], :market.shape[1]], axis=1).mean()))
            equal(models[key], details)
            probability_rows += int((row_times[verify_rows] >= hi).sum())
            calibration_rows += int((row_times[verify_rows] < hi).sum())
            fit_checks.append(dict(head=key, natural_rows=len(fit), positives=len(positive), sampled_rows=len(chosen)))
        reconstructed = {}
        for model, scores in score_map.items():
            pc = probabilities(y[rows['eval']], scores.ravel()[rows['eval']])
            maximum = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=0)
            original_threshold = float(np.quantile(maximum[cal_min], 1 - rate))
            gain = {base: gains(y[rows['eval']], scores.ravel()[rows['eval']], score_map[base].ravel()[rows['eval']],
                                row_times[rows['eval']], plan['fit']['seed']) for base in BASELINES} if model not in BASELINES else None
            for decoder in (('original',) if model in BASELINES else ('original', 'dispersed')):
                key = model + '__' + decoder
                vc = fc['variants'][key]
                equal(vc['probability'], pc)
                calibration = (dict(threshold=original_threshold) if decoder == 'original' else
                               threshold_trials(times, scores, symbols, mid, hi, rate, cap, 30))
                equal(vc['calibration'], calibration)
                assert len(vc['scenarios']) == 2
                events = []
                for delay, sc in zip((1, 2), vc['scenarios']):
                    assert sc['delay'] == delay
                    decoded = replay(times, scores, symbols, calibration['threshold'], cap, hi, end, delay,
                                     30 if decoder == 'dispersed' else 0)
                    assert archive[name + '_' + key + '_delay' + str(delay)] == decoded
                    assert sc['pending_at_end'] == len(decoded['pending'])
                    alerts = decoded['alerts']
                    assert sc['label_ambiguity_alerts'] == sum(not view['timing_ok'][e['minute_index']] for e in alerts)
                    ec = check_events(alerts, truth, sc)
                    ec['leave_largest_symbol'] = check_events([e for e in alerts if e['symbol'] != largest],
                        [e for e in truth if e['symbol'] != largest], sc['leave_largest_symbol'])
                    events.append(ec)
                    if model in OLD and decoder == 'original':
                        assert decoded == old_archive[name + '_' + model + '_delay' + str(delay)]
                        assert sc == old_fc['variants'][model]['scenarios'][delay - 1]
                        assert vc['probability'] == old_fc['variants'][model]['probability']
                        assert original_threshold == old_fc['variants'][model]['threshold']
                    alerts_checked += len(alerts)
                    checked.append(dict(fold=name, variant=key, delay=delay, **ec['primary']))
                reconstructed[key] = dict(probability=pc, scenarios=events, gain=gain)
                pooled[key].append(reconstructed[key])
                if gain is not None:
                    equal(vc['gain_vs_baselines'], gain)
        for key, card in reconstructed.items():
            if card['gain'] is None:
                continue
            p, g = card['scenarios'][0]['primary'], plan['progression']
            flags = dict(log_loss=all(v['relative_gain'] >= g['relative_log_loss_gain_vs_both_baselines'] for v in card['gain'].values()),
                mass=g['event_mass_ratio'][0] <= card['probability']['event_mass_ratio'] <= g['event_mass_ratio'][1],
                precision=p['precision'] >= g['coin_specific_precision_min'], recall=p['recall'] >= g['coin_specific_recall_min'],
                leave_largest=all(card['scenarios'][0]['leave_largest_symbol']['primary']['f1'] >
                    reconstructed[base + '__original']['scenarios'][0]['leave_largest_symbol']['primary']['f1'] for base in BASELINES))
            assert fc['variants'][key]['gate_checks'] == flags
            assert fc['variants'][key]['gate_pass'] == all(flags.values())
            counts[key] += int(all(flags.values()))
        print('VERIFIED', name, 'fits, score formulas, thresholds, alerts and gates', flush=True)
    for key, cards in pooled.items():
        tp, fp, fn = [sum(c['scenarios'][0]['primary'][metric] for c in cards) for metric in ('tp', 'fp', 'fn')]
        minutes = sum(c['probability']['minutes'] for c in cards)
        expected = dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn),
            f1=2 * tp / (2 * tp + fp + fn),
            log_loss=sum(c['probability']['log_loss'] * c['probability']['minutes'] for c in cards) / minutes,
            event_mass_ratio=sum(c['probability']['expected_events'] for c in cards) / sum(c['probability']['events'] for c in cards),
            sensitivity_5m_tp=sum(c['scenarios'][0]['sensitivity_5m']['tp'] for c in cards))
        equal(result['pooled'][key], expected)
    cycles = len({e['cycle_id'] for e in all_truth})
    assert result['pooled_eval_openings'] == len(all_truth) and result['pooled_eval_cycles'] == cycles
    assert result['gates'] == counts and set(result['pooled']) == set(VARIANTS)
    g = plan['progression']
    qualifying = [key for key, count in counts.items() if count >= g['same_candidate_required_folds']
                  and len(all_truth) >= g['min_pooled_eval_openings'] and cycles >= g['min_pooled_eval_cycles']]
    assert result['qualifying_opening_modules'] == qualifying and result['complete_strategy_qualified'] is False
    assert result['exact_v6_alert_replay'] is True and result['changed_original_files'] == []
    assert result['protected_original_files'] == len(protected)
    assert len(checked) == 60
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    output = dict(status='passed', verified_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(out / 'protocol.json'), source_hashes_sha256=sha(out / 'source_hashes.json'),
        verifier_sha256=sha(Path(__file__)), protected_original_files=len(protected), changed_original_files=[],
        exact_v6_alert_and_metric_replay=True, natural_fit_and_sampling_checks=fit_checks,
        clipping='Natural fit quantiles independently reconstructed; clock columns unmodified.',
        probability_verification='Saved coefficients, scaler and offset; no predict or retraining.',
        independently_recomputed_eval_probability_rows=probability_rows,
        independently_recomputed_calibration_probability_rows=calibration_rows,
        matching_verification='Independent maximum bipartite matching, full raw truth denominators.',
        replayed_alert_scenarios=len(checked), replayed_alerts=alerts_checked, checks=checked,
        pooled_eval_openings=len(all_truth), pooled_eval_cycles=cycles, gates=counts,
        qualifying_opening_modules=qualifying, independent_new_data_validation=False)
    (out / 'verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('VERIFIED', len(checked), 'scenarios;', probability_rows, 'evaluation probability rows;', len(all_truth), 'truth openings')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT, help='Existing completed v7 output directory.')
    main(parser.parse_args().output)
