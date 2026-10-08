"""Rebuild v6 inputs, tree paths, alert decisions and matching without its runner."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
from verify_entry_management import aggregate, binary, matching

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/results/joint_entry_v6'
M = 60000


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()*1000)


def tree_probability(bundle, x):
    """Walk saved numeric split nodes; do not call sklearn prediction methods."""
    model = bundle['model']
    z = np.full(len(x), float(model._baseline_prediction[0, 0]))
    for iteration in model._predictors:
        assert len(iteration) == 1
        nodes = iteration[0].nodes
        assert not nodes['is_categorical'].any()
        locations = np.zeros(len(x), dtype=int)
        active = np.arange(len(x))
        while len(active):
            at = nodes[locations[active]]
            leaves = at['is_leaf'].astype(bool)
            z[active[leaves]] += at['value'][leaves]
            active = active[~leaves]
            at = at[~leaves]
            goes_left = x[active, at['feature_idx']] <= at['num_threshold']
            locations[active] = np.where(goes_left, at['left'], at['right'])
    raw = np.clip(expit(z), 1e-12, 1-1e-12)
    return expit(np.log(raw/(1-raw))+bundle['offset'])


def card(y, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return dict(log_loss=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
                brier=float(np.square(y-p).mean()), expected_events=float(p.sum()),
                event_mass_ratio=float(p.sum()/y.sum()), average_precision=float(average_precision_score(y, p)))


def assert_match(predicted, actual, saved, tolerance):
    answer = matching(predicted, actual, tolerance)
    for k in ('tp', 'fp', 'fn'):
        assert answer[k] == saved[k], (k, answer, saved)
    tp, fp, fn = answer['tp'], answer['fp'], answer['fn']
    assert saved['precision'] == (tp/(tp+fp) if tp+fp else 0.)
    assert saved['recall'] == (tp/(tp+fn) if tp+fn else 0.)
    assert saved['f1'] == (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def main():
    protocol, result, archive = [read(OUT/f) for f in ('protocol.json', 'results.json', 'predictions.json')]
    protected = read(OUT/'source_hashes.json')
    changed = [p for p, digest in protected.items() if sha(ROOT/p) != digest]
    assert not changed, changed
    assert sha(OUT/'protocol.json') == result['protocol_sha256']
    panel = dict(np.load(ROOT/'analysis/results/reverse_panel.npz', allow_pickle=False))
    extra = dict(np.load(ROOT/'analysis/results/observed_behavior_v3/features.npz', allow_pickle=False))
    meta = read(ROOT/'analysis/results/reverse_panel_meta.json')
    source = read(ROOT/'analysis/results/leader_events.json')
    old = read(ROOT/'analysis/results/entry_management_v5/models.json')
    saved = np.load(OUT/'scores.npz', allow_pickle=False)
    symbols, times = meta['symbols'], panel['timeline_ms']
    n, ns = len(times), len(symbols)
    features = {name: panel['X'][:, i] for i, name in enumerate(meta['feature_names'])}
    features.update({str(name): extra['X_extra'][:, i] for i, name in enumerate(extra['feature_names'])})
    market = np.column_stack([features[name] for name in protocol['market_features']])
    agg, count = aggregate(market, ns, n)
    minute_ready = (count >= 20) & np.isfinite(agg).all(axis=1)
    angle = times % (1440*M)/(1440*M)*2*np.pi
    clock = np.c_[np.sin(angle), np.cos(angle)]
    x = np.c_[market, np.tile(clock, (ns, 1))]
    ready = np.isfinite(x).all(axis=1) & np.tile(minute_ready, ns)
    y = np.zeros(ns*n, int)
    any_y = np.zeros(n, int)
    label_minute = minute_ready.copy()
    primary = []
    for e in source:
        if e['symbol'] not in symbols or not times[0] <= e['time_ms'] < times[-1]+M:
            continue
        if e['direction'] != 'SHORT' or e['action'] != 'open':
            continue
        minute = int((e['time_ms']//M*M-times[0])//M)
        row = symbols.index(e['symbol'])*n+minute
        if e['classification'] == 'history_anchored':
            y[row] = 1
            any_y[minute] = 1
            primary.append({**e, 'panel_row': row})
        else:
            label_minute[minute] = False
    label_ok = ready & np.tile(label_minute, ns)
    for key, expected in [('times', times), ('symbols', np.asarray(symbols)), ('labels', y), ('public_ready', ready), ('label_ok', label_ok)]:
        np.testing.assert_array_equal(saved[key], expected)
    checked_alerts, checked_scenarios, score_rows = 0, 0, 0
    gate_counts = {v: 0 for v in protocol['candidates']}
    row_times = np.tile(times, ns)
    for fold, fc in zip(protocol['folds'], result['folds']):
        name = fold['name']
        lo, mid, hi, end = [stamp(fold[k]) for k in ('train_start', 'calibration_start', 'train_end', 'eval_end')]
        rows = {}
        for part, a, b in [('fit', lo, mid), ('cal', mid, hi), ('eval', hi, end)]:
            rows[part] = np.flatnonzero(label_ok & (row_times >= a) & (row_times < b))
            np.testing.assert_array_equal(saved[name+'_'+part+'_rows'], rows[part])
        chosen, weights = saved[name+'_fit_sample_rows'], saved[name+'_fit_sample_weights']
        assert set(chosen[y[chosen] == 1]) == set(rows['fit'][y[rows['fit']] == 1])
        assert np.all(np.isin(chosen, rows['fit']))
        assert np.isclose(weights.sum(), len(rows['fit']))
        assert np.isclose(weights[y[chosen] == 1].sum(), y[rows['fit']].sum())
        assert len(set(chosen)) == len(chosen)
        assert (row_times[rows['fit']].max() < row_times[rows['cal']].min() and
                row_times[rows['cal']].max() < row_times[rows['eval']].min())
        truth = [e for e in primary if hi <= e['time_ms'] < end]
        assert truth == archive[name+'_truth']
        assert len(truth) == fc['actual_openings']
        assert len({e['cycle_id'] for e in truth}) == fc['actual_cycles']
        largest = Counter(e['symbol'] for e in truth).most_common(1)[0][0]
        assert largest == fc['largest_symbol']
        fit_minute = (times >= lo) & (times < mid) & label_minute
        cal_minute = (times >= mid) & (times < hi) & label_minute
        rate = any_y[fit_minute].mean()
        cap = int(np.ceil(2*any_y[fit_minute].sum()/((mid-lo)/(1440*M))))
        assert cap == fc['daily_cap']
        risk = rows['eval']
        prng = np.random.default_rng(20261004)
        probe = np.unique(np.r_[risk[y[risk] == 1], prng.choice(risk, 2048, replace=False), rows['cal'][y[rows['cal']] == 1]])
        count_all = ready.reshape(ns, n).sum(axis=0)
        rank = np.zeros(ns*n)
        rank[ready] = binary(old[name+'_selector']['head'], market[ready].astype(float))
        rank = rank.reshape(ns, n)
        for variant, vc in fc['variants'].items():
            scores = saved[name+'_'+variant+'_scores']
            np.testing.assert_array_equal(np.isfinite(scores).ravel(), ready)
            if variant in protocol['candidates']:
                bundle = joblib.load(OUT/(name+'_'+variant+'.joblib'))
                if variant == 'joint_tree':
                    expected = tree_probability(bundle, x[probe])
                    assert len(bundle['model']._predictors) == 100
                else:
                    scaler, model = bundle['scaler'], bundle['model']
                    raw = np.clip(expit(((x[probe]-scaler.mean_)/scaler.scale_) @ model.coef_[0]+model.intercept_[0]), 1e-12, 1-1e-12)
                    expected = expit(np.log(raw/(1-raw))+bundle['offset'])
                np.testing.assert_allclose(scores.ravel()[probe], expected, rtol=1e-10, atol=1e-12)
            else:
                minute = probe % n
                if variant == 'clock_uniform':
                    expected = binary(old[name+'_clock_timing']['head'], clock[minute])/count_all[minute]
                else:
                    timing = binary(old[name+'_market_clock_timing']['head'], np.c_[clock, agg][minute])
                    expected = timing*rank.ravel()[probe]/rank[:, minute].sum(axis=0)
                np.testing.assert_allclose(scores.ravel()[probe], expected, rtol=1e-10, atol=1e-12)
            score_rows += len(probe)
            for k, value in card(y[risk], scores.ravel()[risk]).items():
                assert np.isclose(value, vc['probability'][k], rtol=1e-10), k
            drop = risk[risk//n != symbols.index(largest)]
            for k, value in card(y[drop], scores.ravel()[drop]).items():
                assert np.isclose(value, vc['leave_largest_probability'][k], rtol=1e-10), k
            maximum = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=0)
            threshold = float(np.quantile(maximum[cal_minute], 1-rate))
            assert threshold == vc['threshold']
            for sc in vc['scenarios']:
                delay = sc['delay']
                key = name+'_'+variant+'_delay'+str(delay)
                aa = archive[key]
                expected, pending, day_count = [], [], Counter()
                for i in np.flatnonzero((times >= hi) & (times < end) & minute_ready):
                    now = int(times[i])
                    if maximum[i] < threshold or day_count[now//(1440*M)] >= cap:
                        continue
                    coins = np.flatnonzero(np.isfinite(scores[:, i]))
                    coin = sorted(coins, key=lambda j: (-scores[j, i], hashlib.sha256(f'{symbols[j]}|open|{now}'.encode()).hexdigest()))[0]
                    day_count[now//(1440*M)] += 1
                    item = (symbols[coin], now, now+delay*M, int(i), float(scores[coin, i]))
                    (expected if now+delay*M < end else pending).append(item)
                extract = lambda records: [(e['symbol'], e['issue_ms'], e['time_ms'], e['minute_index'], e['score']) for e in records]
                assert extract(aa['alerts']) == expected
                assert extract(aa['pending']) == pending
                assert sc['pending_at_end'] == len(pending)
                assert_match(aa['alerts'], truth, sc['primary'], M)
                assert_match(aa['alerts'], truth, sc['sensitivity_5m'], 5*M)
                assert_match([{**e, 'symbol': 'ALL'} for e in aa['alerts']], [{**e, 'symbol': 'ALL'} for e in truth], sc['timing_only'], M)
                pp = [e for e in aa['alerts'] if e['symbol'] != largest]
                tt = [e for e in truth if e['symbol'] != largest]
                assert_match(pp, tt, sc['leave_largest_symbol']['primary'], M)
                assert_match(pp, tt, sc['leave_largest_symbol']['sensitivity_5m'], 5*M)
                checked_alerts += len(expected)
                checked_scenarios += 1
            if variant in protocol['candidates']:
                gains = []
                for base in ('clock_uniform', 'v5_factorized'):
                    gain = 1-vc['probability']['log_loss']/fc['variants'][base]['probability']['log_loss']
                    assert np.isclose(gain, vc['gain_vs_baselines'][base]['relative_gain'])
                    gains.append(gain)
                p = vc['scenarios'][0]['primary']
                g = protocol['progression']
                flags = dict(log_loss=min(gains) >= g['relative_log_loss_gain_vs_both_baselines'],
                    mass=g['event_mass_ratio'][0] <= vc['probability']['event_mass_ratio'] <= g['event_mass_ratio'][1],
                    precision=p['precision'] >= g['coin_specific_precision_min'], recall=p['recall'] >= g['coin_specific_recall_min'],
                    leave_largest=all(vc['scenarios'][0]['leave_largest_symbol']['primary']['f1'] > fc['variants'][base]['scenarios'][0]['leave_largest_symbol']['primary']['f1'] for base in ('clock_uniform', 'v5_factorized')))
                assert flags == vc['gate_checks']
                assert all(flags.values()) == vc['gate_pass']
                gate_counts[variant] += all(flags.values())
    assert gate_counts == result['gates']
    assert checked_scenarios == protocol['budget']['alert_scenarios']
    assert len(list(OUT.glob('*.joblib'))) == protocol['budget']['learned_heads']
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'analysis/tests', '-p', 'test_joint_entry.py', '-v'],
                           cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    output = dict(status='passed', verified_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/'protocol.json'), protected_original_files=len(protected), changed_original_files=changed,
        replayed_alert_scenarios=checked_scenarios, replayed_alerts=checked_alerts, independently_recomputed_probability_rows=score_rows,
        tree_verification='Numeric tree node traversal, no sklearn predict method', matching_verification='Independent maximum bipartite matching',
        source_labels_and_public_inputs_rebuilt=True, gates=gate_counts, tests=tests.stderr,
        independent_new_data_validation=False,
        new_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'analysis/research_joint_entry.py', Path(__file__), ROOT/'analysis/tests/test_joint_entry.py']})
    (OUT/'verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('tests', 'new_source_hashes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
