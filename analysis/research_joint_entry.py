"""Frozen v6 public-only joint opening hazard experiment; no trade execution."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from behavior_hazard import M, causal_match, offset_calibration, probability_card, stable_key
from entry_management import clock_features
from run_entry_management import load_view, stamp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/results/joint_entry_v6'
VARIANTS = ('clock_uniform', 'v5_factorized', 'joint_linear', 'joint_tree')
warnings.filterwarnings('error', category=ConvergenceWarning)


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_inputs(market, times, n_symbols):
    """Pure public row inputs; symbol-major row order, no event/state argument."""
    return np.column_stack([market, np.tile(clock_features(times), (n_symbols, 1))])


def saved_head(details, x):
    z = (np.asarray(x, float) - np.asarray(details['scaler_mean'])) / np.asarray(details['scaler_scale'])
    raw = expit(z @ np.asarray(details['coefficients'][0]) + details['intercept'][0])
    return expit(logit(np.clip(raw, 1e-12, 1-1e-12)) + details['calibration_offset'])


def sample_fit(labels, rows, limit, seed):
    positive = rows[labels[rows] == 1]
    negative = rows[labels[rows] == 0]
    assert len(positive) and len(negative), 'two natural classes required'
    selected = np.random.default_rng(seed).choice(negative, min(limit, len(negative)), replace=False)
    chosen = np.sort(np.r_[positive, selected])
    weights = np.ones(len(chosen))
    weights[labels[chosen] == 0] = len(negative) / len(selected)
    return chosen, weights


def fit_head(variant, x, y, rows, weights, cal_rows, config, seed):
    parameters = {k: v for k, v in config.items() if k not in ('family', 'weighted_standard_scaler')}
    scaler = None
    if variant == 'joint_linear':
        scaler = StandardScaler().fit(x[rows], sample_weight=weights)
        model = LogisticRegression(**parameters, random_state=seed)
        fit_x, cal_x = scaler.transform(x[rows]), scaler.transform(x[cal_rows])
    else:
        model = HistGradientBoostingClassifier(**parameters, random_state=seed)
        fit_x, cal_x = x[rows], x[cal_rows]
    model.fit(fit_x, y[rows], sample_weight=weights)
    offset = offset_calibration(model.predict_proba(cal_x)[:, 1], y[cal_rows])
    return dict(model=model, scaler=scaler, offset=offset)


def predict_head(bundle, x):
    xx = bundle['scaler'].transform(x) if bundle['scaler'] is not None else x
    raw = bundle['model'].predict_proba(xx)[:, 1]
    return expit(logit(np.clip(raw, 1e-12, 1-1e-12)) + bundle['offset'])


def decode(times, scores, symbols, threshold, cap, start, stop, delay):
    """Only current public scores and earlier own alerts affect decisions."""
    used = Counter()
    alerts, pending = [], []
    for minute in np.flatnonzero((times >= start) & (times < stop)):
        now = int(times[minute])
        available = np.flatnonzero(np.isfinite(scores[:, minute]))
        if not len(available):
            continue
        coin = min(available, key=lambda i: (-scores[i, minute], stable_key(symbols[i], 'open', now)))
        probability = float(scores[coin, minute])
        if probability < threshold or used[now // (1440*M)] >= cap:
            continue
        used[now // (1440*M)] += 1
        alert = dict(symbol=symbols[coin], action='open', issue_ms=now, time_ms=now+delay*M,
                     minute_index=int(minute), score=probability)
        (alerts if alert['time_ms'] < stop else pending).append(alert)
    return alerts, pending


def event_result(alerts, truth):
    timing_p = [{**e, 'symbol': 'ALL'} for e in alerts]
    timing_a = [{**e, 'symbol': 'ALL'} for e in truth]
    return dict(primary=causal_match(alerts, truth), sensitivity_5m=causal_match(alerts, truth, 5*M),
                timing_only=causal_match(timing_p, timing_a), alerts=len(alerts), actual=len(truth))


def loss_array(y, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def bootstrap_gain(y, candidate, baseline, times, seed):
    day = times // (1440*M)
    days = np.unique(day)
    c, b = loss_array(y, candidate), loss_array(y, baseline)
    cs = np.array([c[day == d].sum() for d in days])
    bs = np.array([b[day == d].sum() for d in days])
    picks = np.random.default_rng(seed).integers(0, len(days), size=(500, len(days)))
    gains = 1-cs[picks].sum(axis=1)/bs[picks].sum(axis=1)
    return dict(days=len(days), replicates=500, relative_gain=float(1-c.sum()/b.sum()),
                interval_95=np.quantile(gains, [.025, .975]).tolist())


def main():
    assert not (OUT/'results.json').exists(), 'completed frozen run must not be overwritten'
    protocol = read(OUT/'protocol.json')
    protected = read(OUT/'source_hashes.json')
    assert all(sha(ROOT/p) == digest for p, digest in protected.items())
    view = load_view()
    times = view['data']['timeline_ms']
    symbols = view['meta']['symbols']
    n, ns = len(times), len(symbols)
    x = public_inputs(view['market'], times, ns)
    ready = np.isfinite(x).all(axis=1) & np.tile(view['timing_ready'], ns)
    label_ok = ready & np.tile(view['timing_ok'], ns)
    y = view['rank_y']
    row_times = np.tile(times, ns)
    all_rows = np.flatnonzero(ready)
    source_models = read(ROOT/'analysis/results/entry_management_v5/models.json')
    archives, arrays, fits, folds = {}, {}, {}, []
    arrays.update(times=times, symbols=np.asarray(symbols), labels=y, public_ready=ready, label_ok=label_ok)
    for fold in protocol['folds']:
        name = fold['name']
        lo, mid, hi, end = [stamp(fold[k]) for k in ('train_start', 'calibration_start', 'train_end', 'eval_end')]
        fit_rows = np.flatnonzero(label_ok & (row_times >= lo) & (row_times < mid))
        cal_rows = np.flatnonzero(label_ok & (row_times >= mid) & (row_times < hi))
        eval_rows = np.flatnonzero(label_ok & (row_times >= hi) & (row_times < end))
        fit_min = (times >= lo) & (times < mid) & view['timing_ok']
        cal_min = (times >= mid) & (times < hi) & view['timing_ok']
        rate = float(view['timing_y'][fit_min].mean())
        cap = int(np.ceil(2*view['timing_y'][fit_min].sum()/((mid-lo)/(1440*M))))
        chosen, weights = sample_fit(y, fit_rows, protocol['fit']['negative_limit'], protocol['fit']['seed'])
        arrays[name+'_fit_sample_rows'], arrays[name+'_fit_sample_weights'] = chosen, weights
        for part, rows in [('fit', fit_rows), ('cal', cal_rows), ('eval', eval_rows)]:
            arrays[name+'_'+part+'_rows'] = rows
        score_map = {}
        clock = clock_features(times)
        pub_minutes = np.flatnonzero(view['timing_ready'])
        p_clock = np.full(n, np.nan)
        p_market = np.full(n, np.nan)
        p_clock[pub_minutes] = saved_head(source_models[name+'_clock_timing']['head'], clock[pub_minutes])
        p_market[pub_minutes] = saved_head(source_models[name+'_market_clock_timing']['head'],
                                         np.column_stack([clock, view['aggregate']])[pub_minutes])
        count = ready.reshape(ns, n).sum(axis=0)
        uniform = np.full((ns, n), np.nan)
        uniform[:, pub_minutes] = p_clock[pub_minutes]/count[pub_minutes]
        uniform[~ready.reshape(ns, n)] = np.nan
        score_map['clock_uniform'] = uniform
        rank = np.zeros(ns*n)
        rank[all_rows] = saved_head(source_models[name+'_selector']['head'], view['market'][all_rows])
        rank = rank.reshape(ns, n)
        factorized = np.full((ns, n), np.nan)
        factorized[:, pub_minutes] = rank[:, pub_minutes]/rank[:, pub_minutes].sum(axis=0)*p_market[pub_minutes]
        factorized[~ready.reshape(ns, n)] = np.nan
        score_map['v5_factorized'] = factorized
        for variant, config in protocol['candidates'].items():
            print('FIT', name, variant, 'natural rows', len(fit_rows), 'positives', int(y[fit_rows].sum()), flush=True)
            bundle = fit_head(variant, x, y, chosen, weights, cal_rows, config, protocol['fit']['seed'])
            joblib.dump(bundle, OUT/(name+'_'+variant+'.joblib'))
            scores = np.full(ns*n, np.nan)
            scores[all_rows] = predict_head(bundle, x[all_rows])
            score_map[variant] = scores.reshape(ns, n)
            fits[name+'_'+variant] = dict(natural_rows=len(fit_rows), natural_positives=int(y[fit_rows].sum()),
                sampled_rows=len(chosen), weight_sum=float(weights.sum()), positive_weight=float(weights[y[chosen] == 1].sum()),
                calibration_rows=len(cal_rows), calibration_positives=int(y[cal_rows].sum()), offset=bundle['offset'],
                model_parameters=bundle['model'].get_params(), features=protocol['market_features']+protocol['clock_features'])
        truth = [e for e in view['opens'] if hi <= e['time_ms'] < end]
        largest = Counter(e['symbol'] for e in truth).most_common(1)[0][0]
        fc = dict(fold=name, actual_openings=len(truth), actual_cycles=len({e['cycle_id'] for e in truth}),
                  probability_positive_labels=int(y[eval_rows].sum()),
                  missing_public_feature_openings=sum(not ready[e['panel_row']] for e in truth),
                  largest_symbol=largest, fit_any_open_rate=rate, daily_cap=cap, variants={})
        archives[name+'_truth'] = truth
        for variant in VARIANTS:
            scores = score_map[variant]
            arrays[name+'_'+variant+'_scores'] = scores
            max_score = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=0)
            threshold = float(np.quantile(max_score[cal_min], 1-rate))
            assert np.isfinite(threshold)
            p = scores.ravel()[eval_rows]
            pc = probability_card(y[eval_rows], p)
            vc = dict(threshold=threshold, probability=pc, scenarios=[], rank_oracle={})
            hits = {1: 0, 3: 0}
            for e in truth:
                minute = e['panel_row'] % n
                eligible = np.flatnonzero(np.isfinite(scores[:, minute]))
                ordered = sorted(eligible, key=lambda i: (-scores[i, minute], stable_key(symbols[i], 'open', int(times[minute]))))
                for k in hits:
                    hits[k] += symbols.index(e['symbol']) in ordered[:k]
            vc['rank_oracle'] = dict(top1=hits[1], top3=hits[3], events=len(truth), real_entry_time_supplied=True)
            for delay in protocol['decoder']['latency_minutes']:
                alerts, pending = decode(times, scores, symbols, threshold, cap, hi, end, delay)
                archives[name+'_'+variant+'_delay'+str(delay)] = dict(alerts=alerts, pending=pending)
                scenario = event_result(alerts, truth)
                scenario.update(delay=delay, pending_at_end=len(pending),
                    label_ambiguity_alerts=sum(not view['timing_ok'][e['minute_index']] for e in alerts),
                    leave_largest_symbol=event_result([e for e in alerts if e['symbol'] != largest],
                                                     [e for e in truth if e['symbol'] != largest]))
                vc['scenarios'].append(scenario)
            vc['leave_largest_probability'] = probability_card(y[eval_rows[eval_rows//n != symbols.index(largest)]],
                         scores.ravel()[eval_rows[eval_rows//n != symbols.index(largest)]])
            fc['variants'][variant] = vc
            c = vc['scenarios'][0]['primary']
            print('SCORE', name, variant, 'TP/FP/FN', c['tp'], c['fp'], c['fn'], 'mass', round(pc['event_mass_ratio'], 3), flush=True)
        for variant in protocol['candidates']:
            vc = fc['variants'][variant]
            vc['gain_vs_baselines'] = {base: bootstrap_gain(y[eval_rows], score_map[variant].ravel()[eval_rows],
                score_map[base].ravel()[eval_rows], row_times[eval_rows], protocol['fit']['seed'])
                for base in VARIANTS[:2]}
            pr = vc['scenarios'][0]['primary']
            drop_f1 = vc['scenarios'][0]['leave_largest_symbol']['primary']['f1']
            g = protocol['progression']
            vc['gate_checks'] = dict(
                log_loss=all(v['relative_gain'] >= g['relative_log_loss_gain_vs_both_baselines'] for v in vc['gain_vs_baselines'].values()),
                mass=g['event_mass_ratio'][0] <= vc['probability']['event_mass_ratio'] <= g['event_mass_ratio'][1],
                precision=pr['precision'] >= g['coin_specific_precision_min'], recall=pr['recall'] >= g['coin_specific_recall_min'],
                leave_largest=all(drop_f1 > fc['variants'][base]['scenarios'][0]['leave_largest_symbol']['primary']['f1'] for base in VARIANTS[:2]))
            vc['gate_pass'] = all(vc['gate_checks'].values())
        folds.append(fc)
    gates = {v: sum(f['variants'][v]['gate_pass'] for f in folds) for v in protocol['candidates']}
    pooled_events = sum(f['actual_openings'] for f in folds)
    pooled_cycles = len({e['cycle_id'] for k, records in archives.items() if k.endswith('_truth') for e in records})
    qualifying = [v for v, passed in gates.items() if passed >= protocol['progression']['same_candidate_required_folds']
                  and pooled_events >= protocol['progression']['min_pooled_eval_openings']
                  and pooled_cycles >= protocol['progression']['min_pooled_eval_cycles']]
    changed = [p for p, digest in protected.items() if sha(ROOT/p) != digest]
    assert not changed, changed
    np.savez_compressed(OUT/'scores.npz', **arrays)
    write('predictions.json', archives)
    write('models.json', fits)
    write('results.json', dict(version=protocol['version'], completed_at_utc=datetime.now(timezone.utc).isoformat(),
          protocol_sha256=sha(OUT/'protocol.json'), historical_status=protocol['historical_status'],
          budget=protocol['budget'], folds=folds, gates=gates, qualifying_opening_modules=qualifying,
          pooled_eval_openings=pooled_events, pooled_eval_cycles=pooled_cycles,
          protected_original_files=len(protected), changed_original_files=changed,
          decision='opening_module_followup_only' if qualifying else 'no_qualifying_opening_module',
          complete_strategy_qualified=False))
    print('DONE', gates, 'qualifying', qualifying, flush=True)


if __name__ == '__main__':
    main()
