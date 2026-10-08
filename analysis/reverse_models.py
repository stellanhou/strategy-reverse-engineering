"""Six registered, interpretable, single-step model diagnostics."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from reverse_metrics import match_events, paired_f1_interval
from run_reverse_rules import scoring_mask, timestamp, filter_predictions

OUT = Path(__file__).resolve().parent / 'results'
SEED = 20261002
NEGATIVE_TRAIN_LIMIT = 20000
COOLDOWN_MS = 1800000
CONFIGS = [('logistic', c) for c in [.1, 1., 10.]] + [('tree', d) for d in [1, 2, 3]]
ACTIONS = [(1, 'open'), (2, 'add'), (3, 'reduce'), (4, 'close')]
MARKET = ['ret5', 'ret15', 'ret60', 'qv5_vs_prev60_mean', 'volatility60', 'range15', 'btc_relative15', 'btc_relative60']
POSITION = ['age_minutes', 'minutes_last_action', 'add_count', 'mark_return', 'position_multiple']


def predictions(indices, scores, threshold, data, meta, action):
    above = scores > threshold
    eligible = indices[above]
    out, last = [], {}
    for r in eligible:
        s = meta['symbols'][int(data['symbol_idx'][r])]
        t = int(data['time_ms'][r])
        if s in last and t - last[s] < COOLDOWN_MS:
            continue
        out.append(dict(symbol=s, direction='SHORT', action=action, time_ms=t, panel_row=int(r)))
        last[s] = t
    return out


def gold(indices, data, events):
    out = []
    for r in indices:
        if data['label'][r] == 0 or data['label_direction'][r] != -1:
            continue
        e = events[int(data['event_idx'][r])]
        out.append(dict(symbol=e['symbol'], direction='SHORT', action=e['action'],
                        time_ms=e['order_time_ms'], event_id=e['event_id']))
    return out


def cooldown_events(rows):
    out, last = [], {}
    for row in sorted(rows, key=lambda p: (p['symbol'], p['time_ms'])):
        s = row['symbol']
        if s in last and row['time_ms'] - last[s] < COOLDOWN_MS:
            continue
        out.append(row)
        last[s] = row['time_ms']
    return out


def main():
    with np.load(OUT / 'reverse_panel.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    meta = json.loads((OUT / 'reverse_panel_meta.json').read_text())
    events = json.loads((OUT / 'reverse_event_features.json').read_text())['events']
    cycles = json.loads((OUT / 'leader_cycles.json').read_text())
    protocol = json.loads((OUT / 'reverse_protocol.json').read_text())
    rules = json.loads((OUT / 'reverse_rules.json').read_text())
    assert rules['status'] == 'completed_exploratory_diagnostics'
    paths = json.loads((OUT / 'reverse_rule_predictions.json').read_text())
    # Preserve the initial overrun. The allowed primary set is the original
    # declaration order, capped at ten action configurations per model family.
    family_counts = {'logistic': 0, 'tree': 0}
    primary = {}
    excluded = []
    for _, action in ACTIONS:
        primary[action] = []
        for family, value in CONFIGS:
            if family_counts[family] < 10:
                primary[action].append((family, value))
                family_counts[family] += 1
            else:
                excluded.append({'action': action, 'family': family, 'parameter': value})
    assert 30 + sum(map(len, primary.values())) == 50
    folds = []
    for fold in protocol['folds']:
        train, _ = scoring_mask(data, meta, cycles, timestamp(fold['train_start']), timestamp(fold['train_end']))
        test, _ = scoring_mask(data, meta, cycles, timestamp(fold['train_end']), timestamp(fold['eval_end']))
        modules = {}
        for label, action in ACTIONS:
            features = MARKET + (POSITION if action != 'open' else [])
            cols = [meta['feature_names'].index(f) for f in features]
            opportunity = (data['direction'] == 0) if action == 'open' else (data['direction'] == -1)
            eligible = data['label_mask'] & opportunity & np.isfinite(data['X'][:, cols]).all(axis=1)
            tr, te = np.flatnonzero(train & eligible), np.flatnonzero(test & eligible)
            ytr = (data['label'][tr] == label) & (data['label_direction'][tr] == -1)
            yte = (data['label'][te] == label) & (data['label_direction'][te] == -1)
            if not ytr.any() or not yte.any():
                modules[action] = {'status': 'insufficient_positive_events', 'train_positives': int(ytr.sum()), 'evaluation_positives': int(yte.sum())}
                continue
            rng = np.random.default_rng(SEED)
            neg = np.flatnonzero(~ytr)
            sampled_neg = rng.choice(neg, size=min(len(neg), NEGATIVE_TRAIN_LIMIT), replace=False)
            fit = np.sort(np.concatenate((np.flatnonzero(ytr), sampled_neg)))
            xtr, xte = data['X'][tr][:, cols], data['X'][te][:, cols]
            scaler = StandardScaler().fit(xtr[fit])
            zfit, ztr, zte = scaler.transform(xtr[fit]), scaler.transform(xtr), scaler.transform(xte)
            # A fixed training-only alert quota determines one threshold. Strict
            # > handles coarse tree ties without manufacturing precise timing.
            quota = min(len(tr), 2 * int(ytr.sum()))
            actual_train = [e for e in gold(tr, data, events) if e['action'] == action]
            actual_test = [e for e in gold(te, data, events) if e['action'] == action]
            registered_baselines = []
            for config_id, path in paths.items():
                action_predictions = cooldown_events([p for p in path['single_step'] if p['action'] == action])
                btrain = filter_predictions(action_predictions, train & eligible)
                btest = filter_predictions(action_predictions, test & eligible)
                registered_baselines.append({'config_id': config_id,
                                             'train_event': match_events(btrain, actual_train),
                                             'evaluation_event': match_events(btest, actual_test),
                                             'evaluation_predictions': btest})
            simple_names = ['ret15', 'btc_relative60'] if action == 'open' else ['mark_return']
            simple_sign = -1 if action in ['add', 'reduce'] else 1
            simple = []
            for name in simple_names:
                train_score = data['X'][tr, meta['feature_names'].index(name)] * simple_sign
                test_score = data['X'][te, meta['feature_names'].index(name)] * simple_sign
                cutoff = float(np.partition(train_score, len(train_score) - quota)[len(train_score) - quota])
                simple.append({'feature': name, 'sign': simple_sign,
                               'train_ap': float(average_precision_score(ytr, train_score)),
                               'evaluation_ap': float(average_precision_score(yte, test_score)),
                               'train_event': match_events(predictions(tr, train_score, cutoff, data, meta, action), actual_train),
                               'evaluation_event': match_events(predictions(te, test_score, cutoff, data, meta, action), actual_test)})
                simple[-1]['evaluation_predictions'] = predictions(te, test_score, cutoff, data, meta, action)
            proxy_baseline = sorted(simple, key=lambda r: (-r['train_event']['f1'], -r['train_ap'], r['feature']))[0]
            event_options = registered_baselines + [dict(proxy_baseline, config_id='fixed_score_' + proxy_baseline['feature'])]
            baseline = sorted(event_options, key=lambda r: (-r['train_event']['f1'], r['config_id']))[0]
            trials = []
            for family, value in primary[action]:
                if family == 'logistic':
                    model = LogisticRegression(C=value, class_weight='balanced', solver='liblinear', max_iter=500, random_state=SEED)
                else:
                    model = DecisionTreeClassifier(max_depth=value, min_samples_leaf=50, class_weight='balanced', random_state=SEED)
                model.fit(zfit, ytr[fit])
                ptr, pte = model.predict_proba(ztr)[:, 1], model.predict_proba(zte)[:, 1]
                threshold = float(np.partition(ptr, len(ptr) - quota)[len(ptr) - quota])
                p_train = predictions(tr, ptr, threshold, data, meta, action)
                p_test = predictions(te, pte, threshold, data, meta, action)
                importance = model.coef_[0] if family == 'logistic' else model.feature_importances_
                trials.append({'config_id': f'{family}_{value}', 'family': family, 'parameter': value,
                               'train_ap': float(average_precision_score(ytr, ptr)),
                               'evaluation_ap': float(average_precision_score(yte, pte)),
                               'threshold_from_training': threshold,
                               'train_event': match_events(p_train, actual_train),
                               'evaluation_event': match_events(p_test, actual_test),
                               'evaluation_predictions': p_test,
                               'top_features': sorted(zip(features, importance.tolist()), key=lambda r: -abs(r[1]))[:5]})
            selected = sorted(trials, key=lambda r: (-r['train_event']['f1'], -r['train_ap'], r['config_id']))[0]
            interval = paired_f1_interval(selected['evaluation_predictions'], baseline['evaluation_predictions'],
                                         actual_test, timestamp(fold['train_end']), timestamp(fold['eval_end']))
            modules[action] = {'status': 'exploratory_single_step_only',
                               'train_positive_events': int(ytr.sum()), 'evaluation_positive_events': int(yte.sum()),
                               'train_opportunities': len(tr), 'evaluation_opportunities': len(te),
                               'natural_evaluation_event_rate': float(yte.mean()),
                               'training_fit_rows': len(fit), 'negative_sampling_probability': len(sampled_neg) / len(neg),
                               'features': features, 'training_alert_quota_before_cooldown': quota,
                               'selected_by_training_only': selected['config_id'],
                               'selected_evaluation': selected['evaluation_event'],
                               'selected_evaluation_ap': selected['evaluation_ap'],
                               'baseline': baseline,
                               'ap_score_proxy_baseline': proxy_baseline,
                               'f1_difference': selected['evaluation_event']['f1'] - baseline['evaluation_event']['f1'],
                               'paired_f1_difference_interval_exploratory': interval,
                               'ap_difference': selected['evaluation_ap'] - proxy_baseline['evaluation_ap'],
                               'all_trials': trials}
            print('MODEL_DONE', fold['name'], action, modules[action]['f1_difference'], flush=True)
        folds.append({'name': fold['name'], 'modules': modules})
    result = {'status': 'completed_exploratory_single_step_comparison',
              'configurations': [{'family': f, 'parameter': p} for f, p in CONFIGS],
              'candidate_budget': {'simple_rules': 30, 'primary_action_model_configurations': 20,
                                   'primary_total': 50, 'actual_unique_configurations_executed_initially': 54,
                                   'excess_configurations': excluded, 'maximum': 50, 'execution_budget_pass': False,
                                   'initial_complete_results_preserved': 'reverse_models_initial_54.json',
                                   'primary_exclusion_rule': 'original action and configuration order; maximum ten action-specific configurations per model family; no outcome-based exclusion'},
              'model_scope': 'separate binary action modules conditioned on observed leader position; scores are rankings, not calibrated probabilities; no autonomous model policy or return claim',
              'threshold_contract': 'one training-derived strict score threshold per module; training quota twice positive count before 30-minute per-action cooldown; no evaluation-period threshold adjustment',
              'evaluation_contract': 'full natural-frequency opportunity panel, purged cycles, event one-to-one +/-60 seconds',
              'sampling_contract': 'at most 20000 deterministic training negatives per module, all positives; class-balanced fitting; training-only scaler; full evaluation negatives',
              'folds': folds}
    (OUT / 'reverse_models.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
