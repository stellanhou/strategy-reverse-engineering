"""Registered rules: full-calendar single-step and autonomous diagnostics."""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from reverse_rules import registered_configs, Rule, RuleBar
from reverse_metrics import match_events, score_by_action, paired_f1_interval

OUT = Path(__file__).resolve().parent / 'results'
M = 60000
SEED = 20261002


def timestamp(text):
    return int(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp() * 1000)


def scoring_mask(data, meta, cycles, lo, hi):
    mask = (data['time_ms'] >= lo) & (data['time_ms'] < hi) & data['state_known_mask'] & data['label_mask']
    purged = []
    for c in cycles:
        if c['status'] != 'complete':
            continue
        if any(c['start_ms'] < t <= c['end_ms'] for t in [lo, hi]):
            if c['symbol'] not in meta['symbols']:
                continue
            si = meta['symbols'].index(c['symbol'])
            mask &= ~((data['symbol_idx'] == si) &
                      (data['time_ms'] >= c['start_ms'] // M * M) &
                      (data['time_ms'] <= c['end_ms'] // M * M))
            purged.append(c['cycle_id'])
    return mask, purged


def filter_predictions(predictions, mask):
    return [p for p in predictions if mask[p['panel_row']]]


def actual_events(data, events, mask):
    actual = []
    for e in events:
        r = e['creation_row']
        if (e['source_order_type'] == 'MARKET' and e['direction'] == 'SHORT'
                and r >= 0 and mask[r] and data['label_mask'][r]):
            actual.append(dict(symbol=e['symbol'], direction='SHORT', action=e['action'],
                               time_ms=e['order_time_ms'], event_id=e['event_id']))
    return actual


def single_step(cfg, data, meta):
    """Vectorized equivalent crossings on observed states; no state mutation."""
    names, symbols = meta['feature_names'], meta['symbols']
    x, t, si = data['X'], data['time_ms'], data['symbol_idx']
    f = {name: x[:, names.index(name)] for name in names}
    previous = lambda a: np.concatenate((a[:1], a[:-1]))
    adjacent = (si == previous(si)) & (t == previous(t) + M)
    signal_name = 'btc_relative60' if cfg.family == 'relative60_surge_short' else 'ret15'
    signal = f[signal_name]
    if cfg.family == 'time_frequency':
        signal_cross = t % (cfg.schedule_minutes * M) == 0
    else:
        signal_cross = adjacent & np.isfinite(previous(signal)) & (signal >= cfg.entry_threshold) & (previous(signal) < cfg.entry_threshold)
    flat = data['direction'] == 0
    held = data['direction'] == -1
    same_cycle = adjacent & (data['cycle_idx'] == previous(data['cycle_idx'])) & (previous(data['direction']) == -1)
    unchanged_cost = same_cycle & (f['avgcost'] == previous(f['avgcost'])) & (f['add_count'] == previous(f['add_count']))
    returns = f['mark_return']
    close = np.zeros(len(t), dtype=bool)
    if cfg.max_hold_minutes is not None:
        age = f['age_minutes']
        close = same_cycle & (age >= cfg.max_hold_minutes) & (previous(age) < cfg.max_hold_minutes)
    elif cfg.profit_exit_threshold is not None:
        close = unchanged_cost & (returns >= cfg.profit_exit_threshold) & (previous(returns) < cfg.profit_exit_threshold)
    add = np.zeros(len(t), dtype=bool)
    if cfg.allow_adds:
        add = (unchanged_cost & (returns <= -cfg.add_adverse_threshold) &
               (previous(returns) > -cfg.add_adverse_threshold) &
               (f['add_count'] < cfg.max_adds) &
               (f['minutes_last_action'] >= cfg.cooldown_minutes))
    code = np.zeros(len(t), dtype=np.int8)
    code[signal_cross & flat] = 1
    code[add & held] = 2
    code[close & held] = 4
    valid = data['label_mask'] & np.isfinite(data['next_open'])
    indices = np.flatnonzero((code > 0) & valid)
    last_prediction = {}
    predictions = []
    action_names = {1: 'open', 2: 'add', 4: 'close'}
    # Flat-state position features are intentionally undefined. Obtain the
    # latest observable fill only for the explicit prediction cooldown.
    source_events = json.loads((OUT / 'leader_events.json').read_text())
    fills = {s: np.array(sorted(e['time_ms'] for e in source_events
                               if e['symbol'] == s and e['classification'] == 'history_anchored'))
             for s in symbols}
    for r in indices:
        s = symbols[int(si[r])]
        now, action = int(t[r]), action_names[int(code[r])]
        if action != 'close':
            at = np.searchsorted(fills[s], now, side='left') - 1
            prior_fill = int(fills[s][at]) if at >= 0 else -(10**18)
            if now - max(prior_fill, last_prediction.get(s, -(10**18))) < cfg.cooldown_minutes * M:
                continue
        predictions.append(dict(symbol=s, direction='SHORT', action=action, time_ms=now,
                                panel_row=int(r), event_id=f'{cfg.config_id}-single-{r}'))
        last_prediction[s] = now
    return predictions


def autonomous(cfg, data, meta):
    x, names = data['X'], meta['feature_names']
    ret15 = x[:, names.index('ret15')]
    relative60 = x[:, names.index('btc_relative60')]
    marks = x[:, names.index('close')]
    valid = np.isfinite(x[:, :9]).all(axis=1) & np.isfinite(data['next_open'])
    predictions = []
    for si, symbol in enumerate(meta['symbols']):
        rows = np.flatnonzero((data['symbol_idx'] == si) & valid)
        model = Rule(cfg)
        for r in rows:
            bar = RuleBar(int(data['time_ms'][r]), symbol,
                          {'ret15': float(ret15[r]), 'btc_relative60': float(relative60[r])},
                          float(marks[r]), float(data['next_open'][r]))
            for action in model.step(bar):
                predictions.append(dict(symbol=symbol, direction='SHORT', action=action.action,
                                        time_ms=action.time_ms, panel_row=int(r),
                                        event_id=f'{cfg.config_id}-autonomous-{r}',
                                        fill_time_ms=action.fill_time_ms,
                                        quantity=action.quantity, mark_price=action.mark_price,
                                        execution_price=action.execution_price))
    return predictions


def main():
    with np.load(OUT / 'reverse_panel.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    meta = json.loads((OUT / 'reverse_panel_meta.json').read_text())
    events = json.loads((OUT / 'reverse_event_features.json').read_text())['events']
    cycles = json.loads((OUT / 'leader_cycles.json').read_text())
    protocol = json.loads((OUT / 'reverse_protocol.json').read_text())
    main_cfg, baseline_cfg, relative_cfg = registered_configs()
    # The optional mechanism is admitted only using the first development
    # training interval, never by its evaluation scores.
    first = protocol['folds'][0]
    train_mask, _ = scoring_mask(data, meta, cycles, timestamp(first['train_start']), timestamp(first['train_end']))
    vals = data['X'][:, meta['feature_names'].index('btc_relative60')]
    pos = train_mask & data['label_mask'] & (data['label'] == 1) & (data['label_direction'] == -1)
    neg = train_mask & data['open_negative_eligible']
    relative_enabled = bool(pos.sum() >= 15 and vals[pos].mean() > vals[neg].mean())
    candidates = main_cfg + (relative_cfg if relative_enabled else [])
    all_cfg = candidates + baseline_cfg
    paths = {}
    for cfg in all_cfg:
        paths[cfg.config_id] = {'single_step': single_step(cfg, data, meta),
                                'autonomous': autonomous(cfg, data, meta)}
        print('RULE_DONE', cfg.config_id, {k: len(v) for k, v in paths[cfg.config_id].items()}, flush=True)
    fold_results = []
    for fold in protocol['folds']:
        train, purged_train = scoring_mask(data, meta, cycles, timestamp(fold['train_start']), timestamp(fold['train_end']))
        test, purged_test = scoring_mask(data, meta, cycles, timestamp(fold['train_end']), timestamp(fold['eval_end']))
        gold_train, gold_test = actual_events(data, events, train), actual_events(data, events, test)
        modes = {}
        for mode in ['single_step', 'autonomous']:
            trials = []
            for cfg in all_cfg:
                ptrain = filter_predictions(paths[cfg.config_id][mode], train)
                ptest = filter_predictions(paths[cfg.config_id][mode], test)
                evaluation = match_events(ptest, gold_test)
                by_action = score_by_action(ptest, gold_test)
                total_hours = float(test.sum() / 60)
                evaluation['eligible_symbol_hours'] = total_hours
                evaluation['false_positives_per_symbol_hour'] = evaluation['fp'] / total_hours
                for action, score in by_action.items():
                    opportunity = np.ones(len(test), dtype=bool) if mode == 'autonomous' else (
                        data['direction'] == (0 if action == 'open' else -1))
                    hours = float((test & opportunity).sum() / 60)
                    score['eligible_symbol_hours'] = hours
                    score['false_positives_per_symbol_hour'] = score['fp'] / hours if hours else None
                trials.append({'config_id': cfg.config_id,
                               'role': 'baseline' if cfg in baseline_cfg else 'candidate',
                               'train': match_events(ptrain, gold_train),
                               'evaluation': evaluation,
                               'by_action': by_action,
                               'sensitivity_5m': match_events(ptest, gold_test, 300000)})
            best = lambda role: sorted([r for r in trials if r['role'] == role],
                                      key=lambda r: (-r['train']['f1'], r['config_id']))[0]
            candidate, baseline = best('candidate'), best('baseline')
            interval = paired_f1_interval(
                filter_predictions(paths[candidate['config_id']][mode], test),
                filter_predictions(paths[baseline['config_id']][mode], test), gold_test,
                timestamp(fold['train_end']), timestamp(fold['eval_end']))
            modes[mode] = {'selected_candidate': candidate['config_id'],
                           'selected_baseline': baseline['config_id'],
                           'candidate_evaluation': candidate['evaluation'],
                           'baseline_evaluation': baseline['evaluation'],
                           'f1_difference': candidate['evaluation']['f1'] - baseline['evaluation']['f1'],
                           'paired_f1_difference_interval_exploratory': interval,
                           'candidate_by_action': candidate['by_action'],
                           'baseline_by_action': baseline['by_action'], 'all_trials': trials}
        fold_results.append({'name': fold['name'], 'train_events': len(gold_train), 'evaluation_events': len(gold_test),
                             'purged_train_cycle_ids': purged_train, 'purged_evaluation_cycle_ids': purged_test,
                             'modes': modes})
    out = {'version': 'reverse-rules-1', 'status': 'completed_exploratory_diagnostics',
           'configuration_budget': {'evaluated_unique_configurations': len(all_cfg), 'limit': 30,
                                    'relative60_enabled_on_first_training_only': relative_enabled},
           'registered_and_evaluated': [asdict(c) for c in all_cfg],
           'single_step_contract': 'observed pre-minute state; threshold crossings and 30-minute entry/add cooldown; no reduce prediction',
           'autonomous_contract': 'continuous model-only ledger from calendar start; mark from ended candles, fills at next minute open; no truth masks or positions used to generate actions',
           'scoring_contract': 'unobservable tails and boundary-crossing actual cycles removed from scoring only; autonomous actions still run through those periods; all remaining autonomous predictions count, including predictions inconsistent with leader position',
           'size_contract': 'normalized initial quantity 1, not an account simulation or cost-adjusted return',
           'folds': fold_results,
           'prospective_status': 'not_started; all inspected history exploratory'}
    (OUT / 'reverse_rule_predictions.json').write_text(json.dumps(paths, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    (OUT / 'reverse_rules.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': out['status'], 'configurations': len(all_cfg),
                      'fold_f1_differences': {f['name']: {m: r['f1_difference'] for m, r in f['modes'].items()} for f in fold_results}}, indent=2))


if __name__ == '__main__':
    main()
