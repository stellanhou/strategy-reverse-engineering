"""Trace the saved first round; reproduce selected fits without a new search."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from reverse_models import MARKET, POSITION, SEED, NEGATIVE_TRAIN_LIMIT, predictions, gold
from reverse_metrics import match_events
from run_reverse_rules import scoring_mask, timestamp, actual_events, single_step, filter_predictions
from reverse_rules import RuleConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/results'
DEST = OUT / 'failure_diagnosis'
M = 60000
ACTIONS = {'open': 1, 'add': 2, 'reduce': 3, 'close': 4}


def read(name):
    return json.loads((OUT / name).read_text())


def hashes(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def pairs(predicted, actual, tolerance=M):
    """Return identities as well as counts, using the original matching contract."""
    pp, aa = defaultdict(list), defaultdict(list)
    for i, row in enumerate(predicted):
        pp[(row['symbol'], row['direction'], row['action'])].append((i, row))
    for i, row in enumerate(actual):
        aa[(row['symbol'], row['direction'], row['action'])].append((i, row))
    matched_p, matched_a = set(), set()
    for key in sorted(set(pp) | set(aa)):
        p = sorted(pp[key], key=lambda v: (v[1]['time_ms'], v[1].get('event_id', '')))
        a = sorted(aa[key], key=lambda v: (v[1]['time_ms'], v[1].get('event_id', '')))
        i = j = 0
        while i < len(p) and j < len(a):
            delta = p[i][1]['time_ms'] - a[j][1]['time_ms']
            if delta < -tolerance:
                i += 1
            elif delta > tolerance:
                j += 1
            else:
                matched_p.add(p[i][0]); matched_a.add(a[j][0])
                i += 1; j += 1
    score = match_events(predicted, actual, tolerance)
    assert len(matched_p) == len(matched_a) == score['tp']
    return matched_p, matched_a, score


def nearby(rows, event, tolerance):
    return [r for r in rows if r['symbol'] == event['symbol'] and r['action'] == event['action']
            and r['direction'] == event['direction'] and abs(r['time_ms'] - event['time_ms']) <= tolerance]


def nearest(rows, event):
    candidates = [r for r in rows if all(r[k] == event[k] for k in ['symbol', 'direction', 'action'])]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (abs(r['time_ms'] - event['time_ms']), r['time_ms']))


def basic_error(scope, fold, config, kind, row, reason):
    return dict(scope=scope, fold=fold, config=config, kind=kind, action=row['action'],
                symbol=row['symbol'], time_ms=row['time_ms'], time_utc=utc(row['time_ms']),
                event_id=row.get('event_id'), panel_row=row.get('panel_row'), reason=reason)


def assert_score(score, saved):
    for key in ['tp', 'fp', 'fn', 'predictions', 'actual_events']:
        assert score[key] == saved[key], (key, score, saved)


def model_trace(indices, scores, threshold, data, meta, action):
    raw = [dict(symbol=meta['symbols'][int(data['symbol_idx'][r])], direction='SHORT',
                action=action, time_ms=int(data['time_ms'][r]), panel_row=int(r))
           for r in indices[scores > threshold]]
    kept = predictions(indices, scores, threshold, data, meta, action)
    kept_rows = {p['panel_row'] for p in kept}
    suppressed = [p for p in raw if p['panel_row'] not in kept_rows]
    return raw, kept, suppressed


def stage_summary(indices, scores, threshold, target, data, meta, action):
    raw, kept, suppressed = model_trace(indices, scores, threshold, data, meta, action)
    _, _, raw_score = pairs(raw, target)
    _, _, kept_score = pairs(kept, target)
    windows = 0
    previous = {}
    for p in raw:
        if p['symbol'] not in previous or p['time_ms'] != previous[p['symbol']] + M:
            windows += 1
        previous[p['symbol']] = p['time_ms']
    return dict(opportunities=len(indices), actual_events=len(target),
                threshold=threshold, above_threshold=len(raw), equal_threshold=int((scores == threshold).sum()),
                consecutive_above_threshold_runs=windows, removed_by_cooldown=len(suppressed),
                final_signals=len(kept), before_cooldown_60s=raw_score, after_cooldown_60s=kept_score,
                after_cooldown_5m=match_events(kept, target, 5 * M),
                score_quantiles={str(q): float(np.quantile(scores, q)) for q in [0, .5, .99, .999, 1]})


def diagnose_models(data, meta, events, cycles, protocol, models, errors, score_archive, signal_traces):
    summaries = []
    for fold, saved_fold in zip(protocol['folds'], models['folds']):
        assert fold['name'] == saved_fold['name']
        train, _ = scoring_mask(data, meta, cycles, timestamp(fold['train_start']), timestamp(fold['train_end']))
        test, _ = scoring_mask(data, meta, cycles, timestamp(fold['train_end']), timestamp(fold['eval_end']))
        for action, label in ACTIONS.items():
            saved = saved_fold['modules'][action]
            selected = next(t for t in saved['all_trials'] if t['config_id'] == saved['selected_by_training_only'])
            features = MARKET + (POSITION if action != 'open' else [])
            cols = [meta['feature_names'].index(f) for f in features]
            opportunity = data['direction'] == (0 if action == 'open' else -1)
            eligible = data['label_mask'] & opportunity & np.isfinite(data['X'][:, cols]).all(axis=1)
            tr, te = np.flatnonzero(train & eligible), np.flatnonzero(test & eligible)
            ytr = (data['label'][tr] == label) & (data['label_direction'][tr] == -1)
            yte = (data['label'][te] == label) & (data['label_direction'][te] == -1)
            neg = np.flatnonzero(~ytr)
            sampled = np.random.default_rng(SEED).choice(neg, size=min(len(neg), NEGATIVE_TRAIN_LIMIT), replace=False)
            fit = np.sort(np.concatenate((np.flatnonzero(ytr), sampled)))
            xtr, xte = data['X'][tr][:, cols], data['X'][te][:, cols]
            scaler = StandardScaler().fit(xtr[fit])
            zfit, ztr, zte = scaler.transform(xtr[fit]), scaler.transform(xtr), scaler.transform(xte)
            family, value = selected['family'], selected['parameter']
            if family == 'logistic':
                model = LogisticRegression(C=value, class_weight='balanced', solver='liblinear', max_iter=500, random_state=SEED)
            else:
                model = DecisionTreeClassifier(max_depth=value, min_samples_leaf=50, class_weight='balanced', random_state=SEED)
            model.fit(zfit, ytr[fit])
            ptr, pte = model.predict_proba(ztr)[:, 1], model.predict_proba(zte)[:, 1]
            quota = min(len(tr), 2 * int(ytr.sum()))
            threshold = float(np.partition(ptr, len(ptr) - quota)[len(ptr) - quota])
            assert np.isclose(threshold, selected['threshold_from_training'], atol=1e-14, rtol=0)
            target_train = [e for e in gold(tr, data, events) if e['action'] == action]
            target = [e for e in gold(te, data, events) if e['action'] == action]
            raw, kept, suppressed = model_trace(te, pte, threshold, data, meta, action)
            assert kept == selected['evaluation_predictions']
            assert np.isclose(average_precision_score(yte, pte), selected['evaluation_ap'], atol=1e-14, rtol=0)
            pmatched, amatched, score = pairs(kept, target)
            assert_score(score, selected['evaluation_event'])
            train_summary = stage_summary(tr, ptr, threshold, target_train, data, meta, action)
            assert_score(train_summary['after_cooldown_60s'], selected['train_event'])
            evaluation_summary = stage_summary(te, pte, threshold, target, data, meta, action)
            row_score = dict(zip(te.tolist(), pte.tolist()))
            for split, indices, scores in [('train', tr, ptr), ('evaluation', te, pte)]:
                raw_split, kept_split, _ = model_trace(indices, scores, threshold, data, meta, action)
                accepted_rows = {p['panel_row'] for p in kept_split}
                values = dict(zip(indices.tolist(), scores.tolist()))
                last = {}
                for p in raw_split:
                    prior = last.get(p['symbol'])
                    accepted = p['panel_row'] in accepted_rows
                    signal_traces.append(dict(fold=fold['name'], action=action, config=selected['config_id'],
                                              split=split, symbol=p['symbol'], panel_row=p['panel_row'],
                                              time_ms=p['time_ms'], time_utc=utc(p['time_ms']),
                                              score=values[p['panel_row']], threshold=threshold,
                                              status='accepted' if accepted else 'cooldown_suppressed',
                                              prior_accepted_time_ms=prior,
                                              gap_ms=p['time_ms'] - prior if prior is not None else None))
                    if accepted:
                        last[p['symbol']] = p['time_ms']
                    else:
                        assert prior is not None and p['time_ms'] - prior < 30 * M
            positive_scores = pte[yte]
            sorted_scores = np.sort(pte)
            event_ranks = (len(te) - np.searchsorted(sorted_scores, positive_scores, side='right') + 1).tolist()
            for i, event in enumerate(target):
                if i in amatched:
                    continue
                si = meta['symbols'].index(event['symbol'])
                near_mask = (data['symbol_idx'][te] == si) & (np.abs(data['time_ms'][te] - event['time_ms']) <= M)
                window_scores = pte[near_mask]
                raw_near, kept_near = nearby(raw, event, M), nearby(kept, event, M)
                if kept_near:
                    reason = 'one_to_one_competition'
                elif raw_near:
                    reason = 'cooldown_suppressed_all_near_signals'
                elif np.any(window_scores == threshold):
                    reason = 'strict_threshold_tie'
                else:
                    reason = 'below_threshold_in_60s_window'
                e = events[next(j for j, e in enumerate(events) if e['event_id'] == event['event_id'])]
                r = e['creation_row']
                n = nearest(kept, event)
                err = basic_error('model', fold['name'], selected['config_id'], 'FN', event, reason)
                err.update(panel_row=r, score_at_event=row_score[r], threshold=threshold,
                           score_rank_at_event=int(len(te) - np.searchsorted(sorted_scores, row_score[r], side='right') + 1),
                           max_score_in_60s=float(window_scores.max()), raw_signals_in_60s=len(raw_near),
                           final_signals_in_60s=len(kept_near), final_signals_in_5m=len(nearby(kept, event, 5 * M)),
                           nearest_same_action_delta_ms=n['time_ms'] - event['time_ms'] if n else None)
                errors.append(err)
            for i, pred in enumerate(kept):
                if i in pmatched:
                    continue
                n = nearest(target, pred)
                reason = 'no_scored_event_same_symbol_action' if n is None else (
                    'one_to_one_competition' if abs(n['time_ms'] - pred['time_ms']) <= M else
                    'timing_60s_to_5m' if abs(n['time_ms'] - pred['time_ms']) <= 5 * M else 'no_event_within_5m')
                err = basic_error('model', fold['name'], selected['config_id'], 'FP', pred, reason)
                err.update(score_at_event=row_score[pred['panel_row']], threshold=threshold,
                           nearest_same_action_delta_ms=n['time_ms'] - pred['time_ms'] if n else None)
                errors.append(err)
            key = fold['name'] + '_' + action
            score_archive[key + '_rows'] = te
            score_archive[key + '_scores'] = pte
            relevant = [e for e in errors if e['scope'] == 'model' and e['fold'] == fold['name'] and e['action'] == action]
            summaries.append(dict(fold=fold['name'], action=action, config=selected['config_id'],
                                  training_alert_quota=quota, train=train_summary, evaluation=evaluation_summary,
                                  evaluation_ap=selected['evaluation_ap'], natural_event_rate=float(yte.mean()),
                                  event_rank_quantiles={str(q): float(np.quantile(event_ranks, q)) for q in [0, .5, 1]},
                                  false_negative_reasons=dict(Counter(e['reason'] for e in relevant if e['kind'] == 'FN')),
                                  false_positive_reasons=dict(Counter(e['reason'] for e in relevant if e['kind'] == 'FP')),
                                  exact_saved_predictions_and_metrics_reproduced=True))
            print('TRACED_MODEL', fold['name'], action, flush=True)
    return summaries


def rule_conditions(cfg, data, meta):
    t, si, x = data['time_ms'], data['symbol_idx'], data['X']
    f = {n: x[:, j] for j, n in enumerate(meta['feature_names'])}
    prev = lambda a: np.concatenate((a[:1], a[:-1]))
    adjacent = (si == prev(si)) & (t == prev(t) + M)
    same = adjacent & (data['cycle_idx'] == prev(data['cycle_idx'])) & (prev(data['direction']) == -1)
    unchanged = same & (f['avgcost'] == prev(f['avgcost'])) & (f['add_count'] == prev(f['add_count']))
    signal = f['btc_relative60' if cfg.family == 'relative60_surge_short' else 'ret15']
    entry_cross = adjacent & np.isfinite(prev(signal)) & (signal >= cfg.entry_threshold) & (prev(signal) < cfg.entry_threshold)
    profit_cross = unchanged & (f['mark_return'] >= cfg.profit_exit_threshold) & (prev(f['mark_return']) < cfg.profit_exit_threshold)
    adverse_cross = unchanged & (f['mark_return'] <= -cfg.add_adverse_threshold) & (prev(f['mark_return']) > -cfg.add_adverse_threshold)
    return f, dict(open=entry_cross & (data['direction'] == 0),
                   add=adverse_cross & (data['direction'] == -1),
                   close=profit_cross & (data['direction'] == -1)), profit_cross


def rule_candidate_reason(cfg, r, action, data, features, close_cross, fills, accepted):
    if action == 'add':
        if features['add_count'][r] >= cfg.max_adds:
            return 'add_count_limit_in_60s'
        if features['minutes_last_action'][r] < cfg.cooldown_minutes:
            return 'cooldown_in_60s'
        if close_cross[r]:
            return 'close_action_priority'
    if not data['label_mask'][r]:
        return 'candidate_label_excluded'
    if not np.isfinite(data['next_open'][r]):
        return 'candidate_next_open_missing'
    if action != 'close':
        now = int(data['time_ms'][r])
        si = int(data['symbol_idx'][r])
        at = np.searchsorted(fills[si], now, side='left') - 1
        if at >= 0 and now - fills[si][at] < cfg.cooldown_minutes * M:
            return 'observed_fill_cooldown_in_60s'
        at = np.searchsorted(accepted[si], now, side='left') - 1
        if at >= 0 and now - accepted[si][at] < cfg.cooldown_minutes * M:
            return 'prior_prediction_cooldown_in_60s'
    return 'accepted'


def diagnose_rules(data, meta, events, cycles, protocol, rules, paths, errors):
    configs = {c['config_id']: RuleConfig(**c) for c in rules['registered_and_evaluated']}
    summaries, traces = [], {}
    all_fills = read('leader_events.json')
    fills = {si: np.array(sorted(e['time_ms'] for e in all_fills
                                if e['symbol'] == s and e['classification'] == 'history_anchored'))
             for si, s in enumerate(meta['symbols'])}
    for fold, saved_fold in zip(protocol['folds'], rules['folds']):
        test, _ = scoring_mask(data, meta, cycles, timestamp(fold['train_end']), timestamp(fold['eval_end']))
        target = actual_events(data, events, test)
        for mode in ['single_step', 'autonomous']:
            saved = saved_fold['modes'][mode]
            name = saved['selected_candidate']
            cfg = configs[name]
            kept = filter_predictions(paths[name][mode], test)
            pm, am, score = pairs(kept, target)
            assert_score(score, saved['candidate_evaluation'])
            if mode == 'single_step' and name not in traces:
                assert single_step(cfg, data, meta) == paths[name][mode]
                traces[name] = rule_conditions(cfg, data, meta)
            accepted = {si: np.array(sorted(p['time_ms'] for p in paths[name][mode] if p['symbol'] == s))
                        for si, s in enumerate(meta['symbols'])}
            f = {n: data['X'][:, j] for j, n in enumerate(meta['feature_names'])}
            for i, event in enumerate(target):
                if i in am:
                    continue
                e = next(e for e in events if e['event_id'] == event['event_id'])
                r = e['creation_row']
                flags = []
                if event['action'] == 'reduce':
                    reason = 'reduce_action_not_implemented'
                    flags.append(reason)
                elif mode == 'autonomous':
                    reason = 'no_autonomous_action_in_60s'
                    if nearby(kept, event, M):
                        reason = 'one_to_one_competition'
                else:
                    _, crosses, close = traces[name]
                    si = data['symbol_idx'][r]
                    window = np.flatnonzero(test & (data['symbol_idx'] == si) & (np.abs(data['time_ms'] - event['time_ms']) <= M))
                    action = event['action']
                    candidates = window[crosses[action][window]]
                    candidate_reasons = [dict(panel_row=int(cr), reason=rule_candidate_reason(
                        cfg, cr, action, data, f, close, fills, accepted)) for cr in candidates]
                    if action == 'add':
                        if f['add_count'][r] >= cfg.max_adds:
                            flags.append('add_count_limit_at_event')
                        if f['minutes_last_action'][r] < cfg.cooldown_minutes:
                            flags.append('cooldown_at_event')
                        if f['mark_return'][r] > -cfg.add_adverse_threshold:
                            flags.append('adverse_level_not_reached_at_event')
                    if not len(candidates):
                        reason = 'no_entry_crossing_in_60s' if action == 'open' else 'no_profit_crossing_in_60s'
                        if action == 'add':
                            reason = 'no_adverse_crossing_in_60s'
                    else:
                        # Any passing row must be present in the original path;
                        # for all blocked rows retain each exact gate in the CSV.
                        blocked = [c['reason'] for c in candidate_reasons]
                        reason = blocked[0] if len(set(blocked)) == 1 else 'different_candidate_gates_in_60s'
                        assert 'accepted' not in blocked or nearby(kept, event, M)
                    if nearby(kept, event, M):
                        reason = 'one_to_one_competition'
                n = nearest(kept, event)
                err = basic_error('rule_' + mode, fold['name'], name, 'FN', event, reason)
                err.update(panel_row=r, flags=flags, observed_add_count=float(f['add_count'][r]) if np.isfinite(f['add_count'][r]) else None,
                           observed_minutes_last_action=float(f['minutes_last_action'][r]) if np.isfinite(f['minutes_last_action'][r]) else None,
                           observed_mark_return=float(f['mark_return'][r]) if np.isfinite(f['mark_return'][r]) else None,
                           final_signals_in_5m=len(nearby(kept, event, 5 * M)),
                           nearest_same_action_delta_ms=n['time_ms'] - event['time_ms'] if n else None)
                if mode == 'single_step' and event['action'] != 'reduce':
                    err['candidate_gate_trace'] = candidate_reasons
                errors.append(err)
            for i, pred in enumerate(kept):
                if i in pm:
                    continue
                r = pred['panel_row']
                state_mismatch = int(data['direction'][r]) != (0 if pred['action'] == 'open' else -1)
                n = nearest(target, pred)
                reason = 'observed_position_state_mismatch' if state_mismatch else (
                    'no_scored_event_same_symbol_action' if n is None else
                    'one_to_one_competition' if abs(n['time_ms'] - pred['time_ms']) <= M else
                    'timing_60s_to_5m' if abs(n['time_ms'] - pred['time_ms']) <= 5 * M else 'no_event_within_5m')
                err = basic_error('rule_' + mode, fold['name'], name, 'FP', pred, reason)
                err.update(observed_direction=int(data['direction'][r]),
                           nearest_same_action_delta_ms=n['time_ms'] - pred['time_ms'] if n else None)
                errors.append(err)
            relevant = [e for e in errors if e['scope'] == 'rule_' + mode and e['fold'] == fold['name']]
            summaries.append(dict(fold=fold['name'], mode=mode, config=name, score_60s=score,
                                  score_5m=match_events(kept, target, 5 * M),
                                  false_negative_reasons=dict(Counter(e['reason'] for e in relevant if e['kind'] == 'FN')),
                                  false_positive_reasons=dict(Counter(e['reason'] for e in relevant if e['kind'] == 'FP')),
                                  overlapping_event_flags=dict(Counter(flag for e in relevant for flag in e.get('flags', []))),
                                  saved_metrics_reproduced=True))
            print('TRACED_RULE', fold['name'], mode, flush=True)
    return summaries, len(traces)


def coverage(data, meta, events, all_events, cycles, protocol, exclusions):
    counts = []
    for fold in protocol['folds']:
        lo, hi = timestamp(fold['train_end']), timestamp(fold['eval_end'])
        test, purged = scoring_mask(data, meta, cycles, lo, hi)
        counter = Counter()
        for e in all_events:
            if e['direction'] != 'SHORT' or not lo <= e['order_time_ms'] < hi:
                continue
            if e['classification'] != 'history_anchored':
                reason = 'unanchored_position_state'
            elif e['source_order_type'] == 'LIMIT':
                reason = 'limit_creation_outside_market_score'
            else:
                enriched = next(row for row in events if row['event_id'] == e['event_id'])
                r = enriched['creation_row']
                if r < 0:
                    reason = 'outside_panel'
                elif not data['label_mask'][r]:
                    reason = 'ambiguous_state_or_features_label_excluded'
                elif not test[r]:
                    reason = 'boundary_cycle_purged'
                else:
                    reason = 'scored_market_event'
            counter[(e['action'], reason)] += 1
            if reason != 'scored_market_event':
                exclusions.append(dict(fold=fold['name'], event_id=e['event_id'], symbol=e['symbol'],
                                       action=e['action'], order_type=e['source_order_type'],
                                       creation_time_utc=utc(e['order_time_ms']), fill_time_utc=utc(e['time_ms']),
                                       reason=reason))
        expected = actual_events(data, events, test)
        assert sum(v for (a, reason), v in counter.items() if reason == 'scored_market_event') == len(expected)
        counts.append(dict(fold=fold['name'], counts=[dict(action=a, reason=r, count=n) for (a, r), n in sorted(counter.items())],
                           purged_cycle_ids=purged))
    complete = {c['cycle_id'] for c in cycles if c['status'] == 'complete' and c['direction'] == 'SHORT'}
    rows = [e for e in all_events if e['cycle_id'] in complete]
    by_cycle = defaultdict(list)
    for e in rows:
        by_cycle[e['cycle_id']].append(e)
    short_gaps, third_add, adds, ratios = [], [], [], []
    for cycle_rows in by_cycle.values():
        ordered = sorted(cycle_rows, key=lambda e: (e['time_ms'], e['source_index'], e['event_id']))
        add_count = 0
        for j, e in enumerate(ordered):
            if e['action'] != 'add':
                continue
            adds.append(e)
            add_count += 1
            if e['time_ms'] - ordered[j - 1]['time_ms'] < 30 * M:
                short_gaps.append(e['event_id'])
            if add_count > 2:
                third_add.append(e['event_id'])
            ratios.append(float(e['qty']) / float(ordered[0]['qty']))
    return dict(folds=counts, complete_short_cycles=len(complete),
                action_order_type_counts=[dict(action=a, order_type=t, count=n) for (a, t), n in sorted(Counter((e['action'], e['source_order_type']) for e in rows).items())],
                observed_adds=dict(total=len(adds), previous_fill_gap_below_30m=len(short_gaps),
                                   third_or_later_add=len(third_add), overlapping_union=len(set(short_gaps) | set(third_add)),
                                   short_gap_event_ids=short_gaps, third_or_later_event_ids=third_add,
                                   add_quantity_to_first_quantity_quantiles={str(q): float(np.quantile(ratios, q)) for q in [0, .25, .5, .75, 1]},
                                   quantity_exactly_equal_to_first=int(sum(r == 1 for r in ratios))))


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def table(headers, rows):
    return '<div style="overflow-x:auto"><table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td style="white-space:nowrap">' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


REASONS = {
    'below_threshold_in_60s_window': '±60秒内分数均未超过阈值', 'strict_threshold_tie': '分数等于阈值，被严格大于条件排除',
    'cooldown_suppressed_all_near_signals': '附近高分信号全部被冷却抑制', 'one_to_one_competition': '一对一匹配已被其他事件占用',
    'no_scored_event_same_symbol_action': '本段无同币种同动作评分事件', 'timing_60s_to_5m': '时间相差60秒至5分钟',
    'no_event_within_5m': '附近5分钟无同币种同动作事件', 'reduce_action_not_implemented': '规则没有减仓动作',
    'no_autonomous_action_in_60s': '自主持仓路径在附近未产生该动作', 'no_adverse_crossing_in_60s': '附近未发生浮亏阈值穿越',
    'add_count_limit_in_60s': '附近阈值穿越被加仓次数上限拦截', 'cooldown_in_60s': '附近阈值穿越被30分钟冷却拦截',
    'close_action_priority': '平仓优先覆盖加仓',
    'candidate_label_excluded': '候选分钟不满足标签口径', 'candidate_next_open_missing': '缺少下一分钟开盘价',
    'observed_fill_cooldown_in_60s': '最近一次真实成交触发冷却',
    'prior_prediction_cooldown_in_60s': '之前一次规则预测触发冷却',
    'different_candidate_gates_in_60s': '附近候选被不同门槛拦截，详见逐行记录',
    'no_entry_crossing_in_60s': '附近未穿越入场阈值', 'no_profit_crossing_in_60s': '附近未穿越盈利退出阈值',
    'observed_position_state_mismatch': '自主规则与带单员持仓状态不同',
    'unanchored_position_state': '缺少完整持仓锚点', 'limit_creation_outside_market_score': '限价下单未纳入市价动作评分',
    'ambiguous_state_or_features_label_excluded': '同分钟动作、状态或特征不满足标签口径',
    'boundary_cycle_purged': '跨训练评估边界的周期被剔除', 'outside_panel': '超出分钟面板',
    'scored_market_event': '已纳入评分的市价动作',
}


def build_report(result, errors):
    models, rules, cov = result['models'], result['rules'], result['coverage']
    opens = [m for m in models if m['action'] == 'open']
    open_reasons = Counter()
    for m in opens:
        open_reasons.update(m['false_negative_reasons'])
    tree = next(m for m in models if m['fold'] == 'fold_3' and m['action'] == 'add')
    mode_names = {'single_step': '真实持仓单步', 'autonomous': '自主连续回放'}
    action_names = {'open': '开仓', 'add': '加仓', 'reduce': '减仓', 'close': '平仓'}
    body = '<h1>第一轮失败诊断</h1><p class="lead">检查已有候选从分数到动作的全过程，定位下一轮需要解决的问题。</p>'
    body += '<div class="cards">' + ''.join('<div class="card"><b>' + str(n) + '</b><span>' + label + '</span></div>' for n, label in [
        (sum(m['evaluation']['actual_events'] for m in opens), '评估段开仓事件'),
        (sum(m['evaluation']['final_signals'] for m in opens), '模型最终开仓信号'),
        (sum(m['evaluation']['after_cooldown_60s']['tp'] for m in opens), '开仓命中'), (119, '限价平仓 / 148 次完整空头平仓')]) + '</div>'
    body += '<h2>主要结论</h2><ol>'
    body += '<li>开仓三个评估段共90次真实事件、12个最终信号、0次命中。漏报归因：' + '；'.join(REASONS[k] + ' ' + str(n) + ' 次' for k, n in open_reasons.items()) + '。高分排序有改善，但这些改善没有转化为当前阈值下的动作。</li>'
    body += '<li>第三段加仓树的训练阈值等于最高训练分数。严格“大于阈值”使训练和评估均没有信号；评估中与阈值相等的分钟有 ' + str(tree['evaluation']['equal_threshold']) + ' 个。这是离散分数与报警额度规则的冲突。</li>'
    adds = cov['observed_adds']
    body += f'<li>完整空头周期的 {adds["total"]} 次加仓中，{adds["previous_fill_gap_below_30m"]} 次间隔不足30分钟，{adds["third_or_later_add"]} 次是第三次及以后加仓；两类合并为 {adds["overlapping_union"]} 个不同事件。这里是表达范围核对，不能与评估段漏报直接相加。</li>'
    body += '<li>148次完整空头平仓中119次为限价，44次减仓中20次为限价。规则不生成减仓，市价评分还排除了限价下单。需要分别研究下单与成交。</li></ol>'
    body += '<h2>模型从分数到动作</h2><p>按原种子、负样本、训练段和已选配置复算12个模型；阈值、原始预测、AP与事件计数均核对。没有重新选择配置或调整阈值。±5分钟仅用于原协议中的时间敏感性诊断。</p>'
    body += table(['评估段 / 动作', '模型', '真实事件', '训练额度 / 高分分钟 / 最终信号', '评估高分分钟 → 冷却移除 → 信号', '60秒 TP / FP / FN', '5分钟 TP', '自然事件率 / AP'], [
        [m['fold'] + ' / ' + action_names[m['action']], m['config'], m['evaluation']['actual_events'],
         f'{m["training_alert_quota"]} / {m["train"]["above_threshold"]} / {m["train"]["final_signals"]}',
         f'{m["evaluation"]["above_threshold"]} → {m["evaluation"]["removed_by_cooldown"]} → {m["evaluation"]["final_signals"]}',
         ' / '.join(str(m['evaluation']['after_cooldown_60s'][k]) for k in ['tp', 'fp', 'fn']),
         m['evaluation']['after_cooldown_5m']['tp'], f'{m["natural_event_rate"]:.6f} / {m["evaluation_ap"]:.6f}'] for m in models])
    body += '<p>报警额度按“高分分钟数”设置，连续片段会消耗额度；冷却后剩余的是“动作数”。分数来自负样本抽样与类别平衡训练，不能读作自然频率下的真实概率。AP是在全评估机会中计算的排序指标；事件命中还要求同币种、同动作和准确时间。</p>'
    body += '<h2>误报与漏报归因</h2><p>每个错误只有一个主要归因；加仓次数、间隔等附加标记可以重叠。阈值归因检查事件±60秒内全部可评估分钟，冷却归因要求该窗口确实存在被过滤的高分候选。</p>'
    model_counter = Counter((e['kind'], e['reason']) for e in errors if e['scope'] == 'model')
    body += table(['类型', '主要归因', '事件数'], [[kind, REASONS[reason], n] for (kind, reason), n in sorted(model_counter.items())])
    body += '<h2>完整规则的覆盖与持仓偏离</h2>'
    body += table(['评估段', '运行方式', '已选配置', '60秒 TP / FP / FN', '5分钟 TP', '自主状态不一致误报'], [
        [r['fold'], mode_names[r['mode']], r['config'], ' / '.join(str(r['score_60s'][k]) for k in ['tp', 'fp', 'fn']),
         r['score_5m']['tp'], r['false_positive_reasons'].get('observed_position_state_mismatch', 0)] for r in rules])
    rule_counter = Counter((e['scope'], e['kind'], e['reason']) for e in errors if e['scope'].startswith('rule_'))
    body += table(['运行方式', '类型', '主要归因', '次数'], [[mode_names[scope.removeprefix('rule_')], kind, REASONS[reason], n] for (scope, kind, reason), n in sorted(rule_counter.items())])
    body += '<p>自主回放的漏报只报告动作路径缺失；真实持仓特征不能直接证明自主状态中的具体门槛。自主规则与带单员状态不同的误报已单列。误报只表示未匹配评分事件，不表示该笔模拟交易一定亏损。</p>'
    body += '<h2>哪些操作没进入评分</h2><p>限价单、缺少持仓锚点及边界剔除单列为评价范围限制，不计入已评分事件的漏报。</p>'
    body += table(['评估段', '动作', '范围或排除原因', '次数'], [[f['fold'], action_names[c['action']], REASONS[c['reason']], c['count']] for f in cov['folds'] for c in f['counts']])
    body += '<h2>下一轮的具体输入</h2><ol><li>入场模块：在新版本训练段确定报警频率，并在冷却之后核对动作额度；保留强基准及时间频率对照。先检查高分片段和真实入场分数的差距。</li><li>加仓模块：先估计实际间隔、加仓次数及相对首次数量；区分阈值穿越、阈值状态和分层价位。离散分数需要训练段确定明确的并列处理方式。</li><li>减仓与限价模块：分开识别挂单创建和成交事件，补充取消、未成交及修改订单的生命周期。当前已成功成交的限价记录只能做条件研究。</li><li>模块有同口径优势后再组合自主规则；冻结新候选及T0后留存新观察。第一轮数据仍用于探索。</li></ol>'
    body += '<h2>复核与文件</h2><p>全部12个已选模型与6个规则评估复现原有预测/计数，错误表中的FP、FN逐项加总一致。第一轮来源文件前后摘要一致。原54次配置执行与50次主比较预算记录保留。</p><p><a href="diagnosis.json">完整诊断 JSON</a> · <a href="event_errors.csv">逐事件误报漏报表</a> · <a href="model_signal_trace.csv">高分候选与冷却流水</a> · <a href="score_exclusions.csv">评分范围排除表</a> · <a href="selected_model_scores.npz">已选模型全评估分数</a></p>'
    body += '<p class="muted">范围：第一轮失败诊断。重拟合仅复现已选配置；未新增候选搜索、收益模拟或正式验证。时间均为UTC。</p>'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>第一轮失败诊断</title><style>body{font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#253247;background:#f6f8fb;margin:0}main{max-width:1200px;margin:auto;padding:36px 24px}h1{font-size:32px;margin-bottom:8px}h2{margin-top:36px;font-size:23px}.lead,.muted{color:#68778b}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:white;border:1px solid #dce3ec;border-radius:12px;padding:18px}.card b{display:block;font-size:28px;color:#2464bb}.card span{font-size:14px}table{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:16px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #e3e8ef;overflow-wrap:anywhere}th{background:#eaf0f8}a{color:#2464bb}li{margin:10px 0}@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:20px 12px}table{font-size:12px}th,td{padding:5px}}</style><main>' + body + '</main></html>'


def main():
    DEST.mkdir(exist_ok=True)
    source_paths = sorted(p for p in OUT.iterdir() if p.is_file()) + sorted((ROOT / 'analysis').glob('*.py'))
    source_paths = [p for p in source_paths if p.name != Path(__file__).name]
    before = hashes(source_paths)
    with np.load(OUT / 'reverse_panel.npz', allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    meta, events = read('reverse_panel_meta.json'), read('reverse_event_features.json')['events']
    all_events, cycles, protocol = read('leader_events.json'), read('leader_cycles.json'), read('reverse_protocol.json')
    models, rules, paths = read('reverse_models.json'), read('reverse_rules.json'), read('reverse_rule_predictions.json')
    errors, exclusions, score_archive, signal_traces = [], [], {}, []
    model_summary = diagnose_models(data, meta, events, cycles, protocol, models, errors, score_archive, signal_traces)
    rule_summary, trace_count = diagnose_rules(data, meta, events, cycles, protocol, rules, paths, errors)
    cov = coverage(data, meta, events, all_events, cycles, protocol, exclusions)
    for m in model_summary:
        for kind, count_key in [('FP', 'fp'), ('FN', 'fn')]:
            assert sum(e['scope'] == 'model' and e['fold'] == m['fold'] and e['action'] == m['action'] and e['kind'] == kind for e in errors) == m['evaluation']['after_cooldown_60s'][count_key]
    for r in rule_summary:
        for kind, count_key in [('FP', 'fp'), ('FN', 'fn')]:
            assert sum(e['scope'] == 'rule_' + r['mode'] and e['fold'] == r['fold'] and e['kind'] == kind for e in errors) == r['score_60s'][count_key]
    after = hashes(source_paths)
    assert before == after, 'First-round source files changed during diagnosis'
    result = dict(version='reverse-failure-diagnosis-1', status='completed',
                  scope='diagnosis of saved selected configurations; no new candidate search',
                  matching='original one-to-one symbol/direction/action +/-60s; +/-5m sensitivity only',
                  model_refits='12 existing fold/action selections, deterministic reproduction only',
                  attribution='mutually exclusive main reason per error; observed event flags may overlap',
                  models=model_summary, rules=rule_summary, coverage=cov,
                  preserved_candidate_budget=models['candidate_budget'],
                  verification=dict(model_predictions_and_metrics_reproduced=12, rule_score_sets_reproduced=6,
                                    selected_single_step_paths_reproduced=trace_count, all_error_totals_reconciled=True,
                                    source_files_unchanged=True, source_sha256=before),
                  error_rows=len(errors), exclusion_rows=len(exclusions), high_score_signal_rows=len(signal_traces))
    (DEST / 'diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    write_csv(DEST / 'event_errors.csv', errors)
    write_csv(DEST / 'model_signal_trace.csv', signal_traces)
    write_csv(DEST / 'score_exclusions.csv', exclusions)
    np.savez_compressed(DEST / 'selected_model_scores.npz', **score_archive)
    (DEST / 'report.html').write_text(build_report(result, errors))
    print(json.dumps(dict(status=result['status'], error_rows=len(errors), coverage=cov['observed_adds'],
                          report=str(DEST / 'report.html')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
