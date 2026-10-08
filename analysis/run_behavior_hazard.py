"""Frozen v4 pilot; execute with --project-root and --output-dir if staged.

Preserves every old script/result. Reused history is exploratory throughout.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit

from behavior_hazard import (ACTIONS, MARKET, POSITION, M, Forecast, Position, SizeModel,
                             OnlineDecoder, causal_match, path_card, offset_calibration,
                             probability_card, stable_key)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_days(data, rows, y, p, baseline, seed=20261003):
    """Resample UTC day blocks shared across symbols, never IID minutes."""
    y = np.asarray(y, int)
    loss = lambda q: -(y*np.log(np.clip(q, 1e-12, 1-1e-12))+(1-y)*np.log1p(-np.clip(q, 1e-12, 1-1e-12)))
    delta = loss(baseline)-loss(p)
    day = data['time_ms'][rows]//(1440*M)
    days = np.unique(day)
    sums = np.array([delta[day == d].sum() for d in days])
    counts = np.array([(day == d).sum() for d in days])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(days), size=(1000, len(days)))
    samples = sums[picks].sum(axis=1)/counts[picks].sum(axis=1)
    return dict(utc_days=len(days), mean_loss_improvement=float(delta.mean()),
                exploratory_day_block_95_interval=np.quantile(samples, [.025, .975]).tolist(),
                independent_minute_resampling=False)


def quantity_samples(events, mask, data, meta):
    f = {n: data['X'][:, i] for i, n in enumerate(meta['feature_names'])}
    samples = defaultdict(list)
    for e in events:
        r, action = e['panel_row'], e['action']
        if not mask[r] or action == 'close':
            continue
        qty = float(e['qty'])
        if action == 'open':
            value = qty*float(f['close'][r])
        elif action == 'add':
            value = qty/float(f['first_qty'][r])
        else:
            value = qty/abs(float(f['position'][r]))
        samples[action].append(value)
    return dict(samples)


def quantity_card(events, mask, data, meta, size):
    samples = quantity_samples(events, mask, data, meta)
    out = {}
    for action, values in samples.items():
        a = np.asarray(values)
        d = size.distribution[action]
        out[action] = dict(events=len(a), mean_absolute_target_error=float(np.abs(a-d['median']).mean()),
                           median_absolute_target_error=float(np.median(np.abs(a-d['median']))),
                           relative_total_absolute_error=float(np.abs(a-d['median']).sum()/a.sum()),
                           q25_q75_empirical_coverage=float(((a >= d['q25']) & (a <= d['q75'])).mean()),
                           includes_all_eligible_true_events=True)
    out['close'] = dict(target='remaining own quantity; exact by construction, not a learned sizing result')
    return out


def risk_diagnostic(labels, held, state):
    loss = -state[:, 3]
    bins = [-1e30, 0, .05, .12, .15, .30, 1e30]
    out = []
    for stage in (0, 1, 2):
        sm = (state[:, 2] == stage) if stage < 2 else (state[:, 2] >= 2)
        for low, high in zip(bins[:-1], bins[1:]):
            at = held & sm & (loss >= low) & (loss < high)
            n = int(at.sum())
            k = int((labels[at] == 2).sum())
            out.append(dict(prior_add_stage=str(stage) if stage < 2 else '2+',
                            adverse_low=None if low < -1e20 else low,
                            adverse_high=None if high > 1e20 else high,
                            risk_minutes=n, add_events=k, event_rate=k/n if n else None))
    return out


def old_score_diagnostic(root, data, meta, labels, folds, allowed):
    """No new fit and no new threshold search on v3 stored rank scores."""
    archive = np.load(root/'analysis/results/observed_behavior_v3/scores.npz', allow_pickle=False)
    results = []
    for fold in folds:
        for action_index, action in enumerate(ACTIONS, 1):
            group = fold['name']+'_'+action
            cr, er = archive[group+'_cal_rows'], archive[group+'_eval_rows']
            yc, ye = labels[cr] == action_index, labels[er] == action_index
            for variant in ('base', 'extended'):
                cp, ep = archive[group+'_'+variant+'_cal_scores'], archive[group+'_'+variant+'_eval_scores']
                shift = offset_calibration(cp, yc)
                calibrated = expit(logit(np.clip(ep, 1e-12, 1-1e-12))+shift)
                constant = np.full(len(er), (yc.sum()+.5)/(len(yc)+1))
                results.append(dict(fold=fold['name'], action=action, variant=variant,
                                    offset=shift, evaluation=probability_card(ye, calibrated),
                                    calibration_frequency_baseline=probability_card(ye, constant),
                                    uncertainty=bootstrap_days(data, er, ye, calibrated, constant)))
    # Oracle timing only: at each actual opening minute compare coins that were
    # flat and had finite v3 scores. Missing target/candidate rows are reported.
    oracle = []
    for fold in folds:
        group = fold['name']+'_open'
        er = archive[group+'_eval_rows']
        gold = er[labels[er] == 1]
        fit_start, fit_end = fold['fit_start_ms'], fold['calibration_start_ms']
        frequency = Counter(int(data['symbol_idx'][r]) for r in np.flatnonzero(
            (labels == 1) & (data['time_ms'] >= fit_start) & (data['time_ms'] < fit_end)))
        dayrows = defaultdict(list)
        for i, r in enumerate(er):
            dayrows[int(data['time_ms'][r])].append(i)
        for variant in ('base', 'extended', 'historical_symbol_frequency'):
            score = (archive[group+'_'+variant+'_eval_scores'] if variant != 'historical_symbol_frequency'
                     else np.array([frequency[int(data['symbol_idx'][r])] for r in er]))
            ranks, candidate_counts = [], []
            for target in gold:
                ixs = dayrows[int(data['time_ms'][target])]
                ordered = sorted(ixs, key=lambda i: (-score[i], stable_key(meta['symbols'][int(data['symbol_idx'][er[i]])],
                                                                         'open', int(data['time_ms'][target]))))
                target_i = int(np.searchsorted(er, target))
                ranks.append(ordered.index(target_i)+1)
                candidate_counts.append(len(ordered))
            oracle.append(dict(fold=fold['name'], variant=variant, events=len(ranks),
                               top1=sum(r <= 1 for r in ranks), top3=sum(r <= 3 for r in ranks),
                               median_rank=float(np.median(ranks)),
                               candidate_count_min=min(candidate_counts), candidate_count_max=max(candidate_counts),
                               timing_oracle=True, deployable_strategy=False))
    return dict(probabilities=results, known_open_timing_coin_rank=oracle,
                reused_v3_eligible_rows=True, calibration_changes_probability_not_ranking=True)


def thresholds_and_budgets(forecast, market, state, labels, flat, held, fit_mask, cal_mask, fit_days):
    cr = np.flatnonzero(cal_mask & (flat | held))
    probability = forecast.predict(market[cr], state[cr], held[cr])
    thresholds, budgets = [], []
    for j in range(4):
        risk = flat if j == 0 else held
        rate = int(((labels == j+1) & fit_mask & risk).sum())/int((fit_mask & risk).sum())
        values = probability[risk[cr], j]
        thresholds.append(float(np.quantile(values, 1-rate)))
        budgets.append(int(np.ceil(2*int(((labels == j+1) & fit_mask & risk).sum())/fit_days)))
    return thresholds, budgets


def local_predictions(data, meta, rows, probability, held, size, thresholds, budgets):
    decoder = OnlineDecoder(thresholds, budgets)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[int(data['time_ms'][r])].append(i)
    f = {n: data['X'][:, i] for i, n in enumerate(meta['feature_names'])}
    pending, pp = {}, []
    for now in sorted(groups):
        pending = {s: due for s, due in pending.items() if due > now}
        ix = groups[now]
        symbols = [meta['symbols'][int(data['symbol_idx'][rows[i]])] for i in ix]
        for at, action in decoder.choose(now, symbols, probability[ix], held[rows[ix]], pending):
            r, symbol = int(rows[ix[at]]), symbols[at]
            position = Position(qty=abs(float(f['position'][r])) if held[r] else 0.,
                                first_qty=float(f['first_qty'][r]) if held[r] else 0.)
            qty = size.quantity(action, position, float(f['close'][r]))
            pp.append(dict(symbol=symbol, direction='SHORT', action=action, issue_ms=now,
                           time_ms=now+M, quantity=qty, panel_row=r, forecast_state='observed_prior'))
            pending[symbol] = now+M
    return pp, dict(decoder.rejections)


def autonomous(data, meta, forecast, size, thresholds, budgets, start, end, scoring,
               extra_delay=0, initial_positions=None, injected_drop_action=None):
    """Truth arrays are read ONLY by the scorer after policy execution.

    Decisions depend on cached public market columns, own ledger, online quota.
    Execution price is accessed only when the pending virtual event is due.
    """
    n = len(data['timeline_ms'])
    symbols = meta['symbols']
    states = initial_positions or [Position() for _ in symbols]
    if len(states) != len(symbols):
        raise ValueError('initial state count')
    feature_indices = [meta['feature_names'].index(name) for name in MARKET]
    close_index = meta['feature_names'].index('close')
    position_index = meta['feature_names'].index('position')
    decoder, pending = OnlineDecoder(thresholds, budgets), {}
    pp, issued, own_qty, rows_scored, execution_failures = [], [], [], [], Counter()
    turnover = 0.
    dropped = False
    for minute in np.flatnonzero((data['timeline_ms'] >= start) & (data['timeline_ms'] < end)):
        now = int(data['timeline_ms'][minute])
        r = np.arange(len(symbols))*n+int(minute)
        marks = data['X'][r, close_index]
        selected = scoring[r] & np.isfinite(marks)
        rows_scored.extend(r[selected].tolist())
        own_qty.extend(states[si].qty for si in np.flatnonzero(selected))
        market = data['X'][r][:, feature_indices]
        state = np.vstack([s.features(now, mark) for s, mark in zip(states, marks)])
        held = np.array([s.qty > 0 for s in states])
        valid = np.isfinite(market).all(axis=1) & np.isfinite(marks) & (marks > 0)
        valid &= (~held | np.isfinite(state).all(axis=1))
        ix = np.flatnonzero(valid)
        probabilities = forecast.predict(market[ix], state[ix], held[ix]) if len(ix) else np.empty((0,4))
        current = [symbols[si] for si in ix]
        pending_symbols = {symbols[si] for si in pending}
        for at, action in decoder.choose(now, current, probabilities, held[ix], pending_symbols):
            si = int(ix[at])
            qty = size.quantity(action, states[si], float(marks[si]))
            event = dict(symbol=symbols[si], direction='SHORT', action=action, issue_ms=now,
                         time_ms=now+(1+extra_delay)*M, quantity=qty, panel_row=int(r[si]),
                         forecast_state='own_ledger')
            pending[si] = event
            issued.append(event)
        for si, event in list(pending.items()):
            if event['time_ms'] > now:
                continue
            execution_row = si*n+int(event['panel_row'] % n)+extra_delay
            # This future-indexed value is consumed at execution, never used
            # to decide whether a signal was eligible in the past.
            price = float(data['next_open'][execution_row]) if execution_row < (si+1)*n else np.nan
            if np.isfinite(price) and price > 0:
                if injected_drop_action == event['action'] and not dropped:
                    dropped = True
                    execution_failures['injected_drop_'+event['action']] += 1
                else:
                    states[si].apply(event['action'], event['quantity'], price, now)
                    pp.append({**event, 'execution_price': price})
                    turnover += event['quantity']*price
            else:
                execution_failures['missing_execution_price'] += 1
            del pending[si]
    rows_scored = np.asarray(rows_scored, int)
    marks = data['X'][rows_scored, close_index]
    truth = np.nan_to_num(np.abs(data['X'][rows_scored, position_index]), nan=0.)
    scale = size.distribution['open']['median']
    path = path_card(np.asarray(own_qty), truth, marks, scale)
    path.update(initial_state='flat once at evaluation start; never daily-reset',
                scoring_known_observed_flat_or_short_minutes_only=True,
                residual_open_symbols=sum(s.qty > 0 for s in states),
                residual_quote_exposure=float(sum(s.qty*float(data['X'][si*n+minute, close_index])
                                                         for si, s in enumerate(states) if s.qty > 0)),
                turnover_quote_proxy=turnover,
                fee_stress_quote={str(bps)+'bps':turnover*bps/10000 for bps in (0, 5, 10)},
                fees_are_scenarios_not_account_costs=True, funding_and_return_not_validated=True)
    return dict(predictions=pp, issued=issued, path=path, rejections=dict(decoder.rejections),
                execution_failures=dict(execution_failures), pending_at_window_end=len(pending),
                delay_extra_minutes=extra_delay)


def report(result):
    esc = lambda value: str(value).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    rows = []
    for fold in result['fold_results']:
        for trial in fold['variants']:
            prob = trial['probability']['pooled']
            main = trial['autonomous'][0]
            rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in
                [fold['fold'], trial['variant'], f"{prob['log_loss']:.6g}",
                 f"{prob['expected_events']:.1f}/{prob['events']}",
                 '/'.join(str(main['all_events'][k]) for k in ('tp','fp','fn')),
                 f"{main['path']['normalized_absolute_error']:.3f}",
                 '通过' if trial.get('progression_passed') else '未通过'])+'</tr>')
    body = '<h1>策略逆向建模：受限第四轮</h1><p>历史探索 · 固定33币SHORT · 成功更新/成交代理</p>'
    body += '<p><b>'+esc(result['decision']['summary'])+'</b></p>'
    body += '<p>模型仅比较状态/时钟与增加原有8项公开行情的分层事件概率；数量使用拟合期中位数与区间。所有旧数据、脚本与结果保留。既有历史反复查看，以下不属于未见时间验收。</p>'
    body += '<table><tr><th>时间段</th><th>结构</th><th>自然频率log loss</th><th>预测概率质量/真实事件</th><th>自主TP/FP/FN</th><th>路径误差</th><th>推进</th></tr>'+''.join(rows)+'</table>'
    body += '<p>事件严格禁止匹配早于预测发行的真实事件。±60秒为主、±5分钟为辅助。自主决策只读公开行情与自己账本，真实持仓仅供评分；路径评分含双方状态不一致分钟。首笔数量尺度只取拟合期，不用测试首笔重缩放。</p>'
    body += '<p>复用了完整周期与边界剔除队列；未平周期/未知初始状态尚未扩展，因此不能声称覆盖全部交易者记录。普通LIMIT未等同Maker；没有挂单价、取消或排队模型，不阻塞成功事件代理。0/1/5分钟额外执行延迟为敏感性，不能冒充真实记录可见性延迟已验证。</p>'
    body += '<p>数量误差覆盖全部合格真实事件；收益、账户余额、杠杆、资金费率均未验收。0/5/10bps费用只作用于虚拟周转金额，是情景参数，非账户实际成本。</p>'
    body += '<p><a href="protocol.json">冻结协议</a> · <a href="results.json">完整结果</a> · <a href="diagnostics.json">旧分数与风险集诊断</a> · <a href="verification.json">核验</a></p>'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>策略逆向建模第四轮</title><style>body{font:16px/1.7 -apple-system,"PingFang SC",sans-serif;background:#f5f7fa;color:#233247;max-width:1250px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#2458a0}</style>'+body+'</html>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir or root/'analysis/results/behavior_hazard_v4'
    protocol_path = out/'protocol.json'
    protocol = read(protocol_path)
    protocol_hash = digest(protocol_path)
    protected = read(out/'source_hashes.json')
    for name, expected in protected.items():
        assert digest(root/name) == expected, name
    sys.path.insert(0, str(root/'analysis'))
    from research_observed_behavior import fill_labels, interval_mask
    from run_reverse_rules import timestamp
    data = dict(np.load(root/'analysis/results/reverse_panel.npz', allow_pickle=False))
    meta = read(root/'analysis/results/reverse_panel_meta.json')
    events, cycles = read(root/'analysis/results/leader_events.json'), read(root/'analysis/results/leader_cycles.json')
    labels, allowed, actual, exclusions, long_counts = fill_labels(data, meta, events)
    sources = {e['event_id']:e for e in events}
    actual = [{**e, 'qty':sources[e['event_id']]['qty']} for e in actual]
    market = data['X'][:, [meta['feature_names'].index(n) for n in MARKET]]
    state = data['X'][:, [meta['feature_names'].index(n) for n in POSITION]]
    valid_market = np.isfinite(market).all(axis=1)
    flat = allowed & (data['direction'] == 0) & valid_market
    held = allowed & (data['direction'] == -1) & valid_market & np.isfinite(state).all(axis=1)
    diagnostic_folds = []
    for f in protocol['folds']:
        diagnostic_folds.append({**f, 'fit_start_ms':timestamp(f['train_start']),
                                 'calibration_start_ms':timestamp(f['calibration_start'])})
    diagnostics = old_score_diagnostic(root, data, meta, labels, diagnostic_folds, allowed)
    diagnostics.update(add_risk_set=risk_diagnostic(labels, held, state),
                       source_exclusions=exclusions, source_long_counts_separate=long_counts,
                       negative_semantics='no saved successful event, not observed private hold intention')
    write(out/'diagnostics.json', diagnostics)
    fold_results, prediction_archive, model_archive = [], {}, {}
    for fold in protocol['folds']:
        lo, mid, hi, end = [timestamp(fold[k]) for k in ('train_start','calibration_start','train_end','eval_end')]
        masks, purges = {}, {}
        for part, start, stop in [('fit',lo,mid),('calibration',mid,hi),('evaluation',hi,end)]:
            masks[part], purges[part] = interval_mask(data, meta, cycles, start, stop)
        fit_mask, cal_mask, ev_mask = [masks[k] for k in ('fit','calibration','evaluation')]
        er = np.flatnonzero(ev_mask & (flat | held))
        fit_events = [e for e in actual if fit_mask[e['panel_row']] and (flat | held)[e['panel_row']]]
        gold = [e for e in actual if ev_mask[e['panel_row']] and (flat | held)[e['panel_row']]]
        size = SizeModel(quantity_samples(actual, fit_mask & (flat | held), data, meta))
        fc = dict(fold=fold['name'], purged_cycle_ids=purges,
                  fit_events=len(fit_events), evaluation_events=len(gold),
                  evaluation_minutes=len(er), quantity_fit=size.distribution,
                  quantity_evaluation=quantity_card(actual, ev_mask & (flat | held), data, meta, size), variants=[])
        probability_by_variant = {}
        for variant in protocol['variants']:
            forecast = Forecast(variant).fit(market, state, labels, fit_mask & flat, fit_mask & held)
            forecast.calibrate(market, state, labels, cal_mask & flat, cal_mask & held)
            thresholds, budgets = thresholds_and_budgets(forecast, market, state, labels, flat, held,
                                                        fit_mask, cal_mask, (mid-lo)/(1440*M))
            probability = forecast.predict(market[er], state[er], held[er])
            probability_by_variant[variant] = probability
            cards = {}
            for j, action in enumerate(ACTIONS):
                at = flat[er] if j == 0 else held[er]
                cards[action] = probability_card(labels[er[at]] == j+1, probability[at, j])
            cards['pooled'] = probability_card(labels[er] > 0, probability.sum(axis=1))
            local, rejected = local_predictions(data, meta, er, probability, held, size, thresholds, budgets)
            pred_key = fold['name']+'_'+variant
            prediction_archive[pred_key+'_local'] = local
            model_archive[pred_key] = forecast.details
            local_events = {a:causal_match([p for p in local if p['action']==a],
                                         [e for e in gold if e['action']==a]) for a in ACTIONS}
            trial = dict(variant=variant, probability=cards, thresholds=thresholds, daily_budgets=budgets,
                         local_events=local_events, local_rejections=rejected, autonomous=[])
            # Policy never sees scoring mask. Known/purged truth cohort applies
            # after decisions to event/path assessment only.
            scoring = ev_mask & allowed & np.isin(data['direction'], [0,-1])
            for delay in protocol['extra_execution_delay_minutes']:
                roll = autonomous(data, meta, forecast, size, thresholds, budgets, hi, end, scoring, delay)
                full = roll.pop('predictions')
                issued = roll.pop('issued')
                pp = [p for p in full if ev_mask[p['panel_row']] and allowed[p['panel_row']] and
                      data['direction'][p['panel_row']] in (0,-1)]
                roll.update(all_events=causal_match(pp, gold), sensitivity_5m=causal_match(pp, gold,5*M),
                            events_by_action={a:causal_match([p for p in pp if p['action']==a],
                                              [e for e in gold if e['action']==a]) for a in ACTIONS},
                            scoring_predictions=len(pp), all_executed_predictions=len(full),
                            issued_virtual_events=len(issued), outside_scoring_executions=len(full)-len(pp),
                            fp_per_33_symbol_calendar_day=causal_match(pp,gold)['fp']/((end-hi)/(1440*M)))
                prediction_archive[pred_key+'_delay_'+str(delay)] = dict(executed=full, issued=issued)
                trial['autonomous'].append(roll)
                print('ROLLOUT',pred_key,'extra_delay',delay,'TP/FP/FN',
                      [roll['all_events'][k] for k in ('tp','fp','fn')],
                      'path',roll['path']['normalized_absolute_error'],flush=True)
            fc['variants'].append(trial)
        base, candidate = fc['variants']
        y = labels[er] > 0
        base_p, candidate_p = [probability_by_variant[v].sum(axis=1) for v in protocol['variants']]
        improvement = 1-candidate['probability']['pooled']['log_loss']/base['probability']['pooled']['log_loss']
        candidate['probability_comparison'] = dict(relative_log_loss_improvement=improvement,
            uncertainty=bootstrap_days(data,er,y,candidate_p,base_p))
        largest = Counter(e['symbol'] for e in gold).most_common(1)[0][0]
        keep = data['symbol_idx'][er] != meta['symbols'].index(largest)
        cb, bb = probability_card(y[keep],candidate_p[keep]), probability_card(y[keep],base_p[keep])
        candidate['leave_largest_symbol_probability'] = dict(symbol=largest, candidate=cb, baseline=bb,
                                                            relative_log_loss_improvement=1-cb['log_loss']/bb['log_loss'])
        bp, cp = base['autonomous'][0]['path'], candidate['autonomous'][0]['path']
        candidate['path_comparison'] = dict(relative_error_improvement=1-cp['normalized_absolute_error']/bp['normalized_absolute_error']
                                           if bp['normalized_absolute_error'] else None)
        card = candidate['probability']['pooled']
        event_ok = all(candidate['autonomous'][0]['events_by_action'][a]['precision'] >= .2 and
                       candidate['autonomous'][0]['events_by_action'][a]['recall'] >= .2 for a in ACTIONS)
        candidate['progression_passed'] = bool(improvement >= .05 and .8 <= card['event_mass_ratio'] <= 1.25 and
            candidate['leave_largest_symbol_probability']['relative_log_loss_improvement'] > 0 and event_ok and
            cp['normalized_absolute_error'] <= .5 and candidate['path_comparison']['relative_error_improvement'] >= .2)
        base['progression_passed'] = False
        fold_results.append(fc)
    for name, expected in protected.items():
        assert digest(root/name) == expected, name
    assert digest(protocol_path) == protocol_hash
    passes = sum(f['variants'][1]['progression_passed'] for f in fold_results)
    result = dict(version=protocol['version'], status='completed_exploratory_pilot',
                  protocol_sha256=protocol_hash, frozen_candidate_budget=protocol['budget'],
                  protected_source_files_unchanged=len(protected), fold_results=fold_results,
                  progression_folds=passes,
                  decision=dict(behavior_approximation_validated=False, independent_validation=False,
                                summary='历史探索未通过推进；停止扩展配置，保留失败结果。' if passes < 2 else
                                '探索结果达到推进条件；仍须未见同源操作记录验收。'),
                  limits=protocol['limits'])
    write(out/'results.json',result)
    write(out/'predictions.json',prediction_archive)
    write(out/'models.json',model_archive)
    (out/'report.html').write_text(report(result))
    print('COMPLETED',passes,'progression folds; originals unchanged',len(protected),flush=True)


if __name__ == '__main__':
    main()
