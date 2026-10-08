"""Training-derived, bounded module comparisons after first-round diagnosis."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from diagnose_reverse_failures import hashes, pairs, table, utc
from reverse_models import MARKET, POSITION, gold
from reverse_metrics import match_events, paired_f1_interval
from run_reverse_rules import scoring_mask, timestamp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/results'
DEST = OUT / 'module_research_v2'
M = 60000
MODULES = {'open': ('btc_relative60', 1), 'add': ('mark_return', -1),
           'reduce': ('mark_return', 1), 'close': ('mark_return', 1)}
LABELS = {'open': 1, 'add': 2, 'reduce': 3, 'close': 4}
NAMES = {'open': '入场', 'add': '加仓', 'reduce': '市价减仓', 'close': '市价平仓'}


def read(name):
    return json.loads((OUT / name).read_text())


def quantiles(values):
    assert len(values)
    return {str(q): float(np.quantile(values, q)) for q in [0, .25, .5, .75, 1]}


def generate(data, meta, indices, score, threshold, crossing, cooldown_minutes, action):
    previous = np.concatenate((score[:1], score[:-1]))
    t, si = data['time_ms'], data['symbol_idx']
    adjacent = np.concatenate(([False], (si[1:] == si[:-1]) & (t[1:] == t[:-1] + M)))
    above = score >= threshold
    trigger = above & adjacent & (previous < threshold) if crossing else above
    rows = indices[trigger[indices]]
    last, result = {}, []
    for r in rows:
        s, now = meta['symbols'][int(si[r])], int(t[r])
        if s in last and now - last[s] < cooldown_minutes * M:
            continue
        result.append(dict(symbol=s, direction='SHORT', action=action, time_ms=now, panel_row=int(r)))
        last[s] = now
    return result


def size_diagnostics(action, train_gold, test_gold, events, data, meta):
    if action not in ['add', 'reduce']:
        return None
    by_id = {e['event_id']: e for e in events}
    opening_qty = {e['cycle_id']: float(e['qty']) for e in events if e['action'] == 'open'}
    def ratio(event):
        e = by_id[event['event_id']]
        denominator = opening_qty[e['cycle_id']] if action == 'add' else abs(float(e['position_before']))
        assert denominator > 0
        return float(e['qty']) / float(denominator)
    train = np.array([ratio(e) for e in train_gold])
    test = np.array([ratio(e) for e in test_gold])
    fitted = float(np.median(train))
    baseline = 1.0 if action == 'add' else .5
    return dict(training_ratios=quantiles(train), fitted_median_ratio=fitted,
                evaluation_events=len(test), evaluation_absolute_error_mean=float(np.abs(test - fitted).mean()),
                fixed_baseline_ratio=baseline, fixed_baseline_absolute_error_mean=float(np.abs(test - baseline).mean()),
                denominator='first opening quantity' if action == 'add' else 'observed position before event',
                scope='local quantity prediction at observed events; separate from timing and autonomous replay')


def limits(data, meta, events, cycles, protocol):
    """Describe successful limits; source has avgPrice, no original limit price."""
    result = []
    for fold in protocol['folds']:
        lo, hi = timestamp(fold['train_start']), timestamp(fold['train_end'])
        selected = []
        for e in events:
            if e['direction'] != 'SHORT' or e['source_order_type'] != 'LIMIT' or e['action'] not in ['reduce', 'close']:
                continue
            r = e['creation_row']
            if r < 0 or not lo <= e['order_time_ms'] < hi or not data['state_known_mask'][r]:
                continue
            if data['direction'][r] != -1 or meta['cycle_ids'][int(data['cycle_idx'][r])] != e['cycle_id']:
                continue
            # scoring_mask uses MARKET label eligibility; successful LIMIT
            # creations are outside that label mask. Apply cycle purge separately.
            c = next(c for c in cycles if c['cycle_id'] == e['cycle_id'])
            if any(c['start_ms'] < boundary <= c['end_ms'] for boundary in [lo, hi]):
                continue
            f = e['features_at_creation_minute']
            if f['avgcost'] is None or f['mark_return'] is None:
                continue
            selected.append(dict(event_id=e['event_id'], action=e['action'], symbol=e['symbol'],
                                 creation_utc=utc(e['order_time_ms']), fill_utc=utc(e['time_ms']),
                                 placement_mark_return=f['mark_return'],
                                 fill_delay_minutes=(e['time_ms'] - e['order_time_ms']) / M,
                                 successful_fill_return_from_creation_cost=1 - float(e['price']) / f['avgcost']))
        assert selected
        result.append(dict(fold=fold['name'], training_successful_orders=len(selected),
                           action_counts=dict(Counter(e['action'] for e in selected)),
                           placement_mark_return=quantiles([e['placement_mark_return'] for e in selected]),
                           fill_delay_minutes=quantiles([e['fill_delay_minutes'] for e in selected]),
                           events=selected))
    raw = json.loads((ROOT / 'data/binance/2026-10-02_2026大黑马/leader_orders.json').read_text())
    keys = sorted({k for e in raw if e['type'] == 'LIMIT' for k in e})
    assert 'avgPrice' in keys and 'price' not in keys
    return dict(status='placement_and_execution_candidate_blocked_by_source_fields', source_fields=keys,
                missing=['original limit price', 'cancellations', 'unfilled orders', 'amendments', 'partial fill lifecycle'],
                filled_average_price_is_an_outcome_not_a_placement_feature=True,
                training_only_descriptive_folds=result)


def report(result):
    rows = result['comparisons']
    body = '<h1>第二轮模块研究</h1><p>根据失败诊断，固定8个简单动作候选：每个模块比较“达到阈值”和“穿越阈值”两种触发方式。阈值采用训练事件的中位数；加减仓与平仓的冷却采用训练事件操作间隔的下四分位数。</p>'
    body += '<p><b>本轮结果：' + result['decision']['description'] + '</b></p>'
    body += table(['评估段 / 模块', '训练选定触发', '训练阈值 / 冷却分钟', 'TP / FP / FN', '候选F1', '原强基准F1', '差值', '5分钟TP'], [
        [r['fold'] + ' / ' + NAMES[r['action']], '阈值穿越' if r['selected_by_training_only'] == 'crossing' else '达到阈值',
         f'{r["threshold_from_training"]:.6f} / {r["cooldown_minutes_from_training"]}',
         ' / '.join(str(r['selected_evaluation'][k]) for k in ['tp', 'fp', 'fn']), f'{r["selected_evaluation"]["f1"]:.4f}',
         f'{r["baseline_evaluation"]["f1"]:.4f}', f'{r["f1_difference"]:+.4f}', r['selected_5m']['tp']] for r in rows])
    body += '<h2>推进门槛</h2><p>保持原0.05的F1改善工作目标，至少两段达到；移除最大事件币种后也须保留两段正改善。评估正事件至少30个、至少15个周期。完整规则还需覆盖入场、加仓、减仓和主要退出方式。</p>'
    body += table(['模块', '达到0.05的段数', '去掉最大币种仍改善的段数', '评估事件 / 周期', '模块判定'], [
        [NAMES[a], g['folds_meeting_005'], g['leave_largest_symbol_positive_folds'], f'{g["evaluation_events"]} / {g["evaluation_cycles"]}',
         '满足局部推进门槛' if g['passed'] else '未满足'] for a, g in result['module_gates'].items()])
    body += '<h2>加减仓数量</h2><p>数量以加仓量/首次开仓量、减仓量/操作前仓位为单位，训练段中位数与固定首次数量、固定减半基准比较；这是已知真实持仓下的局部数量误差。</p>'
    body += table(['评估段 / 动作', '训练中位数比例', '数量平均绝对误差', '固定比例基准误差'], [
        [r['fold'] + ' / ' + NAMES[r['action']], f'{r["size"]["fitted_median_ratio"]:.4f}',
         f'{r["size"]["evaluation_absolute_error_mean"]:.4f}', f'{r["size"]["fixed_baseline_absolute_error_mean"]:.4f}']
        for r in rows if r['size'] is not None])
    body += '<h2>限价挂单与成交</h2><p>源数据只有成功订单的成交均价 avgPrice，没有原始挂单价；也缺少取消、未成交、修改及完整分笔成交记录。不能用成交均价反推出当时挂单规则。下表只描述训练段内可追溯的成功订单。</p>'
    body += table(['训练段', '成功限价减仓 / 平仓', '挂单时持仓收益中位数', '挂单至成交中位分钟 / 最大分钟'], [
        [r['fold'], f'{r["action_counts"].get("reduce",0)} / {r["action_counts"].get("close",0)}',
         f'{r["placement_mark_return"]["0.5"]:.4%}', f'{r["fill_delay_minutes"]["0.5"]:.2f} / {r["fill_delay_minutes"]["1"]:.2f}']
        for r in result['limits']['training_only_descriptive_folds']])
    body += '<h2>候选与反例</h2><p>全部8个候选在三个评估段的正负结果保存在比较文件；以下列出每个选定模块的一条漏报或误报反例，时间为UTC。</p>'
    body += table(['评估段 / 模块', '错误类型', '币种', '时间'], [
        [r['fold'] + ' / ' + NAMES[r['action']], r['counterexample']['kind'], r['counterexample']['symbol'],
         r['counterexample']['time_utc']] for r in rows])
    body += '<h2>执行核对</h2><p>参数、候选数、基准与门槛在本轮评分前保存。基准沿用第一轮各训练段选出的强基准，同币种、同自然频率分钟、同边界剔除、同一对一±60秒匹配；未按评估成绩换基准。第一轮文件摘要前后相同。所有历史仍为探索数据。</p><p><a href="protocol.json">本轮协议</a> · <a href="comparisons.json">全部候选、参数与反例</a> · <a href="predictions.json">全部模块预测</a></p>'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二轮模块研究</title><style>body{font:16px/1.7 -apple-system,"PingFang SC",sans-serif;background:#f6f8fb;color:#253247;margin:0}main{max-width:1200px;margin:auto;padding:36px 24px}h2{margin-top:32px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #e3e8ef;overflow-wrap:anywhere}th{background:#eaf0f8}a{color:#2464bb}</style><main>' + body + '</main></html>'


def main():
    DEST.mkdir(exist_ok=True)
    first = read('reverse_protocol.json')
    # Written before producing any v2 training or evaluation metric.
    protocol = dict(version='reverse-modules-2', prerequisite='failure_diagnosis/diagnosis.json completed',
                    status='registered_before_module_scores', source='same exploratory historical observations',
                    folds=first['folds'], candidate_budget=dict(modules=4, variants_per_module=2, unique_candidates=8,
                                                              baseline_policy='reuse frozen first-round training-selected baselines; no new baseline search'),
                    variants=['threshold_state', 'threshold_crossing'],
                    parameter_estimation='median signed feature at training positive events; non-open cooldown=floor(training p25 minutes since last action), min one minute; open cooldown original 30 minutes',
                    state='single-step observed positions only; no autonomous simulation',
                    selection='highest training event F1, tie by variant name',
                    budget_policy='fold parameter refits are not new configurations; no evaluation-driven threshold or variant additions',
                    quantities='one training median ratio estimate per add/reduce module; descriptive local size comparison, not additional timing configurations',
                    gates=dict(f1_improvement=.05, consistent_folds=2, min_evaluation_events=30, min_evaluation_cycles=15,
                               leave_largest_symbol='positive improvement in at least two folds',
                               combination='all necessary modules and a supported main-exit policy must pass'),
                    limit_policy='describe training successful placements and delays only; no policy using final avgPrice as a decision input')
    (DEST / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    source_paths = sorted(p for p in OUT.iterdir() if p.is_file())
    before = hashes(source_paths)
    with np.load(OUT / 'reverse_panel.npz', allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    meta, events = read('reverse_panel_meta.json'), read('reverse_event_features.json')['events']
    cycles, models = read('leader_cycles.json'), read('reverse_models.json')
    comparisons, paths = [], {}
    for fold, model_fold in zip(first['folds'], models['folds']):
        train, _ = scoring_mask(data, meta, cycles, timestamp(fold['train_start']), timestamp(fold['train_end']))
        test, _ = scoring_mask(data, meta, cycles, timestamp(fold['train_end']), timestamp(fold['eval_end']))
        for action, (feature, sign) in MODULES.items():
            cols = [meta['feature_names'].index(f) for f in MARKET + (POSITION if action != 'open' else [])]
            eligible = data['label_mask'] & (data['direction'] == (0 if action == 'open' else -1)) & np.isfinite(data['X'][:, cols]).all(axis=1)
            tr, te = np.flatnonzero(train & eligible), np.flatnonzero(test & eligible)
            train_gold = [e for e in gold(tr, data, events) if e['action'] == action]
            test_gold = [e for e in gold(te, data, events) if e['action'] == action]
            positive_rows = tr[(data['label'][tr] == LABELS[action]) & (data['label_direction'][tr] == -1)]
            assert len(positive_rows) == len(train_gold) > 0
            score = data['X'][:, meta['feature_names'].index(feature)] * sign
            threshold = float(np.median(score[positive_rows]))
            gap_col = meta['feature_names'].index('minutes_last_action')
            cooldown = 30 if action == 'open' else max(1, int(np.floor(np.quantile(data['X'][positive_rows, gap_col], .25))))
            baseline = model_fold['modules'][action]['baseline']
            baseline_predictions = baseline['evaluation_predictions']
            baseline_score = match_events(baseline_predictions, test_gold)
            for k in ['tp', 'fp', 'fn']:
                assert baseline_score[k] == baseline['evaluation_event'][k]
            trials = []
            for crossing in [False, True]:
                variant = 'crossing' if crossing else 'state'
                ptr = generate(data, meta, tr, score, threshold, crossing, cooldown, action)
                pte = generate(data, meta, te, score, threshold, crossing, cooldown, action)
                paths[fold['name'] + '_' + action + '_' + variant] = pte
                trials.append(dict(variant=variant, train=match_events(ptr, train_gold), evaluation=match_events(pte, test_gold),
                                   sensitivity_5m=match_events(pte, test_gold, 5 * M)))
            selected = min(trials, key=lambda t: (-t['train']['f1'], t['variant']))
            pte = paths[fold['name'] + '_' + action + '_' + selected['variant']]
            pm, am, scored = pairs(pte, test_gold)
            false_negatives = [e for i, e in enumerate(test_gold) if i not in am]
            false_positives = [e for i, e in enumerate(pte) if i not in pm]
            assert false_negatives or false_positives
            example = false_negatives[0] if false_negatives else false_positives[0]
            event_by_id = {e['event_id']: e for e in events}
            comparisons.append(dict(fold=fold['name'], action=action, feature=feature, sign=sign,
                                    training_positive_feature_quantiles=quantiles(score[positive_rows]),
                                    threshold_from_training=threshold, cooldown_minutes_from_training=cooldown,
                                    selected_by_training_only=selected['variant'], selected_evaluation=scored,
                                    selected_5m=selected['sensitivity_5m'], baseline_config=baseline['config_id'], baseline_evaluation=baseline_score,
                                    f1_difference=scored['f1'] - baseline_score['f1'],
                                    paired_interval_exploratory=paired_f1_interval(pte, baseline_predictions, test_gold, timestamp(fold['train_end']), timestamp(fold['eval_end'])),
                                    size=size_diagnostics(action, train_gold, test_gold, events, data, meta), all_trials=trials,
                                    evaluation_event_ids=[e['event_id'] for e in test_gold],
                                    evaluation_cycles=sorted({event_by_id[e['event_id']]['cycle_id'] for e in test_gold}),
                                    counterexample=dict(kind='FN' if false_negatives else 'FP', symbol=example['symbol'], time_utc=utc(example['time_ms']),
                                                        event_id=example.get('event_id'), panel_row=example.get('panel_row'))))
            print('MODULE_COMPARED', fold['name'], action, flush=True)
    gates = {}
    for action in MODULES:
        rows = [r for r in comparisons if r['action'] == action]
        ids = {i for r in rows for i in r['evaluation_event_ids']}
        actual = [e for e in events if e['event_id'] in ids]
        largest = min(Counter(e['symbol'] for e in actual).items(), key=lambda v: (-v[1], v[0]))[0]
        gains = []
        for r, fold, model_fold in zip(rows, first['folds'], models['folds']):
            target = [dict(symbol=e['symbol'], direction='SHORT', action=action, time_ms=e['order_time_ms'], event_id=e['event_id'])
                      for e in actual if e['event_id'] in r['evaluation_event_ids'] and e['symbol'] != largest]
            pred = [p for p in paths[r['fold'] + '_' + action + '_' + r['selected_by_training_only']] if p['symbol'] != largest]
            baseline = [p for p in model_fold['modules'][action]['baseline']['evaluation_predictions'] if p['symbol'] != largest]
            cs, bs = match_events(pred, target), match_events(baseline, target)
            gains.append(dict(fold=fold['name'], candidate=cs, baseline=bs, gain=cs['f1'] - bs['f1']))
        hits = sum(r['f1_difference'] >= .05 for r in rows)
        positive = sum(g['gain'] > 0 for g in gains)
        unique_cycles = {c for r in rows for c in r['evaluation_cycles']}
        passed = hits >= 2 and positive >= 2 and len(ids) >= 30 and len(unique_cycles) >= 15
        gates[action] = dict(passed=passed, folds_meeting_005=hits, leave_largest_symbol_positive_folds=positive,
                             evaluation_events=len(ids), evaluation_cycles=len(unique_cycles),
                             largest_event_symbol=largest, leave_largest_symbol_folds=gains)
    limit_summary = limits(data, meta, events, cycles, first)
    assert hashes(source_paths) == before
    result = dict(version=protocol['version'], status='completed_exploratory_module_comparison',
                  unique_candidates_evaluated=8, candidate_fold_evaluations=24, learning_model_searches=0,
                  comparisons=comparisons, module_gates=gates, limits=limit_summary,
                  decision=dict(combine_full_rule=False,
                                description='模块未满足完整组合的推进条件，停止本轮，不进入资金与成本验证。',
                                main_exit_policy='LIMIT placement candidate unavailable from saved source fields'),
                  source_files_unchanged=True, source_sha256=before)
    assert not all(g['passed'] for g in gates.values()) or limit_summary['status'] == 'placement_and_execution_candidate_blocked_by_source_fields'
    (DEST / 'comparisons.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (DEST / 'predictions.json').write_text(json.dumps(paths, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    (DEST / 'report.html').write_text(report(result))
    print(json.dumps(dict(module_gates=gates, decision=result['decision'], report=str(DEST / 'report.html')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
