"""Frozen v3 comparison of causal context for observed SHORT-fill timing.

This is a local conditional behavior study. Observed prior fills supply state;
neither order intent nor an independently tradeable account is reconstructed.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings

from build_reverse_panel import merge_symbol_rows
from diagnose_reverse_failures import pairs, table, utc
from reverse_metrics import match_events, paired_f1_interval
from run_reverse_rules import timestamp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/results'
DEST = OUT / 'observed_behavior_v3'
M = 60000
ACTIONS = {'open': 1, 'add': 2, 'reduce': 3, 'close': 4}
NAMES = {'open': '开仓成交', 'add': '加仓成交', 'reduce': '减仓成交', 'close': '全部平仓成交'}
warnings.filterwarnings('error', category=ConvergenceWarning)


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_sources():
    saved = read(DEST / 'source_hashes.json')
    for name, digest in saved.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    return len(saved)


def market_extras(data, meta, protocol):
    """Every dense index i uses bars [.., i), never candle i or later."""
    supplement = read(OUT / 'reverse_market/manifest.json')['results']
    times = data['timeline_ms']
    warm = 1500
    dense_times = np.arange(int(times[0]) - warm * M, int(times[-1]) + M, M)
    at = np.arange(warm, len(dense_times))
    names = protocol['features_extra']['market']
    values = np.full((len(data['time_ms']), len(names)), np.nan, dtype=np.float32)
    market_audit = {}

    def dense(symbol):
        bars, _ = merge_symbol_rows(symbol, supplement)
        rows = [bars.get(int(t)) for t in dense_times]
        valid = np.array([r is not None for r in rows])
        for t, r in zip(dense_times, rows):
            if r is not None:
                assert int(r[0]) == t and int(r[6]) < t + M
        arrays = {key: np.array([float(r[col]) if r is not None else np.nan for r in rows])
                  for key, col in [('close', 4), ('high', 2), ('low', 3), ('qv', 7)]}
        cumulative = np.r_[0, np.cumsum(valid)]
        ret = {}
        for lag in [240, 1440]:
            ready = cumulative[at] - cumulative[at - lag - 1] == lag + 1
            ret[lag] = np.where(ready, arrays['close'][at - 1] / arrays['close'][at - lag - 1] - 1, np.nan)
        return arrays, cumulative, ret

    _, _, btc = dense('BTCUSDT')
    count = len(times)
    for si, symbol in enumerate(meta['symbols']):
        a, cum, ret = dense(symbol)
        maximum = pd.Series(a['high']).rolling(1440).max().to_numpy()[at - 1]
        minimum = pd.Series(a['low']).rolling(1440).min().to_numpy()[at - 1]
        width = maximum - minimum
        range_pos = np.full(count, np.nan)
        np.divide(a['close'][at - 1] - minimum, width, out=range_pos, where=width > 0)
        qsum = np.r_[0, np.cumsum(np.nan_to_num(a['qv'], nan=0.0))]
        # Zeros above are arithmetic placeholders only; completeness is mandatory.
        previous = (qsum[at - 60] - qsum[at - 1500]) / 1440
        current = (qsum[at] - qsum[at - 60]) / 60
        ready = (cum[at] - cum[at - 1500] == 1500) & (previous > 0)
        volume = np.full(count, np.nan)
        np.divide(current, previous, out=volume, where=ready)
        rows = slice(si * count, (si + 1) * count)
        values[rows] = np.column_stack([ret[240], ret[1440], ret[240] - btc[240],
                                       ret[1440] - btc[1440], range_pos, volume])
        market_audit[symbol] = int(np.isfinite(values[rows]).all(axis=1).sum())
        print('FEATURES', symbol, flush=True)
    return values, market_audit


def state_extras(data, meta, events):
    t, x = data['time_ms'], data['X']
    f = {name: x[:, i] for i, name in enumerate(meta['feature_names'])}
    out = np.full((len(t), 5), np.nan, dtype=np.float32)
    anchors = np.full(len(t), -1, dtype=np.int64)
    grouped = defaultdict(list)
    for e in events:
        if e['classification'] == 'history_anchored':
            grouped[e['cycle_id']].append(e)
    for ci, cid in enumerate(meta['cycle_ids']):
        rows = np.flatnonzero((data['cycle_idx'] == ci) & data['state_known_mask'])
        if not len(rows):
            continue
        increases = sorted([e for e in grouped[cid] if e['action'] in ['open', 'add']],
                           key=lambda e: (e['time_ms'], e['source_index']))
        inc_times = np.array([e['time_ms'] for e in increases], dtype=np.int64)
        inc_prices = np.array([float(e['price']) for e in increases])
        selected = np.searchsorted(inc_times, t[rows], side='left') - 1
        assert (selected >= 0).all()
        anchors[rows] = inc_times[selected]
        assert (anchors[rows] < t[rows]).all()
        direction, cost, close = data['direction'][rows], f['avgcost'][rows], f['close'][rows]
        out[rows, 0] = -direction * (close / inc_prices[selected] - 1)
        out[rows, 1] = (t[rows] - anchors[rows]) / M
        valid_previous = (rows > 0) & (data['symbol_idx'][rows] == data['symbol_idx'][rows - 1])
        out[rows[valid_previous], 2] = (direction * (close - f['close'][rows - 1]) / cost)[valid_previous]
        out[rows, 3] = direction * (close - close / (1 + f['ret5'][rows])) / cost
        marks = f['mark_return'][rows]
        assert np.isfinite(marks).all()
        out[rows, 4] = marks - np.minimum.accumulate(marks)
    return out, anchors


def fill_labels(data, meta, events):
    groups = defaultdict(list)
    symbols = {s: i for i, s in enumerate(meta['symbols'])}
    start, end = int(data['timeline_ms'][0]), int(data['timeline_ms'][-1]) + M
    count = len(data['timeline_ms'])
    for e in events:
        if e['symbol'] in symbols and start <= e['time_ms'] < end:
            row = symbols[e['symbol']] * count + (e['time_ms'] // M * M - start) // M
            groups[int(row)].append(e)
    labels = np.zeros(len(data['time_ms']), dtype=np.int8)
    allowed = data['state_known_mask'].copy()
    actual, excluded, long_count = [], [], Counter()
    for row, ee in groups.items():
        reason = None
        if len(ee) != 1:
            reason = 'multiple_fills_in_same_symbol_minute'
        elif ee[0]['classification'] != 'history_anchored':
            reason = 'unanchored_fill'
        else:
            e = ee[0]
            if e['direction'] != 'SHORT':
                long_count[e['action']] += 1
                continue
            state_ok = (data['direction'][row] == 0 if e['action'] == 'open' else
                        data['direction'][row] == -1 and data['cycle_idx'][row] >= 0 and
                        meta['cycle_ids'][int(data['cycle_idx'][row])] == e['cycle_id'])
            if not allowed[row] or not state_ok:
                reason = 'pre_minute_state_unknown_or_incompatible'
            else:
                labels[row] = ACTIONS[e['action']]
                actual.append(dict(symbol=e['symbol'], direction='SHORT', action=e['action'],
                                   time_ms=e['time_ms'], event_id=e['event_id'], cycle_id=e['cycle_id'],
                                   order_type=e['source_order_type'], panel_row=row))
        if reason is not None:
            allowed[row] = False
            excluded.extend(dict(event_id=e['event_id'], reason=reason, panel_row=row) for e in ee)
    return labels, allowed, actual, excluded, dict(long_count)


def interval_mask(data, meta, cycles, lo, hi):
    mask = (data['time_ms'] >= lo) & (data['time_ms'] < hi)
    purged = []
    for c in cycles:
        if c['status'] != 'complete' or c['symbol'] not in meta['symbols']:
            continue
        if any(c['start_ms'] < b <= c['end_ms'] for b in [lo, hi]):
            si = meta['symbols'].index(c['symbol'])
            mask &= ~((data['symbol_idx'] == si) & (data['time_ms'] >= c['start_ms'] // M * M) &
                      (data['time_ms'] <= c['end_ms'] // M * M))
            purged.append(c['cycle_id'])
    return mask, purged


def predict(rows, scores, threshold, data, meta, action, cooldown):
    kept, last = [], {}
    for r in rows[scores >= threshold]:
        si, now = int(data['symbol_idx'][r]), int(data['time_ms'][r])
        if si in last and now - last[si] < cooldown * M:
            continue
        kept.append(dict(symbol=meta['symbols'][si], direction='SHORT', action=action,
                         time_ms=now, panel_row=int(r)))
        last[si] = now
    return kept


def choice_key(trial):
    s = trial['calibration']
    return (-s['f1'], s['fp'], trial['multiplier'], trial['variant'])


def make_report(result):
    comparisons = result['comparisons']
    passed = [NAMES[a] for a, g in result['gates'].items() if g['passed']]
    verdict = '、'.join(passed) + '达到局部推进门槛' if passed else '四个动作均未达到局部推进门槛'
    body = '<h1>现有数据第三轮研究</h1><p class="sub">4小时 / 24小时行情 · 上次加仓锚点 · 已成交退出</p>'
    body += '<div class="card"><b>' + verdict + '。</b><p>本轮只检验已发生的成交时间。观察到的持仓用于局部预测；没有还原未公开的挂单、撤单和未成交意图，也没有形成独立运行策略。</p></div>'
    body += '<h2>本轮做了什么</h2><p>在原先33个币种和三个时间段上，比较原特征逻辑回归、扩展特征逻辑回归、单指标基准。每类固定3个报警阈值，共36项策略设置、108项分段评分。每段训练期前70%拟合、后30%选择阈值与模型；跨边界的整个持仓周期剔除。所有历史已被前两轮研究看过，结果属于历史探索。</p>'
    body += '<p>扩展输入：4小时和24小时涨跌、相对BTC涨跌、24小时价格位置、最近1小时成交额相对此前24小时；加减仓和退出再加入上次开仓/加仓成交价、间隔、价格变化和从最差浮亏的回升。全部行情截至决策分钟前，持仓只使用此前已经成交的记录。</p>'
    body += '<h2>三个时间段的结果</h2><p>按币种、方向、动作进行一对一匹配，主口径允许前后60秒。基准从原特征模型与单指标规则中按校准段选出；候选从两种逻辑回归中按校准段选出。TP为命中、FP为误报、FN为漏报。</p>'
    body += table(['时间段 / 动作', '校准选定', 'TP / FP / FN', '候选F1', '基准F1', '差值', '精确率 / 召回率', '5分钟F1'], [
        [r['fold'] + ' / ' + NAMES[r['action']], r['selected']['variant'],
         ' / '.join(str(r['selected']['evaluation'][k]) for k in ['tp', 'fp', 'fn']),
         f'{r["selected"]["evaluation"]["f1"]:.4f}', f'{r["baseline"]["evaluation"]["f1"]:.4f}',
         f'{r["gain"]:+.4f}', f'{r["selected"]["evaluation"]["precision"]:.1%} / {r["selected"]["evaluation"]["recall"]:.1%}',
         f'{r["selected"]["sensitivity_5m"]["f1"]:.4f}'] for r in comparisons])
    body += '<h2>汇总与推进判定</h2><p>沿用F1改善至少0.05且三段中至少两段达到、至少30个事件/15个周期、剔除最大事件币种后至少两段仍改善的工作门槛。新退出标签包含限价成交，因此不能与旧版仅市价退出的F1直接比较。</p>'
    body += table(['动作', '评估事件 / 周期', '命中 / 误报 / 漏报', '合并F1', '改善≥0.05段数', '去最大币种仍改善段数', '结论'], [
        [NAMES[a], f'{g["evaluation_events"]} / {g["evaluation_cycles"]}',
         ' / '.join(str(g['pooled'][k]) for k in ['tp', 'fp', 'fn']), f'{g["pooled"]["f1"]:.4f}',
         g['folds_gain_005'], g['leave_largest_positive_folds'], '局部通过' if g['passed'] else '未通过']
        for a, g in result['gates'].items()])
    body += '<h2>扩展特征是否增加信息</h2><p>以下两种模型各自的阈值都只由校准段选出；这里完整展示扩展模型相对原特征模型的结果，不能事后据此重新挑模型。</p>'
    body += table(['时间段 / 动作', '原特征F1', '扩展特征F1', '差值', '扩展模型主要权重'], [
        [r['fold'] + ' / ' + NAMES[r['action']], f'{r["ablation"]["base_f1"]:.4f}',
         f'{r["ablation"]["extended_f1"]:.4f}', f'{r["ablation"]["gain"]:+.4f}',
         ', '.join(n for n, _ in r['model_details']['extended']['top_standardized_coefficients'][:3])]
        for r in comparisons])
    body += '<h2>样本覆盖</h2>'
    body += table(['时间段 / 动作', '拟合正例', '校准正例', '评估正例', '评估可用分钟', '区间内原始短向事件', '边界剔除 / 特征缺失'], [
        [r['fold'] + ' / ' + NAMES[r['action']], r['counts']['fit_positive'], r['counts']['calibration_positive'],
         r['counts']['evaluation_positive'], r['counts']['evaluation_minutes'], r['coverage']['raw_short_events'],
         f'{r["coverage"]["purged_events"]} / {r["coverage"]["feature_or_state_excluded_events"]}'] for r in comparisons])
    body += '<h2>错误实例</h2><p>每个选定模块展示一条漏报或误报；时间为UTC。完整预测和事件编号保存在数据文件中。</p>'
    body += table(['时间段 / 动作', '错误', '币种', '时间UTC'], [[r['fold'] + ' / ' + NAMES[r['action']],
        r['counterexample']['kind'], r['counterexample']['symbol'], r['counterexample']['time_utc']] for r in comparisons])
    body += '<h2>可复现记录</h2><p>固定协议、每个候选的结果、预测、特征缓存和核对结果均已保存。原有数据和报告文件逐一核对摘要保持不变。</p><p><a href="protocol.json">固定研究协议</a> · <a href="results.json">完整比较</a> · <a href="verification.json">核对结果</a> · <a href="predictions.json">全部预测</a></p>'
    body += '<p class="sub">本轮不计算账户收益，不开始模拟盘或实盘。33个币种来自事后观察到的交易，不能据此声称已经恢复全市场选币规则。</p>'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>现有数据第三轮研究</title><style>body{font:16px/1.7 -apple-system,"PingFang SC",sans-serif;background:#f5f7fb;color:#243247;margin:0}main{max-width:1280px;margin:auto;padding:32px 24px}h1{font-size:30px;margin-bottom:0}h2{margin-top:32px;font-size:22px}.sub{color:#64748b}.card{background:#e8effa;border-left:4px solid #3767b0;padding:18px 22px;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px;border-bottom:1px solid #dfe5ef;text-align:left}th{background:#eaf0f8}a{color:#245cad}</style><main>' + body + '</main></html>'


def main():
    protocol = read(DEST / 'protocol.json')
    protocol_digest = hashlib.sha256((DEST / 'protocol.json').read_bytes()).hexdigest()
    protected_count = verify_sources()
    data = dict(np.load(OUT / 'reverse_panel.npz', allow_pickle=False))
    meta, events, cycles = read(OUT / 'reverse_panel_meta.json'), read(OUT / 'leader_events.json'), read(OUT / 'leader_cycles.json')
    mx, market_audit = market_extras(data, meta, protocol)
    sx, anchors = state_extras(data, meta, events)
    extra_names = protocol['features_extra']['market'] + protocol['features_extra']['position']
    extra = np.column_stack([mx, sx])
    np.savez_compressed(DEST / 'features.npz', X_extra=extra, feature_names=np.array(extra_names), anchor_time_ms=anchors)
    label, allowed, actual, excluded, long_counts = fill_labels(data, meta, events)
    write('label_audit.json', dict(target='observed successful SHORT fills only', events=actual, excluded=excluded,
                                   long_event_counts_separate=long_counts, complete_market_minutes_by_symbol=market_audit))
    feature_map = {name: data['X'][:, i] for i, name in enumerate(meta['feature_names'])}
    feature_map.update({name: extra[:, i] for i, name in enumerate(extra_names)})
    comparisons, all_predictions, all_actual, score_archive = [], {}, {}, {}
    mcfg = protocol['model']
    fits = 0
    for fold in protocol['folds']:
        lo, mid, hi, end = [timestamp(fold[k]) for k in ['train_start', 'calibration_start', 'train_end', 'eval_end']]
        ranges, purges = {}, {}
        for part, start, stop in [('fit', lo, mid), ('calibration', mid, hi), ('evaluation', hi, end)]:
            ranges[part], purges[part] = interval_mask(data, meta, cycles, start, stop)
        for action, code in ACTIONS.items():
            basic = protocol['features_base']['market'] + (protocol['features_base']['position'] if action != 'open' else [])
            extended = basic + protocol['features_extra']['market'] + (protocol['features_extra']['position'] if action != 'open' else [])
            matrix = np.column_stack([feature_map[n] for n in extended])
            eligible = allowed & (data['direction'] == (0 if action == 'open' else -1)) & np.isfinite(matrix).all(axis=1)
            indices = {part: np.flatnonzero(mask & eligible) for part, mask in ranges.items()}
            targets = {part: [e for e in actual if e['action'] == action and ranges[part][e['panel_row']] and eligible[e['panel_row']]] for part in ranges}
            for part, rows in indices.items():
                assert int((label[rows] == code).sum()) == len(targets[part]) > 0, (fold['name'], action, part)
                assert all(e['cycle_id'] not in purges[part] for e in targets[part])
            fit_rows = indices['fit']
            yfit = label[fit_rows] == code
            negatives = np.flatnonzero(~yfit)
            sampled = np.random.default_rng(mcfg['seed']).choice(negatives, min(len(negatives), mcfg['negative_fit_limit']), replace=False)
            fit_pick = np.sort(np.r_[np.flatnonzero(yfit), sampled])
            trials, details = [], {}
            group = fold['name'] + '_' + action
            all_actual[group] = targets['evaluation']
            all_actual[group + '_calibration'] = targets['calibration']
            score_archive[group + '_eval_rows'] = indices['evaluation']
            score_archive[group + '_cal_rows'] = indices['calibration']
            for variant in protocol['variants']:
                if variant == 'scalar_baseline':
                    name, sign = protocol['scalar_baselines'][action]
                    scores = {part: sign * feature_map[name][rows] for part, rows in indices.items()}
                    details[variant] = dict(feature=name, sign=sign)
                else:
                    size = len(basic) if variant == 'base' else len(extended)
                    selected_names = extended[:size]
                    scaler = StandardScaler().fit(matrix[fit_rows[fit_pick], :size])
                    model = LogisticRegression(C=mcfg['C'], class_weight=mcfg['class_weight'], solver=mcfg['solver'],
                                               max_iter=mcfg['max_iter'], random_state=mcfg['seed'])
                    model.fit(scaler.transform(matrix[fit_rows[fit_pick], :size]), yfit[fit_pick])
                    scores = {part: model.predict_proba(scaler.transform(matrix[rows, :size]))[:, 1] for part, rows in indices.items()}
                    fits += 1
                    details[variant] = dict(features=selected_names, intercept=float(model.intercept_[0]),
                        coefficients=model.coef_[0].tolist(), scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                        top_standardized_coefficients=sorted(zip(selected_names, model.coef_[0].tolist()), key=lambda p: -abs(p[1]))[:6],
                        fit_positive=int(yfit.sum()), fit_sampled_negative=len(sampled),
                        negative_sampling_probability=len(sampled) / len(negatives))
                score_archive[group + '_' + variant + '_eval_scores'] = scores['evaluation']
                score_archive[group + '_' + variant + '_cal_scores'] = scores['calibration']
                for mult in protocol['thresholds']['raw_alert_quota_multipliers']:
                    quantile = 1 - float(yfit.mean()) * mult
                    assert 0 < quantile < 1
                    threshold = float(np.quantile(scores['fit'], quantile))
                    pp = {part: predict(indices[part], scores[part], threshold, data, meta, action, protocol['cooldown_minutes'])
                          for part in ['calibration', 'evaluation']}
                    trial_id = group + '_' + variant + '_' + str(mult)
                    all_predictions[trial_id] = pp['evaluation']
                    all_predictions[trial_id + '_calibration'] = pp['calibration']
                    trials.append(dict(id=trial_id, variant=variant, multiplier=mult, threshold_from_fit=threshold,
                        threshold_quantile=quantile, fit_equal_threshold=int((scores['fit'] == threshold).sum()),
                        calibration=match_events(pp['calibration'], targets['calibration']),
                        evaluation=match_events(pp['evaluation'], targets['evaluation']),
                        sensitivity_5m=match_events(pp['evaluation'], targets['evaluation'], 5 * M),
                        evaluation_average_precision=float(average_precision_score(label[indices['evaluation']] == code, scores['evaluation']))))
            selected = min([t for t in trials if t['variant'] in ['base', 'extended']], key=choice_key)
            baseline = min([t for t in trials if t['variant'] in ['base', 'scalar_baseline']], key=choice_key)
            by_variant = {v: min([t for t in trials if t['variant'] == v], key=choice_key) for v in protocol['variants']}
            pp, bb = all_predictions[selected['id']], all_predictions[baseline['id']]
            matched_p, matched_a, _ = pairs(pp, targets['evaluation'])
            fn = [e for i, e in enumerate(targets['evaluation']) if i not in matched_a]
            fp = [e for i, e in enumerate(pp) if i not in matched_p]
            assert fn or fp
            example = fn[0] if fn else fp[0]
            raw_events = [e for e in events if e['classification'] == 'history_anchored' and e['direction'] == 'SHORT' and
                          e['action'] == action and hi <= e['time_ms'] < end]
            purged_events = [e for e in raw_events if e['cycle_id'] in purges['evaluation']]
            comparisons.append(dict(fold=fold['name'], action=action, selected=selected, baseline=baseline,
                gain=selected['evaluation']['f1'] - baseline['evaluation']['f1'],
                paired_interval_exploratory=paired_f1_interval(pp, bb, targets['evaluation'], hi, end),
                ablation=dict(base_f1=by_variant['base']['evaluation']['f1'], extended_f1=by_variant['extended']['evaluation']['f1'],
                              gain=by_variant['extended']['evaluation']['f1'] - by_variant['base']['evaluation']['f1']),
                counts={**{part + '_positive': len(targets[part]) for part in indices},
                        **{part + '_minutes': len(indices[part]) for part in indices}},
                coverage=dict(raw_short_events=len(raw_events), purged_events=len(purged_events),
                              feature_or_state_excluded_events=len(raw_events) - len(purged_events) - len(targets['evaluation']),
                              evaluation_order_types=dict(Counter(e['order_type'] for e in targets['evaluation']))),
                purged_cycle_ids=purges, model_details=details, all_trials=trials,
                counterexample=dict(kind='FN' if fn else 'FP', symbol=example['symbol'], time_utc=utc(example['time_ms']),
                                    event_id=example.get('event_id'), panel_row=example['panel_row'])))
            print('COMPARED', group, 'selected', selected['variant'], selected['multiplier'],
                  'TP/FP/FN', [selected['evaluation'][k] for k in ['tp', 'fp', 'fn']], flush=True)
    gates = {}
    for action in ACTIONS:
        rows = [r for r in comparisons if r['action'] == action]
        aa = [e for r in rows for e in all_actual[r['fold'] + '_' + action]]
        pp = [p for r in rows for p in all_predictions[r['selected']['id']]]
        largest = Counter(e['symbol'] for e in aa).most_common(1)[0][0]
        leave = []
        for r in rows:
            keep = lambda zz: [v for v in zz if v['symbol'] != largest]
            gold = keep(all_actual[r['fold'] + '_' + action])
            cs = match_events(keep(all_predictions[r['selected']['id']]), gold)
            bs = match_events(keep(all_predictions[r['baseline']['id']]), gold)
            leave.append(dict(fold=r['fold'], candidate=cs, baseline=bs, gain=cs['f1'] - bs['f1']))
        above = sum(r['gain'] >= protocol['gates']['f1_gain_vs_calibration_selected_baseline'] for r in rows)
        positive = sum(r['gain'] > 0 for r in leave)
        ncycles = len({e['cycle_id'] for e in aa})
        gates[action] = dict(passed=above >= 2 and positive >= 2 and len(aa) >= 30 and ncycles >= 15,
            evaluation_events=len(aa), evaluation_cycles=ncycles, pooled=match_events(pp, aa), folds_gain_005=above,
            largest_event_symbol=largest, leave_largest_positive_folds=positive, leave_largest=leave)
    assert len(comparisons) == 12 and sum(len(r['all_trials']) for r in comparisons) == 108 and fits == 24
    verify_sources()
    assert hashlib.sha256((DEST / 'protocol.json').read_bytes()).hexdigest() == protocol_digest
    result = dict(version='observed-behavior-v3', status='completed_exploratory_conditional_fill_study',
        protocol_sha256=protocol_digest, source_files_unchanged=protected_count, candidate_budget=protocol['candidate_budget'],
        comparisons=comparisons, gates=gates, limits=protocol['limits'],
        decision=dict(complete_strategy=False, reason='Observed-state local models do not supply independent position evolution or unobserved LIMIT intent.'))
    write('results.json', result)
    write('predictions.json', all_predictions)
    write('actual_events.json', all_actual)
    np.savez_compressed(DEST / 'scores.npz', **score_archive)
    (DEST / 'report.html').write_text(make_report(result))
    print('COMPLETED', {a: g['passed'] for a, g in gates.items()}, flush=True)


if __name__ == '__main__':
    main()
