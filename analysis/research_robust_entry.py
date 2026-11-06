"""Bounded v7 entry ablation; preserve the completed v6 experiment."""
import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json

import joblib
import numpy as np

from behavior_hazard import M, probability_card
from entry_management import clock_features
from research_joint_entry import (bootstrap_gain, decode, event_result, fit_head,
                                  predict_head, read, sample_fit, sha)
from robust_entry import FitClipper, calibrate_threshold, decode_refractory, timing_inputs
from run_entry_management import load_view, stamp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'analysis/results/joint_entry_v6'
DEFAULT_OUT = ROOT / 'analysis/results/robust_entry_v7'
LEARNED = ('bounded_linear', 'bounded_timing')
BASELINES = ('clock_uniform', 'v5_factorized')
COOLDOWN = 30


def write(out, name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def protocol(source):
    return dict(
        version='robust-entry-v7', registered_at_utc=datetime.now(timezone.utc).isoformat(),
        historical_status='Exploratory reuse of previously inspected history; no untouched holdout.',
        question='Do bounded market inputs and dispersed alerts improve strict opening replication?',
        folds=source['folds'], market_features=source['market_features'],
        clock_features=source['clock_features'], target=source['target'],
        additional_timing_features=['ret1', 'ret3', 'pullback15'],
        timing_availability='Derived only from the already lagged close and its own-symbol past; pullback15 is relative to the highest of the latest 15 lagged closes, not intrabar highs. Complete windows required.',
        clipping=dict(quantiles=[.005, .995], fit='Full natural fit rows only; never calibration or evaluation.',
                      columns='Market inputs including added timing features; clock is unchanged.'),
        model=source['candidates']['joint_linear'],
        fit={**source['fit'], 'features': '14 market features, optionally 3 causal short-term features, plus unchanged UTC sine/cosine; no trader state or symbol ID.'},
        learned_candidates=list(LEARNED),
        ablations=['Original v6 scores with the original decoder (exact replay).',
                   'Original v6 linear/tree scores with the dispersed decoder.',
                   'Bounded linear and bounded timing scores, each with both decoders.'],
        dispersed_decoder=dict(
            cooldown_minutes=COOLDOWN,
            selection='Highest score above threshold among symbols outside their own-alert cooldown; stable ties.',
            daily_cap=source['decoder']['daily_cap'],
            threshold_quantiles=[.9, .95, .97, .98, .99, .995, .997, .999, .9995, '1-fit_any_open_rate'],
            threshold_selection='Minimize absolute difference between decoded calibration alert count and fit any-open rate times public-ready calibration minutes; ties favor higher threshold. No calibration hit-rate selection.',
            scope='Cooldown, threshold recalibration and eligible-symbol reselection are a joint decoder change.',
            cost='A 30-minute own-alert cooldown can suppress genuine rapid re-entry.'),
        latency_minutes=source['decoder']['latency_minutes'],
        scoring=source['metrics'], progression=source['progression'],
        budget=dict(learned_heads=6, model_hyperparameter_search=0,
                    decoder_threshold_candidates_at_most=10, new_market_downloads=0),
        stop='Run these two fixed learned candidates on three folds once. Do not expand after viewing evaluation.',
        limitations=source['limits'])


def gate_checks(card, fold_cards, rules):
    primary = card['scenarios'][0]['primary']
    drop_f1 = card['scenarios'][0]['leave_largest_symbol']['primary']['f1']
    return dict(
        log_loss=all(g['relative_gain'] >= rules['relative_log_loss_gain_vs_both_baselines']
                     for g in card['gain_vs_baselines'].values()),
        mass=rules['event_mass_ratio'][0] <= card['probability']['event_mass_ratio'] <= rules['event_mass_ratio'][1],
        precision=primary['precision'] >= rules['coin_specific_precision_min'],
        recall=primary['recall'] >= rules['coin_specific_recall_min'],
        leave_largest=all(drop_f1 > fold_cards[b + '__original']['scenarios'][0]['leave_largest_symbol']['primary']['f1']
                          for b in BASELINES))


def pooled_card(cards):
    count = {k: sum(c['scenarios'][0]['primary'][k] for c in cards) for k in ('tp', 'fp', 'fn')}
    tp, fp, fn = (count[k] for k in ('tp', 'fp', 'fn'))
    total = sum(c['probability']['minutes'] for c in cards)
    events = sum(c['probability']['events'] for c in cards)
    return dict(**count, precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn),
                log_loss=sum(c['probability']['log_loss']*c['probability']['minutes'] for c in cards)/total,
                event_mass_ratio=sum(c['probability']['expected_events'] for c in cards)/events,
                sensitivity_5m_tp=sum(c['scenarios'][0]['sensitivity_5m']['tp'] for c in cards))


def render_report(out, result):
    labels = {'clock_uniform': '时钟＋均匀选币', 'v5_factorized': 'v5 时机×选币',
              'joint_linear': '原线性模型', 'joint_tree': '原浅层树',
              'bounded_linear': '截尾线性模型', 'bounded_timing': '截尾＋短周期特征'}
    rows = []
    for name, card in result['pooled'].items():
        model, decoder = name.split('__')
        rows.append([labels[model], '原报警方式' if decoder == 'original' else '去重＋数量校准',
                     card['tp'], card['fp'], card['fn'], f"{card['precision']:.1%}",
                     f"{card['recall']:.1%}", f"{card['log_loss']:.6f}",
                     f"{card['event_mass_ratio']:.3f}", result['gates'].get(name, '参照')])
    def table(headers, items):
        return '<div class="scroll"><table><tr>' + ''.join('<th>'+escape(str(h))+'</th>' for h in headers) + '</tr>' + ''.join(
            '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in items) + '</table></div>'
    status = '有候选通过开仓模块后续研究门槛' if result['qualifying_opening_modules'] else '仍无合格开仓模块'
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>逆向建模 v7 对照结果</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f5f7fa;color:#243348;margin:0}main{max-width:1200px;margin:auto;padding:32px}h1{font-size:30px}.card{background:white;padding:20px;border-radius:12px;border:1px solid #d9e1ec;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;white-space:nowrap;font-size:14px}td,th{padding:10px 12px;border-bottom:1px solid #dde4ed;text-align:left}th{background:#eaf0f8}a{color:#215ca9}code{background:#eaf0f8;padding:2px 4px}</style><main>'''
    body += '<p>策略逆向建模 · v7 · 2026-10-13</p><h1>'+status+'</h1>'
    body += '<div class="card"><b>本轮改动</b><p>行情输入按训练期的 0.5%–99.5% 分位截尾，限制极端值外推；第二个固定候选增加 1 分钟、3 分钟收益率和距近 15 分钟最高收盘价的回撤。所有输入在决策时已经可见。</p><p>新的报警方式给同币设置 30 分钟冷却，并在校准段按实际报警数校准阈值。冷却期间可选择其他超阈值币种。每日额度沿用原规则。</p></div>'
    body += '<p>三个时间段共 '+str(result['pooled_eval_openings'])+' 次真实开仓。主评分仍为同币、一对一、±60 秒，且真实事件不得早于信号发出；默认延迟 1 分钟。±5 分钟仅作辅助。</p>'
    body += table(['模型', '报警方式', '命中', '误报', '漏报', '精确率', '召回率', '概率损失↓', '预测/实际总量', '通过段数/3'], rows)
    detail = []
    for f in result['folds']:
        for name, c in f['variants'].items():
            model, decoder = name.split('__')
            s = c['scenarios'][0]['primary']
            detail.append([f['fold'], labels[model], decoder, f"{s['tp']}/{s['fp']}/{s['fn']}",
                           f"{c['probability']['event_mass_ratio']:.3f}",
                           c['scenarios'][0]['sensitivity_5m']['tp'],
                           c['scenarios'][1]['primary']['tp'],
                           '通过' if c.get('gate_pass') else '未通过' if 'gate_pass' in c else '参照'])
    body += '<h2>分段与延迟检查</h2>'+table(['时段', '模型', '报警方式', 'TP/FP/FN', '概率总量比', '±5分钟命中', '延迟2分钟命中', '门槛'], detail)
    body += '<div class="card"><b>结果如何使用</b><p>概率损失衡量全体币种分钟的预测质量；报警精确率与召回率衡量是否复现具体开仓。改善其中一个指标不能替代其余指标。通过门槛仍要求同一候选至少两段满足原 v6 的全部标准。</p><p>这批历史已被反复研究，当前结果是探索性对照。数据没有新增；交易记录缺少挂单、撤单和未成交生命周期。30 分钟冷却也有代价：评估中有 '+str(sum(f['rapid_reentry_events'] for f in result['folds']))+' 次真实同币再开仓距前一次不足 30 分钟。</p><p>本轮仅改进开仓研究，不包含完整仓位管理、交易执行或收益结论。</p></div>'
    body += '<h2>复查</h2><p>旧四组模型的报警与指标逐条复现；旧产物内容哈希在运行前后核对。阈值只按校准报警数量选择，不按命中率选择。</p><p>'
    body += ' · '.join('<a href="'+name+'">'+name+'</a>' for name in ('protocol.json', 'results.json', 'models.json', 'predictions.json', 'source_hashes.json'))
    body += '</p></main></html>'
    (out/'report.html').write_text(body, encoding='utf-8')


def main(out=DEFAULT_OUT):
    source_protocol = read(SOURCE/'protocol.json')
    source_result = read(SOURCE/'results.json')
    assert sha(SOURCE/'protocol.json') == source_result['protocol_sha256']
    source_predictions = read(SOURCE/'predictions.json')
    protected = read(SOURCE/'source_hashes.json')
    changed = [p for p, digest in protected.items() if sha(ROOT/p) != digest]
    if changed:
        raise ValueError('Frozen source files changed: '+repr(changed))
    for path in sorted(SOURCE.iterdir()):
        if path.is_file():
            protected[str(path.relative_to(ROOT))] = sha(path)
    for name in ('robust_entry.py', 'research_robust_entry.py', 'research_joint_entry.py',
                 'tests/test_robust_entry.py', 'tests/test_joint_entry.py'):
        path = ROOT/'analysis'/name
        protected[str(path.relative_to(ROOT))] = sha(path)
    out.mkdir(parents=True, exist_ok=False)
    plan = protocol(source_protocol)
    write(out, 'protocol.json', plan)
    protocol_hash = sha(out/'protocol.json')
    write(out, 'source_hashes.json', protected)
    view = load_view()
    times, symbols = view['data']['timeline_ms'], view['meta']['symbols']
    n, ns = len(times), len(symbols)
    clock = np.tile(clock_features(times), (ns, 1))
    short, short_names = timing_inputs(view['features']['close'], ns, n)
    assert short_names == plan['additional_timing_features']
    inputs = {'bounded_linear': view['market'],
              'bounded_timing': np.column_stack([view['market'], short])}
    row_times = np.tile(times, ns)
    saved = np.load(SOURCE/'scores.npz', allow_pickle=False)
    y, ready, label_ok = (saved[k] for k in ('labels', 'public_ready', 'label_ok'))
    np.testing.assert_array_equal(y, view['rank_y'])
    np.testing.assert_array_equal(saved['times'], times)
    models, predictions, arrays, folds = {}, {}, {}, []
    for fold, old_fold in zip(plan['folds'], source_result['folds']):
        name = fold['name']
        assert old_fold['fold'] == name
        lo, mid, hi, end = [stamp(fold[k]) for k in ('train_start', 'calibration_start', 'train_end', 'eval_end')]
        fit_min = (times >= lo) & (times < mid) & view['timing_ok']
        cal_min = (times >= mid) & (times < hi) & view['timing_ok']
        rate, cap = float(view['timing_y'][fit_min].mean()), old_fold['daily_cap']
        eval_rows, cal_rows = saved[name+'_eval_rows'], saved[name+'_cal_rows']
        score_map = {v: saved[name+'_'+v+'_scores'] for v in old_fold['variants']}
        for variant, market in inputs.items():
            available = ready & np.isfinite(market).all(axis=1)
            # Identical evaluation/calibration denominators are required for this ablation.
            assert available[eval_rows].all() and available[cal_rows].all()
            fit_rows = np.flatnonzero(label_ok & available & (row_times >= lo) & (row_times < mid))
            clipper = FitClipper(tuple(plan['clipping']['quantiles'])).fit(market[fit_rows])
            x = np.column_stack([clipper.transform(market), clock])
            chosen, weights = sample_fit(y, fit_rows, plan['fit']['negative_limit'], plan['fit']['seed'])
            print('FIT', name, variant, 'positives', int(y[fit_rows].sum()), flush=True)
            bundle = fit_head('joint_linear', x, y, chosen, weights, cal_rows, plan['model'], plan['fit']['seed'])
            bundle['clipper'] = clipper
            joblib.dump(bundle, out/(name+'_'+variant+'.joblib'))
            values = np.full(ns*n, np.nan)
            values[available] = predict_head(bundle, x[available])
            score_map[variant] = values.reshape(ns, n)
            arrays[name+'_'+variant+'_scores'] = score_map[variant]
            arrays[name+'_'+variant+'_fit_rows'] = fit_rows
            arrays[name+'_'+variant+'_sample_rows'] = chosen
            arrays[name+'_'+variant+'_sample_weights'] = weights
            models[name+'_'+variant] = dict(
                natural_rows=len(fit_rows), positives=int(y[fit_rows].sum()),
                sampled_rows=len(chosen), weight_sum=float(weights.sum()),
                clipping_lower=clipper.lower_.tolist(), clipping_upper=clipper.upper_.tolist(),
                features=plan['market_features']+(short_names if variant == 'bounded_timing' else [])+plan['clock_features'],
                calibration_offset=bundle['offset'], calibration_rows=len(cal_rows),
                calibration_positives=int(y[cal_rows].sum()),
                evaluation_clipped_row_fraction=float(np.mean(np.any(market[eval_rows] != x[eval_rows, :market.shape[1]], axis=1))))
        truth = source_predictions[name+'_truth']
        predictions[name+'_truth'] = truth
        largest = old_fold['largest_symbol']
        last, rapid = {}, 0
        for e in sorted(truth, key=lambda e: e['time_ms']):
            previous = last.get(e['symbol'])
            rapid += previous is not None and e['time_ms']-previous < COOLDOWN*M
            last[e['symbol']] = e['time_ms']
        fc = dict(fold=name, actual_openings=len(truth), actual_cycles=old_fold['actual_cycles'],
                  daily_cap=cap, fit_any_open_rate=rate, largest_symbol=largest,
                  rapid_reentry_events=int(rapid), variants={})
        for model_name, scores in score_map.items():
            probability = probability_card(y[eval_rows], scores.ravel()[eval_rows])
            if model_name in old_fold['variants']:
                assert probability == old_fold['variants'][model_name]['probability']
            maximum = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=0)
            original_threshold = float(np.quantile(maximum[cal_min], 1-rate))
            decoders = ('original',) if model_name in BASELINES else ('original', 'dispersed')
            gain = {base: bootstrap_gain(y[eval_rows], scores.ravel()[eval_rows],
                                        score_map[base].ravel()[eval_rows], row_times[eval_rows], plan['fit']['seed'])
                    for base in BASELINES} if model_name not in BASELINES else None
            for decoder in decoders:
                key = model_name+'__'+decoder
                calibration = (dict(threshold=original_threshold) if decoder == 'original' else
                               calibrate_threshold(times, scores, symbols, mid, hi, rate, cap, COOLDOWN))
                card = dict(probability=probability, calibration=calibration, scenarios=[])
                for delay in plan['latency_minutes']:
                    args = (times, scores, symbols, calibration['threshold'], cap, hi, end, delay)
                    alerts, pending = decode(*args) if decoder == 'original' else decode_refractory(*args, cooldown_minutes=COOLDOWN)
                    predictions[name+'_'+key+'_delay'+str(delay)] = dict(alerts=alerts, pending=pending)
                    scenario = event_result(alerts, truth)
                    scenario.update(delay=delay, pending_at_end=len(pending),
                        label_ambiguity_alerts=sum(not view['timing_ok'][e['minute_index']] for e in alerts),
                        leave_largest_symbol=event_result([e for e in alerts if e['symbol'] != largest],
                                                         [e for e in truth if e['symbol'] != largest]))
                    card['scenarios'].append(scenario)
                    if model_name in old_fold['variants'] and decoder == 'original':
                        old = source_predictions[name+'_'+model_name+'_delay'+str(delay)]
                        assert dict(alerts=alerts, pending=pending) == old
                        assert scenario == old_fold['variants'][model_name]['scenarios'][plan['latency_minutes'].index(delay)]
                if gain is not None:
                    card['gain_vs_baselines'] = gain
                fc['variants'][key] = card
                print('SCORE', name, key, card['scenarios'][0]['primary'], flush=True)
        for key, card in fc['variants'].items():
            if 'gain_vs_baselines' in card:
                card['gate_checks'] = gate_checks(card, fc['variants'], plan['progression'])
                card['gate_pass'] = all(card['gate_checks'].values())
        folds.append(fc)
    gates = {name: sum(f['variants'][name]['gate_pass'] for f in folds)
             for name, c in folds[0]['variants'].items() if 'gate_pass' in c}
    events = sum(f['actual_openings'] for f in folds)
    cycles = len({e['cycle_id'] for k, records in predictions.items() if k.endswith('_truth') for e in records})
    qualifying = [name for name, count in gates.items()
                  if count >= plan['progression']['same_candidate_required_folds']
                  and events >= plan['progression']['min_pooled_eval_openings']
                  and cycles >= plan['progression']['min_pooled_eval_cycles']]
    changed = [p for p, digest in protected.items() if sha(ROOT/p) != digest]
    assert not changed, changed
    assert sha(out/'protocol.json') == protocol_hash
    result = dict(version=plan['version'], completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=protocol_hash, folds=folds, gates=gates,
                  pooled={key: pooled_card([f['variants'][key] for f in folds]) for key in folds[0]['variants']},
                  pooled_eval_openings=events, pooled_eval_cycles=cycles,
                  qualifying_opening_modules=qualifying, complete_strategy_qualified=False,
                  protected_original_files=len(protected), changed_original_files=changed,
                  exact_v6_alert_replay=True)
    np.savez_compressed(out/'scores.npz', **arrays)
    write(out, 'models.json', models)
    write(out, 'predictions.json', predictions)
    write(out, 'results.json', result)
    render_report(out, result)
    print('DONE', str(out), 'gates', gates, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT, help='New output directory; must not exist.')
    main(parser.parse_args().output)
