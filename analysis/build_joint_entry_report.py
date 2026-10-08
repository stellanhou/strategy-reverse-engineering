"""Render the frozen joint-entry study from verified result artifacts."""
from html import escape
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/results/joint_entry_v6'
NAMES = {'clock_uniform': '时钟＋均匀选币', 'v5_factorized': 'v5 时机×选币',
         'joint_linear': '联合线性模型', 'joint_tree': '联合浅层树'}


def read(name):
    return json.loads((OUT/name).read_text())


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    r, p, v = read('results.json'), read('protocol.json'), read('verification.json')
    assert v['status'] == 'passed'
    assert not r['qualifying_opening_modules'], 'rewrite interpretation if study passes'
    summary, detailed, gains, ranking, sensitivity = [], [], [], [], []
    for variant in NAMES:
        cards = [f['variants'][variant] for f in r['folds']]
        tp, fp, fn = [sum(c['scenarios'][0]['primary'][k] for c in cards) for k in ('tp', 'fp', 'fn')]
        wide = sum(c['scenarios'][0]['sensitivity_5m']['tp'] for c in cards)
        summary.append([NAMES[variant], tp, fp, fn, wide,
                        '参照' if variant not in r['gates'] else f"{r['gates'][variant]} / 3"])
    for f in r['folds']:
        for variant, c in f['variants'].items():
            s = c['scenarios'][0]['primary']
            detailed.append([f['fold'], NAMES[variant], f"{s['tp']} / {s['fp']} / {s['fn']}",
                f"{s['precision']:.1%} / {s['recall']:.1%}", f"{c['probability']['log_loss']:.6f}",
                f"{c['probability']['event_mass_ratio']:.3f}", f['daily_cap']])
            ranking.append([f['fold'], NAMES[variant], c['rank_oracle']['top1'], c['rank_oracle']['top3'], f['actual_openings']])
            sensitivity.append([f['fold'], NAMES[variant],
                c['scenarios'][0]['primary']['tp'], c['scenarios'][1]['primary']['tp'],
                c['scenarios'][0]['timing_only']['tp'], c['scenarios'][0]['leave_largest_symbol']['primary']['f1'], f['largest_symbol']])
            if 'gain_vs_baselines' in c:
                for base, g in c['gain_vs_baselines'].items():
                    gains.append([f['fold'], NAMES[variant], NAMES[base], f"{g['relative_gain']:+.2%}",
                                  f"{g['interval_95'][0]:+.2%} ～ {g['interval_95'][1]:+.2%}"])
    periods = []
    models = read('models.json')
    for f in p['folds']:
        d = models[f['name']+'_joint_linear']
        periods.append([f['name'], f['calibration_start'], f['train_end'], f['eval_end'],
                        d['natural_positives'], d['calibration_positives'], d['natural_rows']])
    css = '''*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#202b3b;font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1160px;margin:auto;padding:40px 28px 64px}h1{font-size:32px;line-height:1.3;margin:12px 0}h2{font-size:22px;margin:30px 0 12px}.eyebrow{font-size:13px;letter-spacing:.12em;color:#59697e}.sub{color:#5b687a}.card{padding:20px 24px;border-radius:12px;background:#fff;border:1px solid #dbe2eb;margin:20px 0}.verdict{border-left:5px solid #c68432;background:#fff8ed}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.stat{background:#fff;padding:18px;border:1px solid #dbe2eb;border-radius:10px}.stat b{font-size:28px;display:block}.stat span{font-size:14px;color:#526174}p{margin:10px 0}.scroll{overflow:auto;border:1px solid #dbe2eb;border-radius:8px;background:#fff;margin:16px 0}table{border-collapse:collapse;min-width:680px;width:100%;font-size:14px;white-space:nowrap}th,td{padding:11px 14px;border-bottom:1px solid #e1e6ed;text-align:left}th{background:#eaf0f6;color:#354862}tr:last-child td{border-bottom:0}a{color:#255fb0}code{background:#e7edf4;padding:2px 5px;border-radius:4px}.small{font-size:13px;color:#5b687a;overflow-wrap:anywhere}details{margin:18px 0}summary{cursor:pointer;font-weight:600}.links{display:flex;gap:14px;flex-wrap:wrap}@media(max-width:680px){main{padding:24px 14px}h1{font-size:26px}.stats{grid-template-columns:1fr}.stat b{font-size:24px}.card{padding:16px}}'''
    body = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逆向研究 v6 · 联合开仓检验</title><style>'+css+'</style></head><body><main>'
    body += '<div class="eyebrow">策略逆向研究 · 2026-10-04 · 第六轮</div><h1>联合学习开仓时机和选币，仍未通过检验</h1>'
    body += '<p class="sub">固定两种模型、三个时间段；使用已闭合行情，预测每个币下一分钟是否出现已记录的 SHORT 开仓。</p>'
    body += '<div class="card verdict"><b>本轮没有合格的开仓模块。</b><p>线性模型和浅层树均未在主口径中匹配任何真实开仓。浅层树的概率评分略优于 v5 组件组合，但改善不足以满足门槛，也未转化为可用的开仓信号。</p></div>'
    body += '<div class="stats"><div class="stat"><b>94 次</b><span>评估区间真实开仓，全部保留评分</span></div><div class="stat"><b>0 次</b><span>两个候选各自在 ±60 秒内命中</span></div><div class="stat"><b>0 / 3 段</b><span>两个候选各自通过的评估段</span></div></div>'
    body += '<h2>这次解决哪个问题</h2><p>第五轮的选币模型在给定真实开仓时刻后有一定排序能力，但自主开仓时机和仓位管理均未通过检验。本轮检验：把每个币的“会不会在这个分钟开仓”直接建模，能否将选币和时机连接起来。</p>'
    body += '<p>所有行情可用、没有开仓的币种分钟都作为负例，包括交易员当时实际持仓的分钟。模型只使用 14 个公开行情特征及 UTC 日内时钟；预测不读取交易员真实空仓、持仓、开仓时刻或未来平仓记录。</p>'
    body += '<p>这是开仓报警模块：信号选择当前概率最高的币，阈值由较早校准段确定，每日数量上限按时间顺序执行。重复报警会计入误报。没有仓位管理、成交模拟或收益计算。</p>'
    body += '<h2>汇总结果</h2><p>主口径要求币种一致、一对一匹配，名义信号时间距真实成功更新不超过 60 秒，且真实事件不得早于信号发出。默认延迟 1 分钟；±5 分钟仅作敏感性检查。</p>'
    body += table(['方法', '命中 TP', '误报 FP', '漏报 FN', '±5 分钟命中', '通过段数'], summary)
    body += '<p>线性模型共发出 68 次报警，浅层树 110 次；两者均漏掉全部 94 次开仓。扩大匹配窗口至 5 分钟，两个新候选仍没有匹配成功。</p>'
    body += '<h2>分段结果：概率更接近，不代表信号可用</h2>'
    body += table(['区间', '方法', 'TP / FP / FN', '精确率 / 召回率', '每币每分钟 log loss ↓', '预测/实际事件总量', '每日报警上限'], detailed)
    body += '<p>浅层树相对 v5 组件组合的 log loss 改善依次为 3.68%、0.79%、1.90%，均小于事先规定的 5%；按 UTC 日成组计算的探索性区间均包含零。第一段线性模型预测的事件总量约为实际的 6.08 倍，显示概率校准未能跨时间段保持。</p>'
    body += '<details><summary>查看两种参照的概率改善及不确定区间</summary><p>每次整组抽取 UTC 日，共 500 次；负值表示候选更差。历史已经多轮研究，区间只描述本轮分段波动。</p>'
    body += table(['区间', '候选', '参照', 'log loss 相对改善', '日成组重采样 95% 区间'], gains)+'</details>'
    body += '<h2>仍能识别部分选币偏好，时机没有解决</h2><p>额外给定真实开仓分钟后，v5 行情排序的 Top1 合计命中 37/94，联合浅层树命中 36/94。这说明本轮仍观察到一定条件排序信息；给定真实时刻的结果不能作为自主报警成绩。</p>'
    body += '<details><summary>查看各段条件选币排序</summary>'+table(['区间', '方法', 'Top1 命中', 'Top3 命中', '真实开仓数'], ranking)+'</details>'
    body += '<p>改为 2 分钟延迟，两个候选仍无主口径命中。去掉各评估段开仓最多的币，候选也未出现事件 F1 改善。仅忽略币种看时间，浅层树第一段有 1 次匹配，其余候选段均为 0。</p>'
    body += '<details><summary>查看延迟及集中度检查</summary>'+table(['区间', '方法', '1 分钟延迟命中', '2 分钟延迟命中', '仅时间匹配', '去最大币 F1', '剔除币种'], sensitivity)+'</details>'
    body += '<h2>固定范围与推进判定</h2><p>线性模型使用固定正则参数；浅层树固定深度 2、100 轮。每段保留全部拟合正例，最多抽取 10 万负例并按入选概率还原权重；使用独立时间校准段修正截距。共拟合 6 个模型、做 6 次截距校准，没有超参数搜索。</p>'
    body += '<p>参照一是已有时钟概率均匀分配给当前可用币；参照二是已有 v5 时机概率乘以归一化选币分数。参照二是本轮定义的组件组合，不能把它的报警结果等同于 v5 的原自主策略结果。</p>'
    body += '<p>同一个候选至少两段同时满足：相对两个参照的 log loss 均改善 ≥5%；预测事件总量为实际的 0.8–1.25 倍；币种及时间匹配的精确率和召回率均 ≥20%；剔除最大币种后 F1 仍优于两个参照。两个候选均为 0/3，按冻结协议结束这条比较。</p>'
    body += table(['区间', '校准起点 UTC', '评估起点 UTC', '评估终点 UTC', '拟合正例', '校准正例', '拟合币种分钟'], periods)
    body += '<h2>验证与证据边界</h2><p>独立检查从原始面板和成功记录重建标签、可用币池及时间切分；用保存的线性系数和树节点重算概率，以另一套二分图匹配方法复核事件成绩，并按分钟重放全部报警。</p>'
    body += f'<p>已通过 {v["replayed_alert_scenarios"]} 组场景、{v["replayed_alerts"]} 次报警和 {v["independently_recomputed_probability_rows"]:,} 行概率复算；7 项信息边界与计数测试通过。原有 {v["protected_original_files"]} 个文件的内容哈希全部未变。</p>'
    body += '<p>样本仍为此前已研究的 33 个币，行情和记录截至 2026-10-02 11:27 UTC。币池是事后选择的，成功更新时间也不能证明当时已被外部实时接收。当前没有新的未见样本，也没有合格的完整仓位管理候选。</p>'
    body += '<h2>下一步需要什么证据</h2><p>本轮结果支持暂时结束这两种联合模型的历史搜索。继续研究的优先方向是取得同一来源的新成功记录，并同时保存实际首次可见时间；使用已冻结的预测方式检查排序和时机能否在新记录上延续。若研究限价与撤单机制，还需要原始挂单、撤单和未成交生命周期记录。当前数据无法回答这些问题。</p>'
    body += '<h2>复查文件</h2><div class="links">'
    for name in ['protocol.json', 'results.json', 'verification.json', 'models.json', 'predictions.json', 'source_hashes.json', 'run.log']:
        body += f'<a href="{name}">{name}</a>'
    body += '</div><p class="small">复算：<code>PYTHONDONTWRITEBYTECODE=1 python3 analysis/verify_joint_entry.py</code>。该检查不重新拟合。</p>'
    body += '<p class="small">模型实现参考：<a href="'+p['method_reference']+'">scikit-learn 1.6 浅层梯度提升分类器文档</a>。本机实际版本 1.6.1；自动提前停止已关闭，避免随机验证切分参与时间研究。</p>'
    body += f'<p class="small">冻结协议 SHA256：{r["protocol_sha256"]}</p></main></body></html>'
    (OUT/'report.html').write_text(body)
    manifest = {name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['protocol.json', 'results.json', 'verification.json', 'report.html']}
    manifest['report_source'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'report_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(OUT/'report.html')


if __name__ == '__main__':
    main()
