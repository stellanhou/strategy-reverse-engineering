"""Build the readable report and acceptance record for this research round."""
import json
from html import escape
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'results'


def read(name):
    return json.loads((OUT / name).read_text())


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(str(c))}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def main():
    panel = read('reverse_panel_meta.json')
    mechanism = read('reverse_mechanisms.json')
    controls = read('reverse_controls.json')
    rules = read('reverse_rules.json')
    models = read('reverse_models.json')
    audit = read('reverse_audit.json')
    validation = read('reverse_validation.json')
    assert rules['status'] == 'completed_exploratory_diagnostics'
    assert models['status'] == 'completed_exploratory_single_step_comparison'
    assert validation['panel_rebuild_semantic_identity']
    assert panel['exclusions']['causal_close_audit_violations'] == 0
    assert all(f['modes']['autonomous']['f1_difference'] < .05 for f in rules['folds'])
    stages = [
        ['0', '研究口径', '完成', 'UTC分钟网格、空头主线、三折日历、候选预算、事件一对一匹配已固定。'],
        ['1', '数据和标签', '条件范围完成', '33个观察币种补齐行情，建立自然频率负样本和可追溯状态；全历史候选币池仍缺失。'],
        ['2', '机制和对照', '完成', '用完整无操作分钟验证入场强势、逆向加仓和盈利侧平仓；有分段区间与反例。'],
        ['3', '完整简单规则', '完成，未通过筛选', '30套确定规则完成单步及自主回放；不预测分批减仓。'],
        ['4', '可解释模型', '完成，预算执行未通过', '按动作计数主比较20个模型配置；初跑超额4个配置保留并排除。没有自主策略。'],
        ['5', '历史筛选', '完成本轮判定', '没有符合推进门槛的完整候选；保留弱减仓线索；预算超额单列。'],
        ['6', '资金和策略账本', '未开始', '依赖阶段5的合格完整候选；当前没有候选进入成本、资金费与风险验收。'],
        ['7', '新增数据正式验收', '未开始', '尚无冻结候选及T0；需要冻结后新增的30—60日数据。'],
    ]
    acceptance = {
        'round': 'reverse-research-1', 'status': 'historical_round_complete_budget_failed_no_qualified_full_candidate',
        'stages': stages,
        'data_correctness': 'observable conditional panel and source reconciliation passed; exact partial-fill lifecycle and full historical universe unavailable',
        'behavior_explanation': 'relative-hour strength, adverse held return before adds, and profitable close placement are reproducible associations in the conditional historical panel',
        'behavior_reproduction': 'not_passed; autonomous registered candidates fail the +0.05 development improvement target and do not improve consistently',
        'local_reduce_model': 'insufficient_evidence; 24 usable SHORT MARKET reduce events, first two fold gains have 95% interval lower bound zero, third fold no gain',
        'independent_strategy': 'not_evaluated_no_qualified_candidate',
        'prospective_validation': 'not_started_T0_null',
        'configuration_count_contract': 'Action-specific learning fits are configurations. Primary comparison is original declaration order capped at ten per family, 30 rules plus 20 action models. Initial run executed four excess close models; original results preserved and excluded from primary selection. Fold refits are not new configurations.',
        'configurations_initially_executed': 54,
        'configurations_primary': 50,
        'execution_budget_pass': False,
        'limitations': ['full historical universe missing', 'only successful observed LIMIT orders, no canceled/unfilled lifecycle',
                        'leader closingPnl differs from price cashflow by 4.83330835 USDT with unobserved components',
                        'all inspected history remains exploratory', 'no account simulation or net return result'],
        'next_decision': 'do not freeze or launch a strategy from this round; further research should resolve LIMIT placement data and local reduce evidence in a new version with new observations',
    }
    (OUT / 'reverse_acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + '\n')
    short = mechanism['short_mechanics']['event_mechanics']
    test_labels = ['15m return before open', '60m relative BTC before open', 'Adverse return before add', 'Profit return before MARKET close']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, test, label in zip(axes.flat, controls['tests'], test_labels):
        d = test['full_exploratory_calendar']
        ax.bar(['No-action control', 'Action minute'], [100 * d['control_median'], 100 * d['positive_median']], color=['#9fb2bc', '#1e7186'], width=.55)
        ax.axhline(0, color='#37474f', lw=.6)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel('Median score (%)')
        ax.text(.03, .96, f"events={d['positive_minutes']:,}; controls={d['control_minutes']:,}", transform=ax.transAxes, va='top', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Conditional historical evidence — 33 observed symbols', fontsize=15)
    fig.savefig(OUT / 'reverse_evidence.png', dpi=170)
    plt.close(fig)
    evidence_rows = []
    labels_cn = ['开空前15分钟涨幅', '开空前60分钟相对BTC涨幅', '加仓前浮亏幅度', '市价平仓前浮盈幅度']
    for label, test in zip(labels_cn, controls['tests']):
        d = test['full_exploratory_calendar']
        ci = d['1_day_block_95_interval']
        evidence_rows.append([label, f"{d['positive_minutes']:,} / {d['control_minutes']:,}",
                              f"{100*d['positive_median']:.2f}%", f"{100*d['control_median']:.2f}%",
                              f"{100*ci[0]:.2f}—{100*ci[1]:.2f}个百分点", f"{test['positive_direction_folds']}/{test['valid_folds']}"])
    rule_rows = []
    for f in rules['folds']:
        for mode, title in [('single_step', '给定真实状态'), ('autonomous', '自主连续回放')]:
            r = f['modes'][mode]; c, b = r['candidate_evaluation'], r['baseline_evaluation']
            ci = r['paired_f1_difference_interval_exploratory']['1_day_block_95_interval']
            rule_rows.append([f['name'], title, f['evaluation_events'], f"{c['tp']} / {c['fp']} / {c['fn']}",
                              f"{c['f1']:.5f}", f"{b['f1']:.5f}", f"{r['f1_difference']:+.5f}",
                              f"[{ci[0]:+.5f}, {ci[1]:+.5f}]", f"{c['false_positives_per_symbol_hour']:.4f}"])
    model_rows = []
    names_cn = {'open': '开仓', 'add': '加仓', 'reduce': '减仓', 'close': '市价平仓'}
    for f in models['folds']:
        for action, r in f['modules'].items():
            ci = r['paired_f1_difference_interval_exploratory']['1_day_block_95_interval']
            model_rows.append([f['name'], names_cn[action], r['train_positive_events'], r['evaluation_positive_events'],
                               r['selected_by_training_only'], f"{r['selected_evaluation']['f1']:.4f}",
                               f"{r['f1_difference']:+.4f}", f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                               f"{r['selected_evaluation_ap']:.5f}", f"{r['ap_score_proxy_baseline']['evaluation_ap']:.5f}"])
    p = panel['exclusions']
    initial_median = mechanism['short_mechanics']['cycle_initial_notional']['median']
    top = mechanism['short_mechanics']['remove_largest_symbol_sensitivity']
    counters = mechanism['counterexamples']['short_adds_at_or_better_than_weighted_cost']
    counter_rows = [[r['symbol'], r['event_id'], f"{100*r['adverse_from_cost']:.2f}%", r['cycle_id']] for r in counters]
    money = audit['independent_position_replay']['pnl_reconciliation_usdt']
    coverage_fraction = p['rows_with_required_features_ready'] / (p['rows_with_known_state'] - 4725)
    body = f'''<header><div class="muted">2026-10-02 · 首轮历史研究 · UTC 2026-09-03 至 2026-10-02 11:27</div>
<h1>2026大黑马逆向研究结果</h1><p class="lead">发现了可重复的行为线索，尚未得到能稳定复现操作的完整规则。</p>
<div class="callout">当前结论：小额开空、逆向加仓、盈利侧平仓的画像得到支持；精准入场、减仓和挂单触发仍未恢复。阶段0—5已完成本轮研究，阶段6—7尚未进入。</div>
<nav><a href="#progress">执行进度</a><a href="#data">数据验收</a><a href="#evidence">机制证据</a><a href="#rules">规则回放</a><a href="#models">模型比较</a><a href="#decision">验收决定</a></nav></header>
<section id="progress"><h2>计划执行到了哪里</h2>{table(['阶段','对象','状态','交付与结论'],stages)}</section>
<section id="data"><h2>数据和时间口径</h2><div class="cards"><div><b>851,344</b><span>新增公开分钟行情</span></div><div><b>1,400,751</b><span>33合约 × 42,447分钟</span></div><div><b>1,343,814</b><span>可用空仓无操作分钟</span></div><div><b>302</b><span>有效市价创建标签</span></div></div>
<p>独立重放157轮完整周期，逐轮数量、峰值仓位及归零状态均通过。纳入描述的UTC完整周期为156轮，其中147轮空头、9轮多头。窗口开始前已开仓的一轮用于状态预热，完整周期统计排除。</p>
<p>有效市价标签：开仓153（空头144、多头9）、加仓87（80/7）、减仓30（24/6）、平仓32（29/3）。1条创建时间在窗口外、3条开仓与其他仓位变化同分钟的目标明确排除。145条LIMIT记录独立保留创建与更新两条时间线。</p>
<p>实际有效行情1,396,026条；PONS上市前4,725个日历行保留为空。上币后分钟网格无内部缺口；适用状态下特征覆盖约{100*coverage_fraction:.4f}%。60个收益率需要61个已收盘价格，相关预热缺口明确排除。特征未来时点违规为0，下一分钟开盘价只用于成交。两次重建数组内容完全一致。</p>
<p>CT未知仓位整币排除；BTW与龙虾两条未锚定开仓后的状态屏蔽。33币池来自已观察交易，结论限于这些币的条件时机；历史全市场选币尚无可追溯币池。没有把缺失或不适用状态填成零。</p>
<p>带单员价格现金流{float(money['replayed_order_price_cashflow_sum']):.8f} USDT，官方仓位盈亏{float(money['position_history_closingPnl_sum']):.8f} USDT，未解释差额4.83330835 USDT。该差额未用于拟合本策略成本。公开字段没有完整部分成交或挂撤单生命周期，更新时间只作为已执行数量的更新代理。</p></section>
<section id="evidence"><h2>哪些线索得到支持</h2><p>下表用所有合格无操作分钟作对照；数值为决策分钟前已收盘行情。均值差区间来自UTC日块重采样，3日块和最大币种剔除结果见数据文件。这些区间描述已经查看的历史。</p>
{table(['机制','动作 / 对照分钟','动作中位','对照中位','均值差95%区间','分段方向一致'],evidence_rows)}<img src="reverse_evidence.png" alt="四种行为特征与无操作分钟的中位数比较">
<p>按实际操作价格恢复持仓加权成本，空头80次加仓中78次逆向（97.5%）；逆向幅度中位{100*short['add_adverse_from_cost']['median']:.2f}%，距上次动作价格的不利间距中位{100*short['add_adverse_spacing_from_previous_event_price']['median']:.2f}%。首笔名义金额中位约{initial_median:.2f} USDT，加仓数量/首笔数量中位1倍、分布跨度大，未恢复固定倍投规则。</p>
<p>空头完整周期减仓价格相对加权成本的收益中位{100*short['exit_return_by_action']['reduce']['median']:.2f}%，最终平仓为+{100*short['exit_return_by_action']['close']['median']:.2f}%。减仓和最终退出不能共用一个固定止盈解释。移除{top['removed_symbol']}后，逆向加仓比例仍为{100*top['remaining_event_mechanics']['add_adverse_vs_weighted_cost_share']:.1f}%。</p>
<details><summary>逆向加仓解释的反例</summary>{table(['币种','事件','逆向幅度','周期'],counter_rows)}</details>
<p>15分钟涨幅关联在3段中的2段为正，但整体区间跨零；60分钟相对BTC强势在3段中方向一致。关联能缩小调查范围，尚不能确定那一分钟会操作。</p></section>
<section id="rules"><h2>30套简单规则的结果</h2><p>10套15分钟急涨开空、10套相对BTC一小时急涨开空，以及10套固定持仓时间/时间频率基准。每套规则只有固定加仓间距与次数、止盈或固定时间退出；候选和最强基准均只按各训练段选择。主匹配为同币、同方向、同动作一对一±60秒，±5分钟另存。没有预测分批减仓。</p>
{table(['分段','模式','实际动作','命中 / 误报 / 漏报','候选F1','基准F1','提升','提升95%区间','误报 / 币种小时'],rule_rows)}
<p>自主回放只读取行情和模型自己的仓位，从研究起点连续运行；没有用真实持仓重新接回轨道。跨切分周期和不可评分分钟只从分数中排除，不干预模型状态。模型成交于下一分钟开盘，刚成交边界不会用成交前收盘价触发假收益。当前候选没有达到+0.05的开发改善目标。</p>
<p>这些回放使用归一首笔数量1，未建立账户资金、费用、资金费与权益曲线，因此不能据此判断净收益或独立策略有效性。</p></section>
<section id="models"><h2>回归与浅树的局部比较</h2><p>使用3组正则化逻辑回归参数与深度1/2/3的浅树，分别拟合4个动作模块。按动作计数初跑24个模型配置，连同30套规则实际54个，超过50上限。按原登记顺序及每类最多10个动作配置，主比较保留20个模型；4个超额市价平仓配置排除出主选择，原结果完整保留。预算执行验收未通过。初跑存档头部的36是修正前错误计数，当前正确口径为54。</p>
<p>模型只在给定真实持仓状态下预测。开发段最多抽取20,000个负样本，评估保留自然事件率；标准化、动作阈值和模型选择均来自训练段。事件基准在已测30套规则及固定特征指标中按训练段选择，并使用相同的模块冷却；AP另与固定特征分数比较。</p>
{table(['分段','模块','训练正例','评估正例','训练选中模型','事件F1','相对事件基准提升','提升95%区间','AP','指标基准AP'],model_rows)}
<p>阈值在训练段按“最多两倍正例数量的原始警报额度”固定，之后每币每动作冷却30分钟；这是本轮诊断约束，未标定实际使用成本。浅树分数并列时采用严格大于阈值，不靠事后挑分钟制造精确命中。AP改善只反映排序，不能替代操作时间复现。</p>
<p>减仓前两段有少量命中，但配对提升区间下界均为0，第三段没有改善；24次空头市价减仓也未达到计划中30正例的最低类别覆盖。开仓、加仓和市价平仓的主事件F1没有稳定改善。</p></section>
<section id="decision"><h2>本轮验收和后续依赖</h2><ul><li><strong>数据链路：</strong>观察币种范围内通过；历史全币池、限价委托生命周期和精确部分成交仍缺。</li>
<li><strong>机制解释：</strong>保留一小时相对强势、逆向加仓、盈利侧平仓三条条件关联；保留减仓的局部模型线索。</li>
<li><strong>完整行为复现：</strong>未通过本轮开发筛选，没有冻结候选。</li><li><strong>独立策略与正式前瞻：</strong>未验收，阶段6—7未开始，T0为空。</li></ul>
<p>本轮主比较50个配置，实际初跑54个，额外4个市价平仓模型的预算偏差已记录。继续研究应优先补充限价委托价格及挂撤单信息、扩大减仓样本，并在下一版明确候选修改；现有历史不能改称新增验证。没有新增已验证策略，当前结果不支持启动实际交易。</p>
<details><summary>可复核文件和复跑顺序</summary><p>已有补数可直接复用，依次运行数据审计、面板构建、事件机制、无动作对照、规则回放、模型比较及报告生成。代码与产物均在本项目analysis目录。</p><p>
<a href="reverse_protocol.json">研究协议</a> · <a href="reverse_audit.json">独立数据审计</a> · <a href="reverse_panel_meta.json">特征与覆盖</a> · <a href="reverse_controls.json">机制对照</a> · <a href="reverse_mechanisms.json">事件与反例</a> · <a href="reverse_rules.json">全部规则配置与分数</a> · <a href="reverse_rule_predictions.json">规则逐时预测</a> · <a href="reverse_models.json">主比较模型、阈值与预测</a> · <a href="reverse_models_initial_54.json">初跑及超额记录</a> · <a href="reverse_validation.json">重建验证</a> · <a href="reverse_acceptance.json">验收记录</a></p></details>
<p class="muted">行情字段与公开获取方式参照<a href="https://github.com/binance/binance-public-data">币安官方公共数据说明</a>。数据接口和实际补数区间见<a href="reverse_market/manifest.json">补数来源记录</a>；该来源只支持行情数据定义，不支持本报告的模型结论。</p></section>'''
    style = '''body{margin:0;background:#f3f6f7;color:#203441;font:16px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1240px;margin:auto;padding:35px 22px 70px}h1{font-size:33px}h2{font-size:24px;margin-top:0}.lead{font-size:21px}.muted{font-size:14px;color:#637883}section{background:white;border:1px solid #d8e2e8;border-radius:12px;padding:26px;margin:22px 0}.callout{background:#e4f1f3;border-left:4px solid #22758a;padding:18px}nav{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}a{color:#16647b}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.cards div{padding:14px;background:#edf4f6;border-radius:8px}.cards b{display:block;font-size:27px}.cards span{font-size:13px;color:#5e7380}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.65}th,td{padding:10px 9px;text-align:left;border-bottom:1px solid #e1e9ed;vertical-align:top}th{background:#edf3f5;white-space:nowrap}img{max-width:100%;height:auto}summary{cursor:pointer;color:#16647b}details{margin:17px 0}@media(max-width:680px){main{padding:18px 10px}section{padding:17px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:27px}}'''
    (OUT / 'reverse_report.html').write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026大黑马逆向研究结果</title><style>{style}</style></head><body><main>{body}</main></body></html>')
    print('Report and acceptance saved:', OUT / 'reverse_report.html')


if __name__ == '__main__':
    main()
