"""Build a static, local report from the fixed v5 result and verification files.

No model fitting, feature loading, or experiment changes occur here.
"""
from pathlib import Path
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'entry_management_v5'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def esc(value):
    return html.escape(str(value))


def num(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def pct(value):
    return '—' if value is None else f'{100 * value:+.2f}%'


def match(value):
    return '/'.join(str(value[k]) for k in ('tp', 'fp', 'fn'))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + esc(x) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in row) + '</tr>'
        for row in rows
    ) + '</tbody></table></div>'


def section(title, body):
    return '<section><h2>' + esc(title) + '</h2>' + body + '</section>'


def main():
    results, protocol, audit, verification = map(read, (
        'results.json', 'protocol.json', 'observation_audit.json', 'verification.json'))
    assert verification['status'] == 'passed_independent_software_and_result_checks'
    assert results['protocol_sha256'] == verification['protocol_sha256']
    assert hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest() == results['protocol_sha256']
    folds = results['folds']
    sections = []
    dates = []
    for f in protocol['folds']:
        dates.append([f['name'], f['train_start'], f['calibration_start'], f['train_end'], f['eval_end']])
    sections.append(section('固定协议与数据口径',
        '<p>本轮由本地实现和检验产生，以下数值与结论不是 Gemini 原文。原始目标是逼近新行情中的交易与持仓行为；本轮只检验当前有限模型是否做到。</p>'
        '<p>复用事后选出的 33 个币种、公开历史行情及已保存的成功更新记录。时间标签是成功记录的 orderUpdateTime，无法据此还原下单意图、撤单、未成交挂单、Maker 身份或账户仓位。假设成功更新可以立即观察；真实接收时刻尚无证据。</p>'
        f'<p>可识别 SHORT 主样本 {audit["raw_primary_events"]} 条操作，{audit["raw_primary_openings"]} 条开仓，位于 {audit["opening_minutes"]} 个分钟；另有一个右端推断开仓，只作单独敏感性检查。评估区间共 94 个真实新开仓、273 条原始操作（88/100/85），较 v4 的 90/247 口径扩大；所有比较基准均在 v5 同一口径重算，不能把版本之间的数值变化归因于方法进步。</p>'
        '<p>全部历史已经看过，以下是按时间分段的探索性检验，不能称为新的未见样本。拟合/校准/评估严格按分钟分离，不根据未来平仓时间把整条周期从评估中删掉。</p>'
        + table(['区间', '拟合起点 UTC', '校准起点 UTC', '评估起点 UTC', '评估终点 UTC'], dates)))

    timing_rows = []
    for f in folds:
        clock, market = f['timing']
        timing_rows.append([f['fold'], market['probability']['events'], num(clock['probability']['log_loss'], 6),
                            num(market['probability']['log_loss'], 6), pct(f['gates']['timing_log_loss_gain']),
                            num(market['probability']['event_mass_ratio']), match(market['primary']),
                            match(market['sensitivity_5m']), num(market['threshold'], 6), market['daily_cap']])
    sections.append(section('独立开仓时机：仅第二段命中 2 次',
        '<p>每个公开特征齐全的分钟，都预测下一分钟是否出现至少一个 SHORT 开仓。参考模型只用 UTC 日内时钟；候选加 10 个公开横截面行情统计。真实持仓、真实开仓分钟和标签排除条件不能决定线上分钟或候选币池。未交易分钟作为自然负样本，抽样使用逆概率权重还原原始频率。</p>'
        '<p>阈值仅由校准段分位数和拟合段发生率确定。每日数量上限在线执行；不根据当天未来分数选 TopK。发出信号后至少等待一分钟执行；匹配窗口即使扩大，也禁止匹配发出信号之前已发生的真实事件。</p>'
        + table(['区间', '真实开仓分钟', '时钟 log loss', '候选 log loss', '候选相对改善', '预测/真实事件质量', 'TP/FP/FN ±60s', 'TP/FP/FN ±5m', '阈值', '每日上限'], timing_rows)
        + '<p>三个区间均未同时达到：log loss 改善≥5%、事件质量比 0.8–1.25、精确率和召回率均≥20%。第二段的两个命中仍对应 48 次误报和 29 次漏报。</p>'))

    rank_rows = []
    for f in folds:
        for x in f['rank_diagnostic']:
            if x['part'] == 'eval':
                rank_rows.append([f['fold'], '行情排序' if x['selector'] == 'market_rank' else '训练币种频率',
                                  x['events'], x['top1'], x['top3'], '给定真实开仓分钟'])
    sections.append(section('选币诊断：存在排序信息，未解决何时开仓',
        '<p>训练期真实开仓时刻可以作为排序模型的监督样本；自主运行时，只在自身开仓警报发生时对自身空仓币排序。下表额外给定真实开仓分钟，因此是条件诊断，不能作为可执行策略成绩。</p>'
        + table(['区间', '排序模型', '真实开仓数', 'Top1 命中', 'Top3 命中', '额外已知信息'], rank_rows)
        + '<p>每分钟最多开一个币的解码器无法表达同一分钟两个币均开仓。原始 147 次开仓中有一组双币同时开仓，位于拟合段；没有把其中一笔从真实事件集中删掉。</p>'))

    manager_rows = []
    names = {'duration': '拟合段中位时长退出', 'state_clock': '自有基础状态', 'market_state': '行情＋自有扩展状态'}
    for f in folds:
        for x in f['management']:
            if x['extra_delay'] == 0:
                manager_rows.append([f['fold'], names[x['variant']], x['episodes'], num(x['path']['normalized_error']),
                                     num(x['quantity_flow']['relative_gross_flow_error']), match(x['primary']),
                                     x['path']['residual_held_episodes']])
    sections.append(section('给定真实初始开仓的管理：未稳定胜过简单时长退出',
        '<p>每条诊断周期只提供一次真实首次开仓价、量和时间，随后用自有账本决定加仓、减仓、平仓。真实平仓不能结束模拟、纠正仓位或过滤决策。观察窗口固定为开仓后 48 小时或区间终点，先到者为止；真实周期超过 48 小时的尾部不在本检查范围。</p>'
        '<p>基础参考模型仅用 5 个自有状态；候选用 14 个公开行情特征和 10 个自有状态。加减平共用发生概率和竞争动作概率；没有追加 30 分钟冷却、最多加仓两次或事后交易预算。自然未操作的持仓分钟保留为负样本。</p>'
        + table(['区间', '管理方法', '新开仓周期', '持仓路径误差↓', '动作量流误差↓', 'TP/FP/FN ±60s', '期末残余持仓周期'], manager_rows)
        + '<p>路径误差是按拟合段中位开仓报价金额缩放的逐分钟绝对持仓差之和，除以真实持仓暴露之和；零真实暴露周期的虚假持仓仍计入分子。误差 1 相当于全程空仓参考。量流误差按币/诊断周期和动作分开，以固定、不重叠的五分钟箱统计总操作量；同分钟开平不能净额抵消。</p>'
        + '<p>候选未在任何区间同时达到：路径误差≤0.5、比两个参考均改善≥20%、加/减/平各自精确率与召回率≥20%、至少 30 条管理操作及 15 个周期。拟合中位持有时长为 '
        + ' / '.join(num(f['fit_median_duration_minutes'], 1) for f in folds) + ' 分钟。固定时长退出也很差，不能把它视为还原成功。</p>'))

    combo_rows, action_rows, delay_rows, cost_rows = [], [], [], []
    for f in folds:
        for x in f['combinations']:
            if x['extra_delay'] == 0:
                combo_rows.append([f['fold'], x['id'], num(x['path']['normalized_error']),
                                   num(x['quantity_flow']['relative_gross_flow_error']), match(x['primary']),
                                   x['initial_held_symbols'], x['path']['residual_symbols']])
                if x['id'] == 'candidate':
                    for action, m in x['events_by_action'].items():
                        action_rows.append([f['fold'], action, match(m), num(m['precision']), num(m['recall'])])
                    path = x['path']
                    cost_rows.append([f['fold'], num(path['virtual_turnover_quote']),
                                      num(path['fee_stress_quote']['5bps']), num(path['fee_stress_quote']['10bps'])])
            if x['id'] == 'candidate':
                delay_rows.append([f['fold'], '自主组合', 1 + x['extra_delay'],
                                   num(x['path']['normalized_error']), match(x['primary'])])
        for x in f['management']:
            if x['variant'] == 'market_state':
                delay_rows.append([f['fold'], '条件管理', 1 + x['extra_delay'],
                                   num(x['path']['normalized_error']), match(x['primary'])])
    sections.append(section('自主组合：三个区间均未匹配真实操作',
        '<p>参考组合＝时钟开仓＋训练币种频率＋中位时长退出；候选组合＝公开行情时机＋行情选币＋行情/自有状态管理。每段开始只初始化一次可重建的历史边界仓位（每段 1 个持仓币），之后不读真实状态。全空仓起始敏感性未纳入本轮冻结协议，未在看过结果后追加；此结果依赖边界初始化假设。</p>'
        '<p>开仓数量来自拟合段中位开仓报价金额除以当前已闭合价格；加仓用拟合加仓量/首次量中位比例；减仓用拟合减仓量/操作前量中位比例；平仓为自有剩余量。用于数量的价格不会取未来执行价。所有状态只在自有虚拟执行后更新。</p>'
        + table(['区间', '组合', '持仓路径误差↓', '动作量流误差↓', 'TP/FP/FN ±60s', '一次初始持仓币', '末期残余币'], combo_rows)
        + table(['区间', '候选动作', 'TP/FP/FN ±60s', '精确率', '召回率'], action_rows)
        + '<p>候选路径误差相对参考的改善分别为 '
        + ' / '.join(pct(f['gates']['combined_path_gain']) for f in folds)
        + '；移除各段操作最多的币后仍为 '
        + ' / '.join(pct(f['gates']['combined_path_gain_without_largest']) for f in folds)
        + '。所有阶段和联合门槛均为 0/3。</p>'))

    sections.append(section('执行延迟与费用边界',
        '<p>冻结协议分别使用决策后 1/2 分钟的公开分钟开盘价虚拟执行。额外一分钟延迟没有让门槛通过。这只改变虚拟执行延迟，不代表已验证成功记录的真实接收延迟。</p>'
        + table(['区间', '候选部件', '总虚拟执行延迟/分钟', '持仓路径误差↓', 'TP/FP/FN ±60s'], delay_rows)
        + '<p>以下仅对虚拟执行成交额乘 5/10 bps 计算费用情景。没有真实盘口撮合、Maker 费率、滑点、资金费率、账户权益或收益率回测。</p>'
        + table(['区间', '候选虚拟总成交报价金额', '5 bps 费用', '10 bps 费用'], cost_rows)))

    group_rows = [[x['panel_row'], ' / '.join(x['actions']), ' / '.join(x['event_ids']),
                   num(x['gross_quantity']), num(x['net_signed_quantity'])]
                  for x in audit['same_symbol_multiple_groups']]
    sections.append(section('未省略的歧义与截断',
        '<p>同币同分钟多条成功操作只从不兼容的拟合/概率校准标签中排除；原始记录全部留在事件与量流评分中。主样本有 4 组、8 条这类记录，包括分钟内开平。第一决策时刻之前发生的同分钟管理操作无法在线预测，保留为漏报，不能从 FN 中消失。</p>'
        + table(['面板行', '原始动作', '原始记录 ID', '总量', '带方向净量'], group_rows)
        + '<p>右端推断 BTWUSDT 开仓仅在第三段单独模拟：给定真实首次价量后原地持有 638 分钟、没有真实后续操作，持仓误差为 0。这个退化且右删失的条件结果不计入主门槛，也不证明平仓或开仓预测能力。记录可见性、完整候选币池、历史边界和未知初始仓位仍有未验证假设。</p>'))

    sections.append(section('实现修复与独立验证',
        '<p>首次实现错误地让“右端推断开仓标签的排除条件”参与线上 timing_ready，从而可能跳过依赖未来标签的分钟。已将纯公开 readiness 与只用于训练/评分的 label mask 分开；第一次源码、结果、预测、系数及日志完整保存在 implementation_attempt_1。修复后按原协议重跑一次，没有新增配置。</p>'
        '<p>最终固定 21 个学习头、15 次截距校准、0 次超参数搜索；计入修复前后，实际执行 42 个拟合头。独立验证脚本不导入本轮模型/runner/匹配器：由保存的系数复算概率，独立事件匹配、账本路径、量流、公开警报和组合选币，按原始成功操作抽查边界仓位。</p>'
        f'<p>验证通过 {len(verification["checks"])} 组组件/延迟场景、{verification["decoder_openings_checked"]} 次组合开仓、{verification["management_issue_quantities_checked"]} 次管理数量、{len(verification["source_spot_checks"])} 条源记录抽查；29 项单元测试通过。原有 {verification["protected_original_files"]} 个文件哈希全部未变。验证确认软件和报告算术一致，不能将其称为独立新行情上的行为验证。</p>'
        '<p>未进一步扩展模型搜索；未实盘、上传原始数据、购买、发布、推送代码，也未修改原有分析文件。</p>'))

    links = [(name, name) for name in ('protocol.json', 'results.json', 'verification.json', 'observation_audit.json',
                                      'models.json', 'predictions.json', 'scores.npz', 'source_hashes.json', 'run.log')]
    links += [('implementation_attempt_1/repair.json', '首次实现与修复原因'),
              ('../../entry_management.py', '模型与独立解码器'),
              ('../../run_entry_management.py', '固定协议运行器'),
              ('../../verify_entry_management.py', '独立验证器'),
              ('../../tests/test_entry_management.py', '13 项新增测试'),
              ('../../build_entry_management_report.py', '报告生成器')]
    sections.append(section('复查入口与可重复执行',
        '<p>' + ' · '.join('<a href="' + esc(path) + '">' + esc(label) + '</a>' for path, label in links) + '</p>'
        '<pre>cd /Users/stellan/策略逆向建模\n'
        'PYTHONDONTWRITEBYTECODE=1 python3 analysis/verify_entry_management.py\n'
        'PYTHONDONTWRITEBYTECODE=1 python3 analysis/build_entry_management_report.py</pre>'
        '<p>验证和生成报告不重新拟合。重新运行 study 会覆写本轮可再生成的结果，协议及首次实现记录应继续保存；当前成果不需要额外拟合。</p>'
        '<p class="mono">协议 SHA256：' + esc(results['protocol_sha256']) + '</p>'))

    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>策略逆向建模 v5：开仓与管理拆分检验</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#172431;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}
main{max-width:1240px;margin:auto;padding:36px 24px 64px}header,section{background:white;border:1px solid #dce3e9;border-radius:12px;margin-bottom:22px;padding:24px 28px}
h1{font-size:30px;line-height:1.35;margin:6px 0 18px}h2{font-size:22px;margin:0 0 14px}p{margin:10px 0}.label{color:#56697b;font-size:14px}.verdict{border-left:5px solid #b14636;padding-left:16px;font-size:19px}
.scroll{overflow:auto}table{border-collapse:collapse;min-width:660px;width:100%;font-size:14px;margin:16px 0}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #e3e8ed;vertical-align:top}th{background:#eef3f7;color:#344b61}tr:nth-child(even){background:#fafbfd}a{color:#176493;text-decoration:underline;text-underline-offset:3px}pre{overflow:auto;background:#eff3f6;border-radius:8px;padding:16px;font-size:13px}.mono{font-family:monospace;overflow-wrap:anywhere;font-size:12px}
@media(max-width:700px){main{padding:18px 12px}header,section{padding:20px 16px}h1{font-size:25px}}
</style></head><body><main><header><p class="label">本地固定实验 · v5 · 历史截止 2026-10-02 11:27 UTC</p>
<h1>开仓时机和开仓后的管理都尚未学会</h1>
<p class="verdict">已完成拆分实验、自主组合与独立复算。开仓、管理、组合以及联合门槛全部 0/3。真实开仓时刻下的选币排序仍有信息，但当前方法无法独立逼近交易与持仓行为。</p>
<p>下一阶段的必要条件是冻结实现之后，获得同源公开新记录并持续留存实际接收时刻；当前历史结果不足以支持扩大模型或实盘。所有本轮结论限定于本次模型、口径和观察假设，不能据此断言行为建模原则上不可行。</p></header>'''
    document += ''.join(sections) + '</main></body></html>\n'
    (OUT / 'report.html').write_text(document, encoding='utf-8')
    meta = {'report_sha256': hashlib.sha256(document.encode()).hexdigest(),
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_sha256': {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in
                             ('results.json', 'protocol.json', 'verification.json', 'observation_audit.json')},
            'presentation_only': True, 'fits_performed': 0, 'visual_render_not_performed': True}
    (OUT / 'report_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Built', OUT / 'report.html')


if __name__ == '__main__':
    main()
