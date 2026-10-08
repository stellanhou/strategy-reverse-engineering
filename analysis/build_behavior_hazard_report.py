"""Presentation-only report from frozen saved results; no model fitting."""
from pathlib import Path
import json
import html

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/results/behavior_hazard_v4'


def table(head,rows):
    cell=lambda x:html.escape(str(x))
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+cell(x)+'</th>'for x in head)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'


def main():
    read=lambda n:json.loads((OUT/n).read_text())
    r,p,v,d=map(read,['results.json','protocol.json','verification.json','diagnostics.json'])
    text='<h1>策略逆向建模：第四轮受控实验</h1><p class="sub">固定33币 · SHORT成功更新/成交代理 · 历史探索</p>'
    text+='<div class="verdict"><b>未通过推进；已停止扩大配置。</b><p>软件核验通过不代表行为近似通过。本轮不计算账户收益，不主张恢复委托、撤单、Maker身份或原代码。</p></div>'
    text+='<p>经<a href="'+p['pro_review_url']+'">ChatGPT网页6 Pro方法审阅</a>后，只比较状态/时钟基准与增加原有8项行情的分层事件模型。所有旧历史已经看过；三段结果仍是探索，不能作为未见行情验证。</p>'
    text+='<h2>冻结配置与判定</h2><p>2种结构、3折、3个固定执行延迟，共18回放；无超参数搜索。自然负样本下采样以逆包含权重还原频率，不再class_weight平衡。开仓概率和持仓操作hazard各仅校准1个截距；加/减/平仓共享事件风险并竞争动作。</p>'
    text+='<p>推进要求：概率损失改善至少5%、事件概率总量比0.8～1.25、每动作精确率/召回率各至少20%、自主路径误差≤0.5且比基准改善至少20%、去最大事件币仍改善，至少2折同时满足。本次为0/3折。</p>'
    rows=[]
    for f in r['fold_results']:
        b,c=f['variants'];bp,cp=b['probability']['pooled'],c['probability']['pooled']
        rows.append([f['fold'],f['evaluation_minutes'],f['evaluation_events'],f"{bp['log_loss']:.7f} / {cp['log_loss']:.7f}",
            f"{c['probability_comparison']['relative_log_loss_improvement']:+.2%}",
            f"{bp['brier']:.7f} / {cp['brier']:.7f}",f"{bp['event_mass_ratio']:.3f} / {cp['event_mass_ratio']:.3f}",
            '未通过'])
    text+=table(['时间段','自然风险分钟','真实事件','任意成功操作log loss 基准/市场','市场改善','Brier 基准/市场','概率事件总量比 基准/市场','结论'],rows)
    text+='<p>“概率事件总量”是每分钟概率之和，不是实际发出的动作数。日块置信区间只是探索性不确定性；不会将百万分钟当成独立行为样本。</p>'
    text+='<h2>动作与连续自主路径</h2><p>±60秒一对一主匹配额外要求真实事件不早于预测发行时间。执行价格只在虚拟操作到期时进入自己的成本账本；同一分钟到期执行在本分钟决策后应用。自主预测不输入真实持仓。</p>'
    rows=[]
    for f in r['fold_results']:
        for t in f['variants']:
            a=t['autonomous'][0];s=a['all_events'];path=a['path']
            rows.append([f['fold'],t['variant'],f"{s['tp']} / {s['fp']} / {s['fn']}",
                f"{s['precision']:.2%} / {s['recall']:.2%}",f"{a['fp_per_33_symbol_calendar_day']:.2f}",
                f"{path['normalized_absolute_error']:.3f}",f"{path['held_overlap_iou']:.3f}",path['residual_open_symbols']])
    text+=table(['段','结构','自主TP/FP/FN','精确率/召回率','每天33币FP','路径误差','有仓重叠IoU','期末剩余币'],rows)
    text+='<p>持续空仓的路径误差为1。本轮市场模型的主路径误差均大于1；尽管比很差的状态/时钟自主基准改善很多，依然不能称为成功。路径按拟合期首笔名义金额中位数归一化，覆盖双方仓位不一致时间，不按评估期首笔数量缩放。</p>'
    rows=[]
    names={'open':'开仓','add':'加仓','reduce':'减仓','close':'平仓'}
    for f in r['fold_results']:
        for a in names:
            b,c=f['variants'];x,y=b['local_events'][a],c['local_events'][a]
            z=c['autonomous'][0]['events_by_action'][a]
            show=lambda q:'/'.join(str(q[k]) for k in ('tp','fp','fn'))
            bp,cp=b['probability'][a],c['probability'][a]
            rows.append([f['fold'],names[a],show(x),show(y),show(z),
                         f"{bp['log_loss']:.6g} / {cp['log_loss']:.6g}",
                         f"{1-cp['log_loss']/bp['log_loss']:+.2%}",
                         f"{bp['event_mass_ratio']:.3f} / {cp['event_mass_ratio']:.3f}",
                         f"{cp['average_precision']:.6g}"])
    text+=table(['段','动作','单步基准TP/FP/FN','单步市场TP/FP/FN','自主市场TP/FP/FN','概率log loss 基准/市场','市场损失改善','概率事件总量比 基准/市场','市场AP'],rows)
    text+='<h2>数量单独验收</h2><p>以下误差涵盖全部合格真实事件，数量模块对两模型相同，不能把它当市场特征增益。开仓目标为决策前价格折算的名义金额；加仓为首次数量倍数，减仓为当前数量比例。平仓取自己剩余数量，是账本规则，不是学到的数量模型。</p>'
    rows=[]
    for f in r['fold_results']:
        for a in ('open','add','reduce'):
            s=f['quantity_fit'][a];q=f['quantity_evaluation'][a]
            rows.append([f['fold'],names[a],f"{s['n']} / {q['events']}",f"{s['median']:.4g}",
                         f"{q['relative_total_absolute_error']:.2%}",f"{q['q25_q75_empirical_coverage']:.2%}"])
    text+=table(['段','数量目标','拟合/评价事件','拟合中位数','总相对绝对误差','拟合中间50%区间覆盖'],rows)
    text+='<h2>阈值与在线额度</h2><p>每动作只有一个校准分位数，目标频率来自拟合期。每日硬额度=ceil(2×拟合事件数/拟合日数)，只在当前时刻排序，同分用固定哈希处理；不使用一天结束后的Top-K。额度可能漏掉新行情的真实事件，已照计FN。</p>'
    text+=table(['段','结构','开/加/减/平阈值','对应每日33币额度'],[
        [f['fold'],t['variant'],' / '.join(f'{x:.6g}'for x in t['thresholds']),' / '.join(map(str,t['daily_budgets']))]
        for f in r['fold_results'] for t in f['variants']])
    text+='<h2>固定延迟敏感性</h2><p>0/1/5为额外执行延迟，即总延迟1/2/6分钟。它与真实记录的接收/可见延迟不同，本轮没有验证后者。没有选择最好的延迟作为主成绩。</p>'
    text+=table(['段','结构','额外延迟','主TP/FP/FN','辅助±5分钟F1','路径误差'],[
        [f['fold'],t['variant'],a['delay_extra_minutes'],' / '.join(str(a['all_events'][k])for k in ('tp','fp','fn')),
         f"{a['sensitivity_5m']['f1']:.4f}",f"{a['path']['normalized_absolute_error']:.3f}"]
        for f in r['fold_results'] for t in f['variants'] for a in t['autonomous']])
    text+='<h2>失败定位：选币有条件信息，时机与自主路径仍失败</h2><p>给定真实开仓分钟的神谕诊断中，旧第三轮扩展分数Top3覆盖24/29、18/29、26/32，历史币种频率仅3/29、3/29、0/32。这是条件排序信息，输入了真实时机，不能部署为策略。完整自然频率加仓风险集按浮亏与持仓阶段保存在诊断文件，不能只看加仓正例。</p>'
    text+='<h2>独立核验与覆盖限制</h2><p>16项测试通过；独立脚本按原始记录重建标签、按二元事件风险系数复算主要概率指标、以二分图最大匹配复算事件、用执行流重建数量路径，并抽查12个源记录的操作前仓位。18回放结果一致，252个旧文件hash全部未变。</p>'
    text+='<p>同币同分钟8条记录属于4组，含开仓/平仓净变化为零、减仓/平仓、平仓/重开。保留原始动作、数量和组证据，明确排除，未压成无动作或单个净动作；另2条无锚操作排除。主评价仍复用完整可识别周期与边界剔除队列，未扩展左截断/右删失的未平周期，因此不是全部交易记录验收。</p>'
    text+='<p>LIMIT主目标只预测已保存成功操作，不虚构原挂价或Maker身份。未平周期扩展、0/1/5分钟记录可见性敏感性、给定真实开仓后的独立管理回放、真实执行资金费率和账户收益、冻结后的未见同源操作记录验收均未完成。私有订单生命周期与全市场历史增强不会阻塞此MVP。</p>'
    text+='<p>周转费用0/5/10bps仅为情景参数，未计算盈利。探索失败只否定这次有限路线的通过主张，不证明所有公开信息和模型永远无法逼近。</p>'
    text+='<h2>复现</h2><pre>cd /Users/stellan/策略逆向建模\nPYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s analysis/tests -v\nPYTHONDONTWRITEBYTECODE=1 python3 analysis/run_behavior_hazard.py\nPYTHONDONTWRITEBYTECODE=1 python3 analysis/verify_behavior_hazard.py\nPYTHONDONTWRITEBYTECODE=1 python3 analysis/build_behavior_hazard_report.py</pre>'
    text+='<p><a href="protocol.json">冻结协议</a> · <a href="results.json">完整结果</a> · <a href="diagnostics.json">风险集与旧分数诊断</a> · <a href="verification.json">独立核验</a> · <a href="predictions.json">发行与执行流水</a></p>'
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>策略逆向建模第四轮</title><style>body{font:16px/1.7 -apple-system,"PingFang SC",sans-serif;background:#f4f6fa;color:#243247;margin:0}main{max-width:1400px;margin:auto;padding:32px 24px}h1{font-size:30px;margin-bottom:0}h2{margin-top:32px;font-size:22px}.sub{color:#63738a}.verdict{background:#fff0dc;border-left:4px solid #bc6b17;padding:16px 22px;margin:22px 0}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:10px;border-bottom:1px solid #dfe4ee;text-align:left}th{background:#eaf0f8;white-space:nowrap}a{color:#245cad}pre{white-space:pre-wrap;background:#e8edf5;padding:18px}</style><main>'+text+'</main></html>')
    print('REPORT rebuilt from saved results only')


if __name__=='__main__':main()
