# Strategy Reverse Engineering · 策略逆向建模

从交易记录重建仓位与行为，结合公开行情，检验模型能否复现开仓、加仓、减仓和平仓决策。

本仓库公开一项加密资产交易行为研究的 Python 源码与单元测试，涵盖数据核对、交易周期重建、因果特征、行为建模、时间切分评估和独立复核。

## 研究状态

截至 2026-10-13，v7 对训练分位截尾、短周期行情特征和报警方式进行了受控对照。截尾缓解了原线性模型的极端概率外推，但所有候选仍为 **0/3 个评估时间段通过**，严格开仓匹配无命中，没有合格的开仓模块或完整策略。v6 原实验快照保持不变，其原报警结果已在 v7 中逐条回放。

历史样本已被多轮研究使用，结果属于探索性时间切分评估，没有新增未触碰的留出样本。

研究数据缺少原始挂单、撤单及未成交订单的完整生命周期，因此不能据此还原全部下单机制。当前代码是研究快照，不包含实盘交易执行接口。

## 内容

| 路径 | 用途 |
| --- | --- |
| `build_dataset.py`、`finalize_dataset.py` | 整理导出记录、核对数据并构建本地数据集 |
| `fetch_market.py`、`analysis/fetch_reverse_market.py` | 获取公开行情与研究区间的补充数据 |
| `analysis/reconstruct_leader.py`、`analysis/reconstruct_follower.py` | 重建带单与跟单交易周期、事件和仓位 |
| `analysis/reverse_audit.py`、`analysis/build_reverse_panel.py` | 数据审计与研究面板构建 |
| `analysis/reverse_rules.py`、`analysis/reverse_models.py` | 规则和统计模型研究 |
| `analysis/research_observed_behavior.py` | 已观测行为研究（v3） |
| `analysis/behavior_hazard.py`、`analysis/run_behavior_hazard.py` | 行为概率与自主解码研究（v4） |
| `analysis/entry_management.py`、`analysis/run_entry_management.py` | 开仓与仓位管理分解（v5） |
| `analysis/research_joint_entry.py` | 联合开仓时机与选币建模（v6） |
| `analysis/robust_entry.py`、`analysis/research_robust_entry.py` | 训练分位截尾、短周期特征与报警方式受控对照（v7） |
| `analysis/verify_robust_entry.py` | v7 产物、因果输入、模型预测与评分复核 |
| `analysis/verify_*.py`、`analysis/build_*report.py` | 结果独立复核与报告生成 |
| `analysis/tests/` | 因果信息边界、事件匹配、概率与仓位管理测试 |

## 安装与测试

已在 Python 3.9.6 上使用 `requirements.txt` 中的版本验证。建议使用独立环境：

```bash
git clone https://github.com/stellanhou/strategy-reverse-engineering.git
cd strategy-reverse-engineering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s analysis/tests -v
```

49 项单元测试使用构造数据，可以在没有私人交易数据的情况下运行。测试覆盖未来数据不能改变既往决策、真实仓位不能泄漏到自主决策、事件不得回溯匹配、同分钟多币事件计数、抽样权重、训练期截尾及报警冷却等约束。

## 数据与完整复现

公开仓库仅包含源码、测试和说明。以下内容保留在本地，已加入 `.gitignore`：

- `data/`：交易导出、账户响应、标准化记录、数据库及行情快照。
- `analysis/results/`：冻结实验协议、原始文件哈希清单、特征面板、拟合模型、预测、核验产物和报告。

**克隆仓库后可直接安装依赖并运行单元测试；完整历史实验不能仅凭公开文件复现。** 实验入口依赖对应版本的原始数据、冻结协议及上游产物，部分步骤还会验证历史文件哈希。公开仓库没有提供这些文件的替代版本。

脚本保留原研究快照中的固定数据目录 `data/binance/2026-10-02_2026大黑马/`，以及少量报告中的本机路径文字。迁移到自己的数据时，需要适配输入路径、字段格式和实验协议；不能直接把不同数据套入原来的结果结论。部分数据准备脚本在顶层执行，只应在准备好输入后作为脚本运行。

## v7 受控对照

v7 沿用 v6 的三个时间切分、事件评分和晋级门槛，只训练两个固定线性候选，不做模型超参数搜索：

- `bounded_linear`：行情输入按完整自然训练样本的 0.5%–99.5% 分位截尾，校准段和评估段不参与确定边界。
- `bounded_timing`：在上述处理上增加 `ret1`、`ret3`、`pullback15`，分别表示 1 分钟收益率、3 分钟收益率及距近 15 分钟最高收盘价的回撤，仅从已滞后的收盘价和各币自身过去窗口计算。

新旧模型均保留原报警方式作对照。另一种报警方式将同币 30 分钟冷却、冷却期间重选可用币及按校准报警数量选择阈值合并评估；目标数量来自训练期开仓频率，阈值不按校准命中率选择。每日报警额度沿用原规则。冷却可能抑制真实的快速再开仓。

三个评估段的合并结果如下。主评分采用同币、一对一、±60 秒事件匹配，真实事件不得早于信号发出，默认延迟 1 分钟。概率指标不随报警方式变化；原报警方式及新报警方式的严格事件匹配均无命中。

| 模型 | 概率损失（越低越好） | 预测概率总量 / 实际事件数 | 通过时间段 |
| --- | ---: | ---: | ---: |
| v6 原线性模型 | 0.001602 | 2.603 | 0/3 |
| v6 原浅层树 | 0.000909 | 1.126 | 0/3 |
| v7 截尾线性模型 | 0.000929 | 1.096 | 0/3 |
| v7 截尾＋短周期特征 | 0.000928 | 1.121 | 0/3 |

截尾缓解了原线性模型的概率总量偏高问题，但未改善严格开仓复现，也未超过原浅层树的合并概率损失。本轮没有形成交易收益或完整策略结论。

具备本地原始数据和 v6 上游产物后，默认命令运行完整 v7 对照并生成 `results.json`、`report.html` 等产物：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 analysis/research_robust_entry.py
PYTHONDONTWRITEBYTECODE=1 python3 analysis/verify_robust_entry.py
```

默认输出为 `analysis/results/robust_entry_v7/`。实验运行要求输出目录不存在，已有结果不会被覆盖。再次运行时，`--output` 必须指定一个新目录，复核时传入同一目录：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 analysis/research_robust_entry.py --output analysis/results/robust_entry_v7_rerun
PYTHONDONTWRITEBYTECODE=1 python3 analysis/verify_robust_entry.py --output analysis/results/robust_entry_v7_rerun
```

实验在运行前后核对受保护源文件及 v6 产物的内容哈希，并校验旧模型的原报警与指标回放一致。本轮已通过独立产物复核，包括 60 组报警情景。详细实验协议、预测和报告保留在本地输出目录，不随公开仓库分发。

## 许可证

源码以 [MIT License](LICENSE) 发布。未公开的交易数据及第三方数据不随仓库分发。
