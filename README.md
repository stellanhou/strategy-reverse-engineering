# Strategy Reverse Engineering · 策略逆向建模

从交易记录重建仓位与行为，结合公开行情，检验模型能否复现开仓、加仓、减仓和平仓决策。

本仓库公开一项加密资产交易行为研究的 Python 源码与单元测试，涵盖数据核对、交易周期重建、因果特征、行为建模、时间切分评估和独立复核。

## 研究状态

截至 2026-10-08，最新的 v6 联合开仓实验中，线性模型与浅层树均为 **0/3 个评估时间段通过**，没有合格的开仓模块或完整策略。历史样本已被多轮研究使用，结果属于探索性时间切分评估。

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

36 项单元测试使用构造数据，可以在没有私人交易数据的情况下运行。测试覆盖未来数据不能改变既往决策、真实仓位不能泄漏到自主决策、事件不得回溯匹配、同分钟多币事件计数及抽样权重等约束。

## 数据与完整复现

公开仓库仅包含源码、测试和说明。以下内容保留在本地，已加入 `.gitignore`：

- `data/`：交易导出、账户响应、标准化记录、数据库及行情快照。
- `analysis/results/`：冻结实验协议、原始文件哈希清单、特征面板、拟合模型、预测、核验产物和报告。

**克隆仓库后可直接安装依赖并运行单元测试；完整历史实验不能仅凭公开文件复现。** 实验入口依赖对应版本的原始数据、冻结协议及上游产物，部分步骤还会验证历史文件哈希。公开仓库没有提供这些文件的替代版本。

脚本保留原研究快照中的固定数据目录 `data/binance/2026-10-02_2026大黑马/`，以及少量报告中的本机路径文字。迁移到自己的数据时，需要适配输入路径、字段格式和实验协议；不能直接把不同数据套入原来的结果结论。部分数据准备脚本在顶层执行，只应在准备好输入后作为脚本运行。

## 许可证

源码以 [MIT License](LICENSE) 发布。未公开的交易数据及第三方数据不随仓库分发。
