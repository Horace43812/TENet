# TENet / RGA 研究文档索引

本目录是 TENet（Topology-Enhanced Network）/ RGA（Recursive Graph Aggregation）研究项目的**文档中枢**。
所有理论假设、实验计划、机理分析、阶段结论都落在这里，便于长期追溯与论文写作时直接引用。

---

## 文档结构

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| [01_manifesto.md](01_manifesto.md) | 研究宣言（Manifesto v0.1）：立场、假设、与已有范式的关系、可证伪承诺 | 草稿 v0.1 |
| [02_research_plan.md](02_research_plan.md) | 2026-05-24 整体研究计划：阶段划分、里程碑、交付物 | 历史草稿 v0.1 |
| [03_graphification_recursion_plan.md](03_graphification_recursion_plan.md) | **当前研究计划**：图化价值、递归瓶颈、优势范围、阶段决策与运行预算 | v1.0，待执行 |
| [04_graphification_recursion_protocol.md](04_graphification_recursion_protocol.md) | **当前实验协议**：对照矩阵、数据划分、训练/统计/效率规则及实现清单 | v1.0，核心平台已实现，真实实验待执行 |
| [实验平台操作手册](../TENetMechanism/EXPERIMENTS.md) | 统一入口、中文菜单、批量运行/续跑、分析与报告 | 可用 |
| [10_phase1_mechanism/](10_phase1_mechanism/) | **第一阶段**：机理探究（信息流捕获 / 缺失分析、扬长补短） | 进行中 |
| [10_phase1_mechanism/01_information_flow_probe.md](10_phase1_mechanism/01_information_flow_probe.md) | 早期信息流探针实验方案；当前测量与结论边界见新协议 | 历史方案 |
| [20_phase2_applications/](20_phase2_applications/) | **第二阶段**：应用探索（独立新范式 / Transformer 改进元件） | 未开始 |
| [90_results/](90_results/) | 实验结果归档（曲线、表格、checkpoint 索引） | 已有早期结果；新批次待执行 |
| 99_glossary.md | 术语与符号约定 | 待建 |

---

## 当前研究状态（live）

- **当前计划**：先验证图化的独立收益，再验证递归使用的位置与瓶颈，最后评估优势范围。
- **执行入口**：[研究计划](03_graphification_recursion_plan.md)与[实验协议](04_graphification_recursion_protocol.md)。当前处于P0准备阶段，文档完成不代表训练已开始。
- **现有工具**：TENetMechanism集成入口已支持普通/图聚合、混合阶段、冻结数据清单、批量续跑、真实类别探针及A–E、预算选型和报告；完整FLOPs及CIFAR-100专用加载仍待补充。
- **决策依据**：重新训练的对照结果、跨种子复核与真实成本；探针用于定位，不能单独证明不可逆信息损失。

---

## 修订日志

| 日期 | 版本 | 变更 |
| --- | --- | --- |
| 2026-05-24 | v0.1 | 初稿建立：宣言 + 研究计划 + 第一阶段第一部分实验方案 |
| 2026-09-06 | v1.0 | 增加图化与递归研究计划、实验协议；更新当前入口并保留历史方案 |
