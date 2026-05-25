# 第一阶段 · 1.1 信息流探针实验方案

> **版本**：v0.1，2026-05-24
> **状态**：进行中（**当前焦点**）
> **上级文档**：[../02_research_plan.md](../02_research_plan.md)

---

## 1. 实验目标

用**可量化的指标**回答以下三个问题：

1. **TENet 各层捕获了哪些信息？**（带宽 / 任务相关性）
2. **哪些信息在哪里丢失？**（层间互信息衰减）
3. **核心算子 $\phi$、$zLz^\top$ 实际学到了什么？**（算子语义）

实验产物将作为 [02_findings.md](02_findings.md)（待建）和 [03_failure_modes.md](03_failure_modes.md)（待建）的**唯一数据来源**。

---

## 2. 测量工具集

> 工具完整实现将统一放到 `Code/analysis/` 模块（待建）。

### 工具 1：有效秩（Effective Rank）—— 测带宽

**定义**：对特征矩阵 $X \in \mathbb{R}^{N\times C}$ 做 SVD，得 $\sigma_1 \ge \cdots \ge \sigma_C$，归一化 $p_i = \sigma_i / \sum_j \sigma_j$。
有效秩 $= \exp\left(-\sum_i p_i \log p_i\right)$。

**意义**：
- 值远小于 $C$ → 该层**带宽被压扁**，存在信息瓶颈
- 逐层下降至 1~2 → **over-smoothing**
- 稳定接近 $C$ → 带宽充分利用

**最小实现**：
```python
@torch.no_grad()
def effective_rank(x):
    x = x.flatten(start_dim=0, end_dim=-2)
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * (p + 1e-12).log()).sum()
    return entropy.exp().item()
```

---

### 工具 2：Linear CKA —— 测层间信息保留

**定义**：对 $X \in \mathbb{R}^{N\times C_1}$，$Y \in \mathbb{R}^{N\times C_2}$（已中心化）：
$$
\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F} \in [0, 1]
$$

**意义**：测"第 $l$ 层输出与输入"或"第 $l$ 层与第 $l+1$ 层"之间的线性可预测性。

**最小实现**：
```python
@torch.no_grad()
def linear_cka(X, Y):
    X = X.flatten(start_dim=0, end_dim=-2)
    Y = Y.flatten(start_dim=0, end_dim=-2)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (Y.T @ X).pow(2).sum()
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    return (num / den.clamp_min(1e-12)).item()
```

---

### 工具 3：输入梯度归因 —— 测跨子树泄漏

**定义**：对第 $l$ 层位置 $(i,j)$ 的特征 $f^{(l)}_{ij}$，计算
$$
S^{(l)}_{(i,j)}(p, q) = \left|\frac{\partial \|f^{(l)}_{ij}\|^2}{\partial x_{pq}}\right|
$$

**意义**：
- $S$ 严格落在 quadtree 子树范围内 → 树结构生效
- $S$ 子树边界附近梯度突然变化 → **边界切断假设**成立
- $S$ 出现跨子树泄漏 → BatchNorm 等模块意外引入全局通路

---

### 工具 4：层级 Linear Probe —— 测任务相关信息

**做法**：冻结 TENet 主干 → 在每一层输出后接一个 `nn.Linear(d, num_classes)` → 只训这个线性层 → 看精度。

**意义**：probe 精度的下降意味着任务相关信息丢失（**信息瓶颈**的实际表现）。

---

### 工具 5：$\phi$ 函数形状可视化 —— 测算子语义

**做法**：在 $[0, 1]^2$ 网格上采样 $(v_i, v_j)$，画 $\phi(v_i, v_j)$ 的 heatmap。

**期望形状**：
- 类似 $|v_i - v_j|$ 的 V 型山谷 → $\phi$ 确实学差分
- 学到与 $v_i, v_j$ 强相关的非平凡形状 → 学到了任务特定的成对度量
- 噪声形状 → $\phi$ 实际未被有效训练

---

## 3. 实验矩阵

记号：
- $D$ ∈ {MNIST, CIFAR-10}
- $\alpha$ ∈ {1, 4, 8, 16, 32}
- $L_{depth}$ ∈ {2, 3, 4, 5}（层数）
- $M$ ∈ 模块开关组合（baseline / +trace / +edge / +cross / +all）

**主实验矩阵**：

| 实验编号 | 数据集 | $\alpha$ | 层数 | 模块 | 测量工具 |
|---|---|---|---|---|---|
| E1 | MNIST | 1, 8 | 2 | baseline | 1, 2, 4 |
| E2 | CIFAR-10 | 1, 4, 8, 16, 32 | 2 | baseline | 1, 2, 4 |
| E3 | CIFAR-10 | 8 | 2, 3, 4, 5 | baseline | 1, 2, 4 |
| E4 | CIFAR-10 | 8 | 4 | baseline / +trace / +cross / +all | 1, 2, 4 |
| E5 | CIFAR-10 | 8 | 4 | baseline | 3, 5 |

**测量协议**：
- 训练完整模型 → 用 fixed validation batch（≥ 256 样本）→ hook 每层输出 → 算工具 1~2、4
- 工具 3、5 在训练完成后离线分析
- 所有曲线和数值结果写入 `docs/90_results/phase1_probe/` 下对应文件夹

---

## 4. 预期结论模板

实验完成后，应至少能填出下表（数字示例）：

| 实验 | 浅层有效秩 | 深层有效秩 | 浅层 probe acc | 深层 probe acc | CKA(input, 末层) | 结论关键词 |
|---|---|---|---|---|---|---|
| E1（MNIST, α=1） | ? | ? | ? | ? | ? | 信息保留好 / 瓶颈 |
| E2（CIFAR, α=1） | ? | ? | ? | ? | ? | 带宽瓶颈 / 充分 |
| E2（CIFAR, α=32） | ? | ? | ? | ? | ? | 带宽瓶颈是否解除 |
| E3（CIFAR, 深 5 层） | ? | ? | ? | ? | ? | over-smoothing |
| E4（CIFAR, +all） | ? | ? | ? | ? | ? | 模块带来的信息增益 |

填完此表后，将其归档为 [02_findings.md](02_findings.md) 的主要内容。

---

## 5. 工程任务清单

> 这一节是**可执行的 TODO 列表**。每完成一项打勾，并把对应代码 / 报告链接补上。

- [x] 建立 [`Code/analysis/`](../../Code/analysis) 目录
- [x] 实现 [`effective_rank.py`](../../Code/analysis/effective_rank.py)（工具 1）
- [x] 实现 [`cka.py`](../../Code/analysis/cka.py)（工具 2）
- [x] 实现 [`input_grad.py`](../../Code/analysis/input_grad.py)（工具 3）
- [x] 实现 [`linear_probe.py`](../../Code/analysis/linear_probe.py)（工具 4）
- [x] 实现 [`phi_visualize.py`](../../Code/analysis/phi_visualize.py)（工具 5）
- [x] 实现 [`hooks.py`](../../Code/analysis/hooks.py)：统一注册中间层输出 hook（`ActivationCollector`）
- [x] 实现 [`run_probe.py`](../../Code/analysis/run_probe.py)：一键编排 train/load → 探针 → 报告
- [x] 建立 [`docs/90_results/phase1_probe/`](../90_results/phase1_probe) 归档目录
- [x] 跑 E1（MNIST baseline）并归档结果 → [02_findings.md §1](02_findings.md#1-e1--mnist-baselineα12-层)
- [x] 跑 E2（CIFAR-10 alpha 扫描）并归档结果 → [02_findings.md §2](02_findings.md#2-e2--cifar-10-α-扫描2-层)（CPU 跑通 α∈{1,8,32}，**未收敛**）
- [x] 跑 E3（CIFAR-10 深度扫描）并归档结果 → [02_findings.md §3](02_findings.md#3-e3--cifar-10-深度扫描α8l234)
- [x] 跑 E4（CIFAR-10 模块开关）并归档结果 → [02_findings.md §4](02_findings.md#4-e4--cifar-10-模块消融α8-l4)
- [x] 跑 E5（CIFAR-10 工具3/5 离线分析）并归档结果 → [02_findings.md §5](02_findings.md#5-e5--离线-φ-heatmap--输入梯度归因)
- [x] 整理结果到 [02_findings.md](02_findings.md)（v0.2）

---

## 6. 风险与降级方案

| 风险 | 概率 | 降级方案 |
|---|---|---|
| 现有 checkpoint 不可复现 | 中 | 用相同 seed 重新训练再测 |
| 工具 3 输入梯度计算成本高 | 中 | 用 batch=1 + 少量样本可视化即可 |
| 工具 4 linear probe 需重新训练 | 低 | 用 sklearn LogisticRegression 在 CPU 上跑 |
| CIFAR 失败模式不止一种且混杂 | 高 | 接受混杂；用对照实验逐一隔离 |

---

## 7. 版本记录

| 日期 | 版本 | 主要变更 |
|---|---|---|
| 2026-05-24 | v0.1 | 初稿；定义 5 工具 + 5 实验矩阵 + 任务清单 |
| 2026-05-24 | v0.2 | E3/E4/E5 跑通，任务清单全勾；参见 02_findings v0.2 |
