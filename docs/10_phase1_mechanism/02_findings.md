# 第一阶段 · 1.2 信息流探针 — 初步发现

> **版本**：v0.2，2026-05-24
> **状态**：E1 / E2 / E3 / E4 / E5 全部跑通（CPU 探针通路验证，**未收敛**）
> **上级文档**：[01_information_flow_probe.md](01_information_flow_probe.md)
> **原始数据**：[`docs/90_results/phase1_probe/`](../90_results/phase1_probe)

---

## 0. 执行环境（很重要的限定）

- 设备：CPU（PyTorch 2.5.1，CUDA 不可用）
- 训练：`epochs=2`，每 epoch 只跑 **50 个 batch**（每个 batch 128 样本）
- 探针：1024 样本，per-layer linear probe 训 15 epoch
- 模块开关：`use_directed_flow=False`、`use_chebyshev=False`、`use_spatial_bridge=False`（仅 trace + edge + cross 三件套）
- 该规模**远未收敛**；下述数字只用来验证"探针指标随 α/数据集的相对变化"，**不**用来下结论模型最终性能

---

## 1. E1 — MNIST baseline（α=1，2 层）

> 原始：[`20260524_190847_mnist_a1_L2_tr+eg+cg/`](../90_results/phase1_probe/20260524_190847_mnist_a1_L2_tr+eg+cg)

- #params: **8,543** ；final val_acc: **94.66%**

| 层 | C | ER | ratio | CKA(input, layer) | CKA(layer, next) | probe acc |
|---|---|---|---|---|---|---|
| `layer1` | 2 | 1.621 | 0.810 | **0.9447** | 0.7941 | 90.75% |
| `layer2` | 3 | 2.964 | 0.988 | 0.7327 | — | 93.25% |

**观察**：
- 浅层 CKA(input) 极高（0.94）→ 表示在 MNIST 上，第一层几乎是"输入近似复制 + 轻度重编码"
- 两层 ER/C 比都很高（0.81、0.99），**没有带宽瓶颈**
- probe acc 与全模型 val_acc 几乎一致（93% vs 94%），说明任务可解性已在前两层完全提供

---

## 2. E2 — CIFAR-10 α 扫描（2 层）

> 原始目录（按时间）：
> - α=1：[`20260524_191500_cifar10_a1_L2_tr+eg+cg/`](../90_results/phase1_probe/20260524_191500_cifar10_a1_L2_tr+eg+cg)
> - α=8：[`20260524_191617_cifar10_a8_L2_tr+eg+cg/`](../90_results/phase1_probe/20260524_191617_cifar10_a8_L2_tr+eg+cg)
> - α=32：[`20260524_191844_cifar10_a32_L2_tr+eg+cg/`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg)

### 2.1 总览

| α | #params | val_acc | layer0 probe | layer1 probe |
|---:|---:|---:|---:|---:|
| 1  | 67,067    | 30.59% | 35.05% | 40.65% |
| 8  | 261,674   | 36.94% | 38.65% | 45.05% |
| 32 | 1,088,114 | 43.05% | 43.60% | 48.45% |

> 三档都呈现 **probe(layer1) > probe(layer0) > val_acc**：表层线性可分性比骨干分类头还高，提示 (a) 训练严重欠拟合分类头，或 (b) 多分类头并未充分利用层 1 信息。

### 2.2 各层 ER / CKA 细节

| α | 层 | C | ER | ratio | CKA(input) |
|---:|---|---:|---:|---:|---:|
| 1  | layers.0 | ? | 3.565  | 0.891 | 0.4549 |
| 1  | layers.1 | ? | 3.776  | 0.755 | 0.5900 |
| 8  | layers.0 | 11 | 8.378  | 0.762 | 0.6147 |
| 8  | layers.1 | 19 | 13.215 | 0.696 | 0.5862 |
| 32 | layers.0 | 35 | 23.144 | 0.661 | 0.5804 |
| 32 | layers.1 | 67 | 53.522 | 0.799 | 0.5805 |

### 2.3 关键观察

1. **有效秩随 α 线性增长但 ratio 下降**：α 从 1 → 8 → 32，layers.0 ER 从 3.6 → 8.4 → 23.1（绝对带宽扩张），但 ratio 从 0.89 → 0.76 → 0.66（**通道利用率下降**）。**带宽瓶颈缓解，但并未真正打开**。
2. **CKA(input) 非单调**：在 α=1、α=8 上都出现 `CKA(input, layer1) ≥ CKA(input, layer0)`（α=1: 0.59 ≥ 0.45；α=8: 0.59 ≈ 0.61），这与"信息瓶颈应单调递减"的标准直觉矛盾；α=32 上两层 CKA 完全相等（0.5804 vs 0.5805），更像噪声地板。可能解释：
   - quadtree pooling 在 layer1 重新引入了空间均值类的 input-correlated 结构
   - 1024 样本 + 仅 100 batch 训练，CKA 估计本身处于噪声地板
   - **需在 E3（更深层数）和更大样本量上复测**
3. **probe acc 随 α 单调上升**（35→39→44 和 41→45→48）：增大通道宽度对线性可分性单调有益，与 ER 的"绝对带宽"变化方向一致
4. **MNIST 上 CKA(input) ≈ 0.94 vs CIFAR α=32 上 CKA(input) ≈ 0.58**：数据集复杂度直接拉低了"输入像素与浅层特征"的线性可预测性，与"CIFAR 需要更非线性的早期变换"的直觉吻合

### 2.4 待澄清问题（推到 E3/E4）

- [ ] CKA(input) 非单调是 over-smoothing 反向、是空间池化引入的 artefact、还是采样噪声？
- [ ] 增大 num_layers 后，最深层 ER ratio 是否进一步坍缩到 over-smoothing 区？
- [ ] 关掉 trace_branch / edge_enhance / cross_guidance 中的任何一个，probe acc 与 ER 怎么变？

---

## 3. E3 — CIFAR-10 深度扫描（α=8，L∈{2,3,4}）

> 原始目录：
> - L=2：[`20260524_191617_cifar10_a8_L2_tr+eg+cg/`](../90_results/phase1_probe/20260524_191617_cifar10_a8_L2_tr+eg+cg)（沿用 E2 α=8 行）
> - L=3：[`20260524_192204_cifar10_a8_L3_tr+eg+cg/`](../90_results/phase1_probe/20260524_192204_cifar10_a8_L3_tr+eg+cg)
> - L=4：[`20260524_192355_cifar10_a8_L4_tr+eg+cg/`](../90_results/phase1_probe/20260524_192355_cifar10_a8_L4_tr+eg+cg)

### 3.1 总览

| L | #params | val_acc | probe(L0) | probe(L1) | probe(L2) | probe(L3) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 261,674 | 36.94% | 38.65% | **45.05%** | — | — |
| 3 | 165,302 | 26.74% | 38.75% | **42.00%** | 38.85% | — |
| 4 | 208,554 | 23.27% | **40.10%** | 39.80% | 35.75% | 31.05% |

### 3.2 各层 ER / CKA（L=3, L=4）

| L | 层 | C | ER | ratio | CKA(input) |
|---:|---|---:|---:|---:|---:|
| 3 | layers.0 | 11 | 8.214  | 0.747 | 0.5679 |
| 3 | layers.1 | 19 | 14.402 | 0.758 | 0.5686 |
| 3 | layers.2 | 27 | 22.677 | 0.840 | 0.5357 |
| 4 | layers.0 | 11 | 8.772  | 0.797 | 0.5221 |
| 4 | layers.1 | 19 | 13.726 | 0.722 | 0.3795 |
| 4 | layers.2 | 27 | 23.340 | 0.864 | 0.4495 |
| 4 | layers.3 | 35 | 24.116 | 0.689 | **0.2963** |

### 3.3 关键观察

1. **probe acc 越深越差**（L=4：40 → 40 → 36 → 31）。最深层 probe 比第 0 层低 **9 个百分点**，这是 over-smoothing / 任务相关信息逐层流失的直接证据。
2. **CKA(input) 随深度单调衰减**（L=4：0.52 → 0.38 → 0.45 → 0.30，仅 L2 反弹一次），与 probe 趋势一致。
3. **ER 仍随 C 同步抬升**（L=4 末层 ER=24/35=0.69）——带宽并未坍缩到 1-2 的退化区，说明丢失的不是"通道维度"而是"任务相关性"。
4. **val_acc 与深度负相关**（37% → 27% → 23%）：在 50 batch×2 epoch 的极小训练预算下，深度纯粹是负担——参数没机会被训。这条结论**只对当前 CPU 探针预算成立**，不能外推到正常训练。

---

## 4. E4 — CIFAR-10 模块消融（α=8, L=4）

> 原始目录：
> - 全开 (tr+eg+cg)：[`20260524_192355_cifar10_a8_L4_tr+eg+cg/`](../90_results/phase1_probe/20260524_192355_cifar10_a8_L4_tr+eg+cg)
> - 关 trace_branch (eg+cg)：[`20260524_192544_cifar10_a8_L4_eg+cg/`](../90_results/phase1_probe/20260524_192544_cifar10_a8_L4_eg+cg)
> - 关 edge_enhance (tr+cg)：[`20260524_192730_cifar10_a8_L4_tr+cg/`](../90_results/phase1_probe/20260524_192730_cifar10_a8_L4_tr+cg)
> - 关 cross_guidance (tr+eg)：[`20260524_192911_cifar10_a8_L4_tr+eg/`](../90_results/phase1_probe/20260524_192911_cifar10_a8_L4_tr+eg)

### 4.1 总览

| 配置 | #params | val_acc | probe(L0) | probe(L1) | probe(L2) | probe(L3) |
|---|---:|---:|---:|---:|---:|---:|
| **tr+eg+cg** (全开) | 208,554 | 23.27% | 40.10% | 39.80% | 35.75% | 31.05% |
| eg+cg (-trace)      | 206,670 | 21.84% | 36.45% | 35.00% | 27.70% | 23.90% |
| tr+cg (-edge)       | 208,546 | 23.71% | 40.50% | 41.10% | 35.70% | 30.00% |
| tr+eg (-cross)      | 203,314 | 23.51% | 41.00% | 40.50% | 35.05% | 30.45% |

### 4.2 各层 ER ratio / CKA(input) 末层 对比

| 配置 | L0 ratio | L3 ratio | CKA(L3, input) |
|---|---:|---:|---:|
| tr+eg+cg | 0.797 | 0.689 | 0.2963 |
| eg+cg    | 0.894 | 0.924 | 0.3811 |
| tr+cg    | 0.799 | 0.748 | 0.3011 |
| tr+eg    | 0.902 | 0.601 | 0.2939 |

### 4.3 关键观察

1. **关 trace_branch 影响最大**：probe 全层下降 ~4-7 个百分点（L3：31 → 24），val_acc 也由 23.3 → 21.8。**trace_branch 是当前预算下 probe 信息量贡献最大的模块。** 其参数代价仅 ~1,884（< 1%）。
2. **关 edge_enhance / 关 cross_guidance 在 probe 上几乎无变化**（甚至 tr+cg 的 L1 probe 还略升至 41.1%），val_acc 也基本持平。在当前训练预算下，这两个模块对 probe 没有显式增益，可能：
   - (a) 它们带来的信号在 50 batch 训练里没机会被学到
   - (b) 它们对 probe 度量本来不敏感（CKA/probe 反映线性可分性，而 edge/cross 更可能影响更高阶结构）
3. **关 trace 时 ER ratio 反而更高**（L3=0.92 vs 0.69）：少了 trace_branch 的特征压缩通路，通道分布更接近均匀——但任务相关性 probe 反而更低。**与"高 ER ratio = 好"的朴素直觉相反**——再次说明 ER 是带宽指标而非任务可解性指标。
4. **cross_guidance 关掉后 CKA(L3, input) = 0.294 ≈ 全开 0.296**：cross-stream guidance 既不显著改变"末层与输入的线性相关"，也不显著改变 probe，**在当前微训练预算下其效果不可见**。

---

## 5. E5 — 离线 φ heatmap + 输入梯度归因

> 数据来自 α=32 L=2 全开模型：[`20260524_191844_cifar10_a32_L2_tr+eg+cg/`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg)

### 5.1 φ 函数形状

用 [`Code/analysis/plot_phi.py`](../../Code/analysis/plot_phi.py) 渲染：

- [`phi_layers_0.png`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg/phi_layers_0.png)：channel 0，值域 [0.067, 1.072]
- [`phi_layers_1.png`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg/phi_layers_1.png)：channel 0，值域 [0.052, 1.056]

**观察**：值域很窄、跨越 ~1 的量级——φ 学到了**非平凡但接近 affine** 的形状，**未出现明显的 |v_i - v_j| V 型山谷**。这与"边强度 = 节点差分"的设计直觉有 gap：当前训练规模下 φ 还没显式学到差分形状，更像"对 (v_i, v_j) 的弱非线性混合"。**需要在收敛模型上重测**才能下确定结论。

### 5.2 输入梯度归因

用 [`Code/analysis/run_input_grad.py`](../../Code/analysis/run_input_grad.py) 对 4 张 CIFAR 测试图算 $|\partial \|f^{(l)}_{ij}\|^2 / \partial x|$（取层中心位置）：

- [`input_grad/sample_00_y3.png`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg/input_grad/sample_00_y3.png)
- [`input_grad/sample_01_y8.png`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg/input_grad/sample_01_y8.png)
- [`input_grad/sample_02_y8.png`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg/input_grad/sample_02_y8.png)
- [`input_grad/sample_03_y0.png`](../90_results/phase1_probe/20260524_191844_cifar10_a32_L2_tr+eg+cg/input_grad/sample_03_y0.png)

**观察（待人工核对图像）**：
- layer 0 中心位置的输入梯度图应该集中在 32×32 输入的**对应 quadtree 子树范围内**（中心区域）
- layer 1 中心位置（下采样后）的输入梯度应该覆盖**更大但仍非全图**的区域
- 若发现梯度严格 0 之外的区域出现于子树边界外 → 存在跨子树泄漏（BatchNorm 等模块全局通路）

本节作为**通路验证**：脚本 + 渲染 + 文件均已就绪，定性结论留待真实收敛模型 + 多样本统计。

---

## 6. 与原计划"预期结论模板"的对照

| 实验 | 浅层 ER | 深层 ER | 浅层 probe | 深层 probe | CKA(input, 末层) | 结论关键词 |
|---|---:|---:|---:|---:|---:|---|
| E1（MNIST, α=1） | 1.621 | 2.964 | 90.75% | 93.25% | 0.7327 | 信息保留好，无瓶颈 |
| E2（CIFAR, α=1）  | 3.565  | 3.776  | 35.05% | 40.65% | 0.5900 | 严重带宽不足 |
| E2（CIFAR, α=32） | 23.144 | 53.522 | 43.60% | 48.45% | 0.5805 | 带宽扩，ratio 下降 |
| E3（CIFAR, L=4） | 8.772 | 24.116 | 40.10% | **31.05%** | **0.2963** | **probe & CKA 随深度衰减** |
| E4（CIFAR, -trace, L=4） | 8.790 | 26.169 | 36.45% | **23.90%** | 0.3811 | trace_branch 是 probe 主贡献 |

---

## 7. 综合结论（v0.2）

1. **带宽未坍缩，但任务可解性随深度下降**：ER 仍随 C 线性抬升（ratio ≥ 0.6），但 probe acc 在 L≥2 后逐层流失，CKA(input, 末层) 从 L=2 的 0.58 跌到 L=4 的 0.30。**这是当前架构在 CPU 微训练预算下的主要病灶**——不是带宽瓶颈，而是"深层把任务相关性 wash 掉"。
2. **trace_branch 是 probe 贡献最大的模块**（关掉它 probe 全层降 4-7 pp），但参数仅占 < 1%。edge_enhance 和 cross_guidance 在 50 batch×2 epoch 下尚未表现出 probe 增益。
3. **φ 形状目前是"弱非线性混合"而非"差分"**，与设计直觉有 gap；需在收敛模型上复测。
4. **输入梯度归因脚本就绪**，可在后续真实训练流程中用于跨子树泄漏诊断。

---

## 8. 下一步

1. 把训练预算从 50 batch×2 epoch 扩到 **全 epoch×≥20 epoch**，把上述"探针通路验证"升级为"可下结论的实验"
2. 在 L=4 配置下做 **trace_branch 单独保留** 的反向消融（仅 tr，无 eg / cg），确认 trace_branch 的贡献是否独立
3. 收敛模型上**复测 φ heatmap**，确认是否出现 V 型山谷
4. 用更多样本（≥50 张）批量跑输入梯度，做跨子树泄漏的**定量**统计（梯度落在子树内 / 外的能量比）

---

## 9. 版本记录

| 日期 | 版本 | 主要变更 |
|---|---|---|
| 2026-05-24 | v0.1 | E1 + E2（α∈{1,8,32}）跑通，初步发现归档 |
| 2026-05-24 | v0.2 | 补 E3 深度扫描、E4 模块消融、E5 φ heatmap 与输入梯度可视化；新增综合结论 §7 |
