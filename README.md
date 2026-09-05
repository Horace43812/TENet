# TENet — Topology-Enhanced Network

独立机制研究工作区：[TENetMechanism](TENetMechanism/README.md)。支持同一 ImageFolder 数据集上的独立训练、层内类别可分性分析、退化样本定位和固定权重干预。

TENet（Topology-Enhanced Network，拓扑增强网络）是一种将**图论 / 代数拓扑**与卷积神经网络相结合的新型神经网络架构。
它的核心思想是：**把每一个 2×2 局部图像块显式地“图化（graphify）”为一张带有边特征的小图**，再通过预定义的拉普拉斯算子提取图的二次型 / 谱特征，与传统的内容流（content stream）双流交互，从而在保持卷积归纳偏置的同时引入显式的拓扑结构信息。

---

## 1. 核心思想

对输入张量 $x \in \mathbb{R}^{B \times C \times H \times W}$，TENet 在每个 $2\times 2$ 局部块上构造一张有 4 个顶点、6 条无向边的完全图 $K_4$：

- 顶点集合：$\{v_1, v_2, v_3, v_4\}$，对应 2×2 块的 4 个像素（按通道展开）。
- 边集合：$\{e_{ij}\} = \{\phi(v_i, v_j)\}$，由可学习的连接强度函数 $\phi$ 给出。
- 将 4 个顶点与 6 条边拼成一个 10 维特征序列 $z$，并用预先注册的**拉普拉斯矩阵** $L \in \mathbb{R}^{10\times 10}$ 计算图二次型：

$$
G = z\,L\,z^\top, \qquad z = [v_1, v_2, v_3, v_4, e_{12}, e_{13}, e_{14}, e_{23}, e_{24}, e_{34}]
$$

- 再用一个 $10\times 10$ 的卷积核把 $G$ 投影到输出通道（topology extractor），同时辅以 trace 分支 $\sum z_i^2 L_{ii}$ 做补强。
- 与之并行的**内容流**对同一个 2×2 块做均值池化 + 1×1 卷积升维；两条流通过可学习门控 $\gamma, \beta$ 进行**跨流引导**（C→T 与 T→C）。

每经过一层 `GraphifiedLayer`，空间分辨率减半（2×2 聚合），通道数由 $C$ 升到 $C + \alpha$。

---

## 2. 模块结构

代码位于 [Code/](Code/)，按职责分层组织：

| 文件 | 作用 |
| --- | --- |
| [Code/model_common.py](Code/model_common.py) | 核心 `GraphifiedLayer` 与 `TrainablePhi`，所有数据集模型共用 |
| [Code/model_mnist28.py](Code/model_mnist28.py) | MNIST（28×28 灰度）模型入口，从 `history/` 导出 current / legacy 两套架构 |
| [Code/model_rgb32.py](Code/model_rgb32.py) | RGB 32×32 模型（如 CIFAR-10），可配置层数 |
| [Code/model_rgb256.py](Code/model_rgb256.py) | RGB 256×256 模型（如 PlantVillage），可配置层数；`TENet_PlantVillage = TENet_RGB256` |
| [Code/model_arch.py](Code/model_arch.py) | 早期参考实现（保留用于阅读） |
| [Code/model_arch_legacy.py](Code/model_arch_legacy.py) | 不含双流 / 跨流引导的最早版本，作为对照基线 |
| [Code/history/](Code/history) | 按版本归档的历史模型快照 |
| [Code/train.py](Code/train.py) | 统一训练入口，支持 mnist / cifar10 / rgb256(plantvillage) |
| [Code/TENet_PlantVillage.py](Code/TENet_PlantVillage.py) | 独立的 PlantVillage 训练脚本（含 AMP、tqdm 等） |

### 2.1 `GraphifiedLayer` 关键开关

| 开关 | 说明 |
| --- | --- |
| `use_learnable_phi` | 是否使用可学习的 $\phi$（小型 MLP + 温度 + 残差），关闭则退化为 $|v_i - v_j|$ |
| `use_trace_branch` | 是否启用 trace 补强分支并加权融合 |
| `use_edge_enhance` | 是否在边特征上叠加平方项与乘积项 |
| `use_directed_flow` | 是否引入有向关联矩阵 $B$ 计算流量项 $B^\top z_v$ |
| `use_cross_stream_guidance` | 是否启用 topo 流与 content 流双向门控引导 |
| `use_chebyshev` | 是否使用 $L^2$ 做二阶切比雪夫式融合 |
| `use_spatial_bridge` | 是否启用 depthwise dilated conv 的空间桥接 |

---

## 3. 安装

依赖（推荐 Python ≥ 3.9）：

```bash
pip install torch torchvision tqdm
```

如需 GPU 训练，请按官方指引安装匹配 CUDA 的 PyTorch。

---

## 4. 训练

### 4.1 通用入口 [Code/train.py](Code/train.py)

支持 `mnist` / `cifar10` / `plantvillage`（即 `rgb256`）三种数据集，统一处理 warmup-cosine 调度、AMP、label smoothing、trace 分支的差异化学习率、best checkpoint 保存等。

```bash
cd Code

# MNIST（默认使用 current 架构）
python train.py --dataset mnist --epochs 20 --batch-size 128 --lr 8e-4 \
                --alpha 3 --trace-init 0.005

# MNIST，使用 legacy 架构做对照
python train.py --dataset mnist --model-version legacy --epochs 20

# CIFAR-10（RGB 32）
python train.py --dataset cifar10 --epochs 50 --batch-size 128 --lr 1e-3 \
                --rgb32-num-layers 2 --alpha 8

# PlantVillage（RGB 256，39 类）
python train.py --dataset plantvillage --data-dir /path/to/PlantVillage-Dataset-master \
                --epochs 20 --batch-size 8 --lr 1e-3 \
                --rgb256-num-layers 4 --image-size 256
```

常用开关（详见 `python train.py -h`）：

- `--no-learnable-phi` / `--no-trace-branch` / `--no-edge-enhance` / `--no-cross-guidance`：消融关闭对应模块
- `--directed-flow` / `--use-chebyshev` / `--use-spatial-bridge`：启用对应增强模块
- `--no-aug`：关闭训练增强
- `--no-save-best`：关闭最优权重保存
- `--resume <path>`：从 checkpoint 继续训练
- `--cpu`：强制 CPU 训练

默认 best checkpoint 路径：

- MNIST current: `checkpoints/tenet_current_best.pth`
- MNIST legacy:  `checkpoints/tenet_legacy_best.pth`
- CIFAR-10:      `checkpoints/tenet_rgb32_cifar10_best.pth`
- PlantVillage:  `checkpoints/tenet_rgb256_best.pth`

### 4.2 独立 PlantVillage 脚本 [Code/TENet_PlantVillage.py](Code/TENet_PlantVillage.py)

一个早期、独立的 PlantVillage 训练流程（含 AMP、tqdm、train/val/test 三划分）。运行前请把脚本末尾 `data_root` 修改为本地数据路径：

```bash
cd Code
python TENet_PlantVillage.py
```

> PlantVillage 数据目录结构应为标准 `ImageFolder` 格式：每个类别一个子目录。默认期望 39 类。

---

## 5. 数据集

| 数据集 | 输入 | 类数 | 来源 |
| --- | --- | --- | --- |
| MNIST | 1×28×28 | 10 | `torchvision.datasets.MNIST` 自动下载 |
| CIFAR-10 | 3×32×32 | 10 | `torchvision.datasets.CIFAR10` 自动下载 |
| PlantVillage | 3×256×256 | 39 | 需手动准备 `ImageFolder` 目录 |

---

## 6. 模型版本

- **current**：当前主推的双流 + 跨流引导 + 边增强 + 可选有向流 / 切比雪夫 / 空间桥架构。
- **legacy**：保留单流、不含跨流引导的最早版本，用作消融基线，仅适配 MNIST。

历史快照保存在 [Code/history/](Code/history)，文件名带日期后缀，便于复现论文/实验。

---

## 7. 引用

如该工作对你有帮助，欢迎在论文 / 项目中注明出处。后续正式发表后会在此补充 BibTeX 条目。

---

## 8. 许可证

本项目遵循根目录 [LICENSE](LICENSE) 中声明的开源协议。
