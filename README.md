# TENet — Topology-Enhanced Network

TENet 是一个研究**递归局部图聚合**的神经网络项目。它把每个 2×2 局部特征窗口表示为包含顶点与边特征的小图，通过固定的图连接结构和可学习投影，将四个空间位置聚合为一个新的表示，再逐层扩大所覆盖的区域。

项目关注的问题是：**能否用明确的局部关系与层次结构，在减少空间表示数量的同时，保留分类任务需要的区别？** 这里的压缩通过聚合完成：多个输入共同形成新特征，输出不必对应某一个被保留的原始位置。

TENet 的结构先验来自局部窗口、顶点—边关联关系和递归组织；边映射、投影与流间融合仍包含可学习参数。研究动机包括减少对复杂全局交互的依赖，以及探索资源受限场景中的规则计算结构。与注意力相当的表达能力、任务信息保留程度和实际硬件收益，都是需要验证的研究问题。

当前工作以**独立 TENet 的机制研究**为主：先从真实类别的区分退化定位问题，再检查具体算子及递归过程的作用。将图聚合用于词元嵌入、层次压缩或 Transformer，是后续应用方向。仓库同时保留了一个独立的 ViT token 评分剪枝实验分支。

## 1. 工作区与入口

| 目录 | 内容 | 使用场景 |
| --- | --- | --- |
| [TENetMechanism/](TENetMechanism/README.md) | 可配置独立 TENet、真实类别探针、层内干预、A–E 归一化实验 | 当前机制研究入口 |
| [TENetTiny/](TENetTiny/README.md) | current/legacy历史基准、补强与局部归一化开关、全程进度与最佳训练记录 | MNIST、EMNIST、KMNIST/K49、Fashion-MNIST、CIFAR、ImageFolder快速实验 |
| [TENetText/](TENetText/README.md) | 一维TENet、TextCNN与词向量平均基线，共享词表与训练流程 | AG News新闻主题分类、自定义中文/英文CSV文本分类 |
| [ViTPruning/](ViTPruning/README.md) | 标准 ViT 与 identity / magnitude / linear / tenet_local 对照 | Token 评分剪枝实验 |
| [Code/](Code/) | MNIST、CIFAR-10、RGB256 模型与早期训练、分析脚本 | 历史实现与实验追溯 |
| [docs/](docs/00_index.md) | 研究宣言、计划、早期发现与结果归档 | 查阅研究过程及原始证据 |

各工作区的训练参数、模型实现及 checkpoint 格式不同，应在各自目录运行命令。历史文档中的强主张和阶段结论保留其原始语境，不应直接当作当前已验证结论。

当前实验安排见[图化与递归研究计划](docs/03_graphification_recursion_plan.md)和[实验协议](docs/04_graphification_recursion_protocol.md)：先分解图化收益，再比较递归位置与瓶颈，最后验证优势范围；方案包含待实现接口，尚未代表实验结果。

集成操作入口：[实验平台使用手册](TENetMechanism/EXPERIMENTS.md)。`TENetMechanism/experiment.py`提供固定数据划分、批量对照、断点恢复、分析和HTML汇总；未运行的真实实验不作为结果。

## 2. 单层 TENet 实际计算什么

以下以 [TENetMechanism/graph.py](TENetMechanism/graph.py) 的默认单拉普拉斯图层为准。设输入为 $X\in\mathbb{R}^{B\times C\times H\times W}$，对每个非重叠 2×2 窗口、每个通道取四个标量 $v_1,v_2,v_3,v_4$，依次对应左上、右上、左下、右下位置。

### 2.1 从四个位置构造十个节点

四个位置之间枚举六对关系，与完全图 $K_4$ 的六条边对应：

$$
e_{ij}=\phi(v_i,v_j),\qquad 1\leq i<j\leq4.
$$

基础边特征为绝对差分。开启可学习映射时，代码在绝对差分上叠加小型 MLP 产生的门控残差；还可选择平方差、乘积和有向差分增强。因此，边特征并不总是纯距离，可学习映射也没有强制交换对称性。

将四个顶点特征与六个边特征组成：

$$
z=[v_1,v_2,v_3,v_4,e_{12},e_{13},e_{14},e_{23},e_{24},e_{34}]^\top\in\mathbb{R}^{10}.
$$

实际使用的固定图有 **10 个节点、12 条无向连接**：每个边特征节点只连接其两个端点。它是上述四顶点图的顶点—边关联结构，不是十节点完全图。其邻接矩阵为 $A$，组合拉普拉斯为 $L=D-A$，其中 $D_{ii}=\sum_j A_{ij}$。

### 2.2 构造矩阵，而不是直接求一个能量标量

代码中的逐元素乘法对应：

$$
G=\operatorname{diag}(z)L\operatorname{diag}(z)\in\mathbb{R}^{10\times10},
\qquad G_{ij}=z_iL_{ij}z_j.
$$

每个窗口、每个输入通道都会产生一张 10×10 矩阵。它包含受固定连接结构约束的二阶交互项。与它相关但不同的两个量是：

$$
\sum_{i,j}G_{ij}=z^\top Lz,
\qquad \operatorname{tr}(G)=\sum_i L_{ii}z_i^2.
$$

前者是拉普拉斯二次型，后者是代码中 trace 分支使用的量；二者不能混用。TENet 主分支保留矩阵再投影，并非只计算一个二次型标量。当前实现没有显式特征分解或同调计算，项目名中的 topology 指这里预设的图连接结构。

### 2.3 归一化、投影与双流融合

默认对每张局部矩阵独立计算均值与标准差：

$$
\widehat G=\frac{G-\mu}{s},\qquad s=\max(\operatorname{std}(G),10^{-6}).
$$

然后用 `Conv2d(C, C+alpha, kernel_size=10)` 将全部输入通道的局部矩阵映射为一个输出向量。这里的 10×10 卷积作用在图矩阵坐标上；它同时进行跨通道混合。

内容流对相同窗口做均值池化和 1×1 投影。拓扑流可加入 trace 投影及内容流引导，经过 BatchNorm / ReLU 后，再向内容流提供门控反馈。默认分类头读取拓扑流，实验工作区也支持读取内容流或两者拼接。

| 开关（TENetMechanism） | 对应机制 |
| --- | --- |
| `--learnable-phi` | 绝对差分上的可学习边映射残差 |
| `--trace` | 未归一化矩阵的 trace 补充分支 |
| `--edge` | 平方差与乘积增强 |
| `--directed` | 六对端点的有向差分 |
| `--cross` | 两个流之间的门控引导 |
| `--chebyshev` | 分别归一化基于 L、L² 的矩阵后加权混合；名称为历史沿用，并非完整标准 Chebyshev 滤波器 |
| `--spatial` | 深度可分组的空洞卷积空间桥接 |

开关均可用对应 `--no-...` 关闭；当前默认启用 learnable-phi、trace、edge、cross。

### 2.4 递归与计算约束

每个流经过一层后：

$$
B\times C\times H\times W
\longrightarrow
B\times(C+\alpha)\times H/2\times W/2.
$$

空间位置数量变为 1/4；`alpha=0` 可以保持通道宽度。多层聚合扩大感受区域，但是否保留任务所需的关系仍需实验检验。

在每个窗口成本固定的假设下，各层窗口数构成递减的几何级数。不过，实际成本还取决于通道宽度、10×10 中间矩阵、归一化、分支及显存访问。位置数减少不能直接等同于同比例加速，规则结构也不自动意味着已实现低面积或高硬件复用率。

## 3. 当前研究：从类别退化到机制解释

当前平台按以下路径工作：

1. 冻结训练好的独立 TENet，采集某层的输入、图节点、原始矩阵、归一化矩阵、投影及输出。
2. 用相同维度和训练预算的探针比较类别可分性；探针仅在训练集拟合。
3. 根据验证集 discovery 部分定位混淆增加的真实类别对，在 confirmation 部分复核相同类别对，导出原图路径。
4. 对具体操作做受控实验，再判断是否值得进行模型修改与重新训练。

探针描述符包含空间池化和固定哈希投影。因此，探针准确率下降既可能来自算子，也可能来自描述符压缩或读取器能力。CKA、有效秩、梯度图和 probe accuracy 都不能单独证明任务信息不可逆丢失。

### 3.1 已集成的 A–E 归一化实验

所有分支在池化和降维之前构造，保持网络权重不变：

| 组 | 提供给探针的矩阵 | 用途 |
| --- | --- | --- |
| A | G | 原始表示基准 |
| B | (G − μ) / s | 当前归一化 |
| C | s × B = G − μ | 补回尺度，只保留中心化 |
| D | B + μ | 单独加回偏移的干预 |
| E | s × B + μ | 完整恢复检查 |

E 必须在容差内恢复 A，否则分析停止。类别对统一按 A→B 在 discovery 上的混淆增量排序；各组不重新挑选有利样本。

当前讨论中的一组服务器结果表现为 C 接近 A、优于 B，提示局部尺度可能帮助当前探针读取类别信息。该结果尚未在仓库归档为完整可复现实验，不能视为跨种子结论，也不能换算成“任务信息保留率”。仅中心化当前是分析分支，尚未作为正式训练的可选归一化策略。

## 4. 独立 TENet：数据与快速运行

数据布局：

```text
parent/
├── TENet/
│   ├── TENetMechanism/
│   └── ViTPruning/
└── dataset/
    ├── train/<class_name>/*.JPEG
    ├── val/<class_name>/*.JPEG
    └── output/                       # ignored by these loaders
```

类别由目录名推断；默认从 train/val 共有类别中选择子集。图片按需加载为 RGB，训练使用随机裁剪缩放与翻转，验证使用确定性中心裁剪。两者使用同一归一化常数。`output/` 不参与读取。

在已有匹配硬件的 PyTorch 环境中，从仓库根目录执行：

```bash
cd TENetMechanism
pip install -r requirements.txt
python train.py --help
python train.py
```

| 参数 | 当前默认值 |
| --- | --- |
| 图像尺寸 | 64×64 |
| 图聚合层 | 3 |
| 初始通道 / 每层增量 | 3 / 8 |
| 分类头 | 4×4 池化 → 432 → 256 → 实际类别数 |
| 数据子集 | 前 50 个共有类别，每类最多 500 张训练图 |
| 验证数据 | 所选类别的全部验证图 |
| 训练 | 100 轮，batch 16，workers 4，seed 42 |

默认形状为 `3×64×64 → 11×32×32 → 19×16×16 → 27×8×8`。类别或图片不足时以实际读取数量为准；类别数至少为 2。

调整子集与模型的完整参数见 [TENetMechanism 使用说明](TENetMechanism/README.md)。默认权重和日志保存到 `TENetMechanism/runs/<UTC时间戳>/`，含 `last.pth`、`best.pth`、`config.json`、`manifest.json` 和 `metrics.json`。训练启动打印参数量，每轮打印训练/验证 loss 与准确率。

```bash
# Analyze the latest last.pth; check the printed checkpoint path.
python analyze.py --layer 0

# Integrated normalization study on the same checkpoint.
python analyze.py --layer 0 --normalization-study

# Use an explicit path for controlled comparisons.
python analyze.py --checkpoint runs/your_run/last.pth --layer 0 --normalization-study

# Test with a temporary JPEG dataset, including training, analysis and plots.
python -m unittest discover -s tests -v
```

`runs/your_run/last.pth` 需要替换为实际路径。默认路径依据脚本位置确定，显式相对路径依据终端工作目录解释。自动选择使用修改时间最新的 `last.pth`，不保证它就是目标实验。

分析写入所选 checkpoint 目录下的 `analysis/`。首先查看 `report.md`、`class_pairs.csv`、`samples.csv`；归一化实验另生成 `normalization_report.md` 和五组对比 CSV。类别对梯度图使用 `inspect_pair.py`，固定权重干预和其他层的分析方法见工作区说明。

## 5. ViT 评分剪枝分支

[ViTPruning](ViTPruning/DATASET_TRAINING.md) 实现纯 PyTorch ViT，以及四种可选方案：

| 方案 | 行为 |
| --- | --- |
| identity | 不剪枝 |
| magnitude | 按 patch token 的特征范数评分，保留 Top-K |
| linear | 可学习线性评分，训练时使用直通门控近似 |
| tenet_local | 用局部十节点图的顶点能量贡献评分，再保留 Top-K |

`tenet_local` 没有使用独立 TENet 的递归聚合、可学习图投影和双流结构；输出仍是选中的原有 token。它与“将四个位置聚合成一个新表示”是两种不同机制，不能互相代替验证。

从仓库根目录执行：

```bash
cd ViTPruning
python train.py --pruner identity
python train.py --pruner magnitude
python train.py --pruner linear
python train.py --pruner tenet_local
python compare_results.py
```

该分支目前默认 20 类、输入 128×128、patch 16、维度 256、12 层、4 个头，100 轮；与独立 TENet 的默认配置不同。四组使用独立输出目录，完成的实验追加到 `checkpoints/training_results.jsonl`，对比脚本读取末尾四条非空记录。总训练耗时不是推理延迟，现有实验尚不能证明硬件加速。

## 6. 历史实现与实验资料

| 文件或目录 | 说明 |
| --- | --- |
| [Code/train.py](Code/train.py) | MNIST / CIFAR-10 / RGB256（plantvillage）训练入口 |
| [Code/model_common.py](Code/model_common.py) | RGB32、RGB256 使用的历史双流图层 |
| [Code/model_mnist28.py](Code/model_mnist28.py) | 从 history 导入 MNIST current / legacy 快照 |
| [Code/model_rgb32.py](Code/model_rgb32.py) / [Code/model_rgb256.py](Code/model_rgb256.py) | RGB 分类网络 |
| [Code/model_arch.py](Code/model_arch.py) / [Code/model_arch_legacy.py](Code/model_arch_legacy.py) | 早期实现参考 |
| [Code/TENet_PlantVillage.py](Code/TENet_PlantVillage.py) | 独立 PlantVillage 训练流程 |
| [Code/analysis/](Code/analysis/) | 层级探针、有效秩、CKA、φ 可视化与输入梯度 |
| [Code/history/](Code/history/) | 按日期保存的模型快照 |
| [阶段发现](docs/10_phase1_mechanism/02_findings.md) / [结果归档](docs/90_results/phase1_probe/README.md) | 早期微训练预算下的记录与原始指标 |

历史入口示例，从仓库根目录执行：

```bash
cd Code
python train.py --dataset mnist --epochs 20
python train.py --dataset cifar10 --rgb32-num-layers 2 --alpha 8 --epochs 50
python -m analysis.run_probe --dataset cifar10 --alpha 8 --num-layers 2 --epochs 5
```

历史 RGB256 入口从一个类别目录根目录内部划分数据；不要直接套用新平台的 train/val 根目录。详细选项以各脚本 `--help` 为准。

**实现版本差异：** TENetMechanism 修正了历史图层的多通道窗口分组顺序。旧代码在 `B,C,H/2,W/2,2,2` 排列后直接 reshape 为 `window,C,4`，多个通道和窗口可能被混排；新实现使用 `B,H/2,W/2,C,2,2`。新平台还使用了可配置池化分类头。历史权重不应直接用于新平台，跨版本精度变化不能全部归因于算法改进。

## 7. 结论与复现边界

- 先从真实类别的区分退化出发，再追踪算子；人为构造的差异只用于解释已发现的问题。
- 探针比较统一描述符尺寸与预算，但仍需检查池化、投影、随机种子和读取器容量的影响。
- discovery / confirmation 是探索性拆分；若验证集已参与模型选择，它就不是完全独立的最终测试集。
- 固定权重干预可能引入分布偏移；推断模型修复有效还需要重新训练及独立评估。
- 当前代码支持机制实验和对照训练；不宣称已证明关键任务信息保真、注意力替代能力、代数拓扑不变量或硬件优势。

## 8. 许可证

本项目采用 [Apache License 2.0](LICENSE)。
