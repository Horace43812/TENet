# TENet 任务信息机制实验工作区

目的：从真实类别的区分退化出发，定位 TENet 单层图化、矩阵变换、聚合中的候选问题，再研究递归。这里是独立 TENet，不包含 Transformer 或 token 评分剪枝。

## 来源与边界

- 图层基于 `../Code/model_common.py`，网络参考 `../Code/model_rgb32.py`。保留 4 顶点＋6 边节点、`diag(z) L diag(z)` 矩阵、图归一化、10×10 投影、trace 和内容流等机制。
- 参考 `../docs/10_phase1_mechanism/` 的信息流探针研究。旧研究为微训练预算，不能把 probe 或 CKA 下降直接当作不可逆信息损失的证明。
- **修正旧版窗口分组顺序**：旧代码先排列 B,C,H/2,W/2,2,2 再 reshape 成窗口,C,4，C>1 且有多个窗口时会混排。这里先排列 B,H/2,W/2,C,2,2；测试逐元素检查真实局部窗口。旧文件未修改，因此旧 checkpoint 不支持直接迁入，旧结果也不能视为同实现基线。
- 分类头使用可配置空间池化＋MLP，避免单层大分辨率下全展开分类头过大；这是相对旧网络的显式变化。默认保留 RGB 输入，增加 `--channels` 才启用可学习 1×1 stem。
- `datasets.py` 是 ViTPruning 加载器的本地副本，使用同一 ImageFolder 类别排序、抽样种子和预处理。实验不依赖 ViTPruning 的训练代码。

## 安装、测试和独立训练

### 按功能查看参数

`python train.py --help` 按六组显示参数：数据集与子集、主干与分类头、图分支开关、训练与设备、保存与续训、假数据测试。命令行分组和帮助文本均使用英文，并显示默认值。

### 标准训练命令

以下每行依次对应：数据路径、类别范围、图片数量、主干、分类头、启用分支、关闭分支、训练批次、优化参数、输出目录。整段可直接复制到 Bash；反斜杠后不要添加空格或注释。

```bash
python train.py \
  --dataset imagefolder --data-dir ../../dataset \
  --common-classes-only --class-start 0 --class-count 50 \
  --train-per-class 500 --val-per-class 0 \
  --image-size 64 --layers 3 --channels 3 --alpha 8 \
  --head-grid 4 --head-hidden 256 --dropout 0.1 --readout topology \
  --learnable-phi --trace --edge --cross \
  --no-directed --no-chebyshev --no-spatial \
  --epochs 100 --batch-size 16 --workers 4 --seed 42 \
  --lr 3e-4 --weight-decay 0.05 --label-smoothing 0.1 \
  --output-dir runs/tenet_3layer_50class_seed42
```

省略 `--output-dir` 会自动生成时间戳目录，适合重复实验。其他参数都等于当前默认值，也可以直接运行 `python train.py`。

在服务器已有 PyTorch 环境中执行：

```bash
cd ~/TENet/TENetMechanism
pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py
```

默认：TENet 同级 `dataset/train/<类别>/*.JPEG`、`dataset/val/<类别>/*.JPEG`；前 50 个共有类别，每类最多 500 张训练图，验证使用这些类别全部图片；输入 64×64、3 层、RGB 3 通道、每层增加 8 通道、分类头 4×4 池化＋256 隐藏单元；100 轮、batch 16、workers 4。根目录 `output` 不参与读取。初始使用 64 分辨率是为了控制显式 10×10 图矩阵内存，并非沿用 ViT 的模型尺寸。

```bash
# 指定单层实验目录（非空目录禁止覆盖）
python train.py --layers 1 --alpha 8 --output-dir runs/single_layer

# 恒定通道：每层不扩宽；channels=16 会启用 1×1 stem
python train.py --channels 16 --alpha 0 --layers 1

# 多层递归，其他预算保持一致
python train.py --layers 3 --alpha 8

# 数据范围可调整；0 表示全部；使用严格类别匹配加 --no-common-classes-only
python train.py --class-count 10 --train-per-class 100 --val-per-class 20 --epochs 2

# 无需真实数据的 CPU 冒烟训练
python train.py --dataset fake --image-size 16 --layers 1 --epochs 1 --batch-size 4 --workers 0 --cpu

# 中断续训：保留原模型和训练参数，包括原 epochs 总数
python train.py --layers 1 --alpha 8 --resume runs/single_layer/last.pth
```

其他模型参数：`--image-size`、`--layers`、`--channels`、`--alpha`、`--head-grid`、`--head-hidden`、`--dropout`、`--readout topology|content|both`。每层 H/W 减半；alpha 控制每层通道增量；head-grid 不得超过最终空间边长。

分支开关均支持正反形式：`--learnable-phi / --no-learnable-phi`、`--trace / --no-trace`、`--edge / --no-edge`、`--cross / --no-cross`、`--directed / --no-directed`、`--chebyshev / --no-chebyshev`、`--spatial / --no-spatial`。默认前三项及 cross 开启，其余关闭，接近 RGB32 历史默认。`--help` 显示全部参数。

每次训练默认写入独立 `runs/<UTC时间戳>/`：`config.json`、`manifest.json`（每张图片路径和标签）、`metrics.json`、`best.pth`、`last.pth`。支持续训，但不承诺恢复随机状态后逐位复现。manifest 检查路径和标签，不校验图片内容哈希。

## 层内机制分析

默认训练保存到时间戳目录，`runs/single_layer` 仅在训练时显式指定了该目录才存在。
不知道具体路径时，直接运行 `python analyze.py --layer 0`：自动使用工作区 `runs/` 下按修改时间最新的 `last.pth`，启动时打印实际路径。
也可传 `--checkpoint latest`、具体 `.pth` 文件或训练目录（自动取其 `last.pth`）。显式路径错误时列出可用文件，不会悄悄切换实验。自定义输出到 `runs/` 之外的模型需显式指定路径。

训练启动时打印总参数量和可训练参数量，每轮打印学习率、训练/验证 loss 与准确率，结束时打印 `last.pth` 的完整路径。

```bash
python analyze.py --checkpoint runs/single_layer/last.pth --layer 0

# 把疑点收窄到图归一化前后
python analyze.py --checkpoint runs/single_layer/last.pth --layer 0 \
  --before graph_raw --after graph_normalized

# 非线性读取能力对照，其他测量参数保持一致
python analyze.py --checkpoint runs/single_layer/last.pth --probe-hidden 64
```

`--layer` 从 0 开始；多层训练后对不同层分别运行。默认采集以下阶段：

| 阶段 | 含义 |
| --- | --- |
| input / content_input | 本层两个流的输入 |
| vertices / nodes | 正确分组的四顶点；加入六边后的十节点 |
| graph_raw | 未归一化的 `diag(z) L diag(z)` |
| graph_normalized | 归一化后、投影前矩阵；Chebyshev 开启时为混合结果 |
| projected | 10×10 投影后，补充分支融合前 |
| trace | 图矩阵对角项之和 |
| pre_activation | trace/内容引导融合后，BN/ReLU 前 |
| output / content_output | 本层两个流的最终输出 |

网络冻结并使用 eval 模式。训练集与验证集均使用确定性中心裁剪。每个阶段先汇总为同样的空间网格（默认 4×4），再经固定 signed-hash 投影到相同维度（默认 256），使探针参数量相同；只在 train 拟合标准化及预测器，训练固定轮数，不用 val 选择 probe 最佳轮次。分类标签来自整张图片，不给局部窗口强加类别标签。

**测量本身也有信息瓶颈**：池化、哈希碰撞、读取器容量都可能造成差异。使用 `--grid`、`--probe-dim`、`--probe-seed` 和 `--probe-hidden` 重复检查。`input -> vertices` 只是可逆重排，也可作为测量失真的参照，不能把它的 probe 差异解释为算子丢失。

验证样本按类别固定分成 discovery / confirmation 两半，各类至少 2 张。仅根据 discovery 排序类别对，confirmation 报告相同类别对的变化。训练过程已使用 val 做监测或选 checkpoint，因此 confirmation 只是探索性复核，不是完全未触碰的最终测试集。`last.pth` 可避免额外挑最佳轮次，但不能消除人工调参泄漏。

输出位于 checkpoint 目录的 `analysis/layer<编号>_<干预>_<时间>/`：

- `report.md` / `report.json`：各阶段预测能力、候选退化类别对及复核结果。
- `class_pairs.csv`：全部类别对，双向互相误判次数及增量；分母是这两个类别的样本总数，其他误判不计入该对指标。
- `samples.csv`：所有验证图片路径、真实类别、聚合前后预测，标记 lost（对→错）和 recovered（错→对）。
- `confusion.json`：各阶段完整混淆矩阵，行真实类别、列预测类别；类别顺序见 report。
- `probes.pth`：预测概率、标签、探针权重、训练集标准化参数、固定分组和模型预测，支持复查。
- `--save-features` 可额外保存全部探针描述符；默认不缓存巨大的原始图矩阵。样本多时可用 `--stages input output` 控制分析内存。

## 从真实类别对定位线索

先阅读 report 和 samples，选定实际退化的类别对，再画聚合前后读取器的类别差值梯度：

```bash
python inspect_pair.py --analysis-dir runs/single_layer/analysis/实际目录 \
  --class-a n01440764 --class-b n01443537 --max-samples 8
```

类别名替换为报告中的实际目录名。默认只看 discovery，优先显示 lost 样本。输出原图中心裁剪、before/after 的类别 A-B margin 输入梯度图、原始梯度及样本信息。不同图独立色标，不能直接按亮度比较幅值；梯度表示局部敏感性，不是语义因果证明，也不证明该区域已被删除。

## 单点干预

### 集成 A–E 归一化实验

无需重新训练，使用同一个 checkpoint 一次完成五组读取实验：

```bash
python analyze.py --checkpoint /path/to/last.pth --layer 0 --normalization-study
# 也可自动使用最近一次训练的 last.pth
python analyze.py --layer 0 --normalization-study
```

| 组 | 探针输入（池化和降维之前构造） |
| --- | --- |
| A | 原始图矩阵 G |
| B | (G − μ) / s |
| C | s × B，补回尺度 |
| D | B + μ，单独加回偏移，不是完整恢复 |
| E | s × B + μ，完整恢复检查 |

μ、s 均按原代码逐局部窗口、逐通道计算；s 使用相同的 std 和 1e-6 下限。实验分支仅用于观测，不改变网络前向输出、权重或 BN 统计。每组使用相同的描述符尺寸、固定随机投影、探针初始化种子、训练预算；探针各自在 train 拟合，标准化也只拟合 train。

自动替换 stages 为 A–E，before/after 固定为 A/B。每批原始矩阵上检查 E≈A（rtol=1e-5、atol=1e-5）；失败直接停止。默认单拉普拉斯模式支持，Chebyshev 混合以及同时指定 intervention 时拒绝运行。

除了原有分析文件，还输出 `normalization_report.md`、`normalization_report.json`、`normalization_summary.csv` 和 `normalization_pairs.csv`。报告包括五组准确率、相对 B 的变化、发现/复核子集准确率、恢复误差、A/E 预测一致率，以及固定类别对在五组中的互相误判数。类别对始终按 A→B 在 discovery 上的退化排序，confirmation 不重新选对。

先检查 E/A 恢复误差和预测一致性，再看 C/D 是否改善 B，以及是否改善同一批真实退化类别对。E 只检验实现，不是修复效果的证据；C/D 的改进表示统计量帮助当前读取器，尚不能证明整个网络应采取该修改。重复实验保持 checkpoint 不变，修改 `--probe-seed 124`、`--probe-dim 512` 或 `--grid 8`（不得超过该层输出边长）检查测量敏感性。

### 固定权重分支干预

```bash
python analyze.py --checkpoint runs/single_layer/last.pth --layer 0 --intervention no_trace
python analyze.py --checkpoint runs/single_layer/last.pth --layer 0 --intervention no_cross
python analyze.py --checkpoint runs/single_layer/last.pth --layer 0 --intervention no_graph_norm
```

仅在目标层关闭 trace、双向 cross guidance 或图归一化。模型权重、BN 统计及在原表示上训练的 probe 保持不变；输出 baseline/intervened 的模型及固定探针结果，逐样本预测保存在 `intervention.pth`。这测的是原模型对该操作的依赖，可能受分布偏移影响，不能等同于修复成功。no_graph_norm 不支持 Chebyshev 混合模式。

进一步因果研究需要根据实际类别线索提出针对性修复，并在独立数据上复核。平台不会把 probe 下降、梯度图或某次干预结果自动判为信息损失，也不输出没有理论依据的“信息保留百分比”。

## 测试范围

`tests/test_platform.py` 覆盖多 batch/多通道窗口顺序、均值对应关系、观察器不改变前向、梯度、干预有限值、discovery/confirmation 无交叉及真实 JPEG 临时目录上的训练→分析→梯度图端到端链路。
