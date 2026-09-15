# TENetTiny：历史基准、补强开关与多数据集训练

原 `MNISTTiny` 已改名为 `TENetTiny`。本目录独立运行，不改动 `Code/history` 和
`TENetMechanism` 的已冻结实验代码。统一入口为 `train.py`；`train_datasets.py` 调用同一训练实现，所有开关完全相同。

## 1. 安装与快速检查

在服务器已配置的 PyTorch 环境中：

```bash
conda activate lxl_pytorch
cd ~/TENet
git pull --ff-only origin main
cd TENetTiny
python -m pip install -r requirements.txt
python train.py --help
python -m unittest discover -v
```

依赖已经满足可跳过安装。默认CUDA可用时使用CUDA，`--cpu` 强制CPU。
先查看参数而不下载数据、不训练：

```bash
python train.py --model-version legacy --describe
python train.py --model-version current --describe
python train.py --model-version current --head linear --no-directed --describe
python train.py --model-version tiny --describe
```

## 2. 模型版本与2,429参数基准

以下参数量均指MNIST、28×28灰度、两层、alpha=1、10类：

| 选择 | 结构/默认配置 | 可训练参数 |
| --- | --- | ---: |
| `--model-version legacy` | 历史单流图层，phi/trace/edge，Linear(147,10)+ReLU分类头 | 2,360 |
| `--model-version current` | 历史双流图层，phi/trace/edge/directed/cross，147→48→10分类头 | 8,555 |
| `--model-version current --head linear --no-directed` | 双流紧凑配置，147→10线性分类头 | **2,429** |
| `--model-version tiny` | 之前的纯G骨架，不含补强，147→10线性分类头 | 2,295 |

current/legacy的默认配置对应仓库 `Code/history/model_arch_{current,legacy}_20260314.py`
中的模型构造函数默认值；测试已验证默认权重可严格加载、前向结果一致。
**这些默认值并非旧 `Code/train.py` 的CLI默认值**：该CLI还会覆盖alpha、trace初始化、有向流等选项。

2,429参数配置由保存的current结构得到，与旧文档中的参数数目吻合；
但96.95%(45)的完整旧训练配置和当时分类头激活尚未核实，不能仅凭参数数目宣称已复现成绩。
如果当时输出层带ReLU，使用 `--head linear-relu`，参数量仍为2,429；新默认线性头不带输出ReLU。

### 2,429参数：只改变局部归一化的直接对照

以下三条只改归一化和输出目录；其余模型、数据和训练设置相同：

```bash
python train.py --dataset mnist --model-version current --head linear --no-directed --normalization standardized --window-layout historical --epochs 100 --seed 42 --output-dir runs/mnist_2429_std_s42
python train.py --dataset mnist --model-version current --head linear --no-directed --normalization raw --window-layout historical --epochs 100 --seed 42 --output-dir runs/mnist_2429_raw_s42
python train.py --dataset mnist --model-version current --head linear --no-directed --normalization centered --window-layout historical --epochs 100 --seed 42 --output-dir runs/mnist_2429_centered_s42
```

### 历史current、legacy默认结构及raw版本

```bash
python train.py --dataset mnist --model-version legacy --normalization standardized --output-dir runs/mnist_legacy_std
python train.py --dataset mnist --model-version legacy --normalization raw --output-dir runs/mnist_legacy_raw
python train.py --dataset mnist --model-version current --normalization standardized --output-dir runs/mnist_current_std
python train.py --dataset mnist --model-version current --normalization raw --output-dir runs/mnist_current_raw
python train.py --dataset mnist --model-version tiny --normalization raw --output-dir runs/mnist_tiny_raw
```

不带 `--output-dir` 时自动生成包含数据集、版本、归一化、种子和配置哈希的目录。
同样的配置再次启动需要 `--resume`；更改补强开关不会静默覆盖另一组实验。

## 3. 补强手段与模型开关

所有版本和所有数据集都接受下列正反开关；未指定时继承版本默认值。
在legacy/tiny上打开原本没有的机制表示一个扩展版本，不再是历史默认基准。

兼容旧名称：`--trace-branch`、`--edge-enhance`、`--directed-flow`、`--cross-guidance`、
`--use-chebyshev`、`--use-spatial-bridge`及它们的 `--no-...` 形式；`--no-aug` 等同 `--no-augment`。

| 开关 | 作用 | legacy默认 | current默认 | tiny默认 |
| --- | --- | --- | --- | --- |
| `--learnable-phi / --no-learnable-phi` | 绝对差边上的可学习残差 | 开 | 开 | 关 |
| `--trace / --no-trace` | 原始图矩阵trace补强及门控 | 开 | 开 | 关 |
| `--edge / --no-edge` | 平方差与乘积边增强 | 开 | 开 | 关 |
| `--directed / --no-directed` | 有向关联差分 | 关 | 开 | 关 |
| `--cross / --no-cross` | 内容与拓扑之间双向引导 | 关 | 开 | 关 |
| `--content / --no-content` | 独立均值内容流 | 关 | 开 | 关 |
| `--chebyshev / --no-chebyshev` | 分别处理L、L²矩阵再可学习混合 | 关 | 关 | 关 |
| `--spatial / --no-spatial` | 深度可分空洞卷积空间补强 | 关 | 关 | 关 |
| `--main-gate / --no-main-gate` | trace融合时主图分支的幅度门控 | 开 | 开 | 关 |

`--cross` 会启用内容流；显式组合 `--cross --no-content` 会报错。
单独 `--no-cross` 保留current的内容流参数；完全移除使用 `--no-cross --no-content`。
为了保留历史结构，current中最后一层内容反馈等参数可能不参与拓扑分类输出；
参数量包括这些历史参数。主门控沿用历史语义：关闭trace时不作用于主分支。
Chebyshev名称沿用旧实现，表示L/L²混合，并非完整标准Chebyshev滤波器。

局部归一化选择（不改变参数量）：

| 选项 | 图矩阵处理 |
| --- | --- |
| `--normalization standardized` | 每个窗口、每个通道减均值后除标准差，标准差下限1e-6 |
| `--normalization centered` | 只减局部均值 |
| `--normalization raw` | 使用原始 `diag(z)Ldiag(z)`，跳过上述两步 |

别名 `--graph-normalization` 也可使用。L²分支启用时同样遵守所选归一化。
输入图片归一化和投影后的BatchNorm不受这个选项影响。
current/legacy默认standardized，tiny默认raw。

窗口排列：

- `--window-layout historical`：保留历史快照排列，用于旧结构对照；多通道时存在窗口/通道混排。
- `--window-layout corrected`：每个真实2×2窗口正确分组。新结构探索可选此项，但对照双方应保持相同。
- current/legacy默认historical，tiny默认corrected。切换不改变参数量，但会改变计算语义。

其他结构参数：`--layers`（默认2）、`--alpha`（每层通道增量，默认1，可为0）、
`--image-size`、`--head linear|linear-relu|mlp`、`--head-hidden`（默认48）、
`--classifier-dropout`（仅MLP分类头使用）。输入边长必须能被 `2**layers` 整除；
例如原生28×28只支持最多两层，三层可显式改成32×32。

补强初始化：`--trace-init .1`、`--phi-temp-init 1.0`、`--phi-residual-init .1`。
它们分别控制trace初始门控、phi温度和phi残差幅度。

### 补强组合示例

```bash
# 在2,429参数骨架上关闭trace，其他部分保持原配置
python train.py --model-version current --head linear --no-directed --normalization raw --no-trace --output-dir runs/mnist_no_trace

# current全补强：包含有向流、L²混合和空间桥接
python train.py --model-version current --normalization raw --learnable-phi --trace --edge --directed --cross --chebyshev --spatial --output-dir runs/mnist_current_all

# legacy也可以打开新机制：此时是legacy分类头的扩展模型
python train.py --model-version legacy --normalization raw --cross --directed --spatial --output-dir runs/mnist_legacy_extended

# 纯G：从tiny起步，使用正确窗口排列
python train.py --model-version tiny --normalization raw --window-layout corrected --output-dir runs/mnist_pure_g
```

## 4. 多数据集训练命令

以下命令使用同一个current紧凑配置、raw图矩阵和修正窗口。改版本、关闭某个补强或更换归一化，
直接使用第3节相同开关；`train.py` 与 `train_datasets.py` 可以互换。

```bash
# MNIST：1×28×28，10类
python train_datasets.py --dataset mnist --model-version current --head linear --no-directed --normalization raw --window-layout corrected --output-dir runs/mnist_current_raw_corrected

# Fashion-MNIST：1×28×28，10类
python train_datasets.py --dataset fashionmnist --model-version current --head linear --no-directed --normalization raw --window-layout corrected --output-dir runs/fashion_current_raw

# CIFAR-10：3×32×32，10类
python train_datasets.py --dataset cifar10 --model-version current --head linear --no-directed --normalization raw --window-layout corrected --output-dir runs/cifar10_current_raw

# CIFAR-100：3×32×32，100类
python train_datasets.py --dataset cifar100 --model-version current --head linear --no-directed --normalization raw --window-layout corrected --output-dir runs/cifar100_current_raw

# 自定义图片：默认RGB64×64；从train/val按相同类别目录读取
python train_datasets.py --dataset imagefolder --data-dir ../../dataset --class-count 50 --train-per-class 500 --image-size 64 --model-version current --head linear --no-directed --normalization raw --window-layout corrected --output-dir runs/imagefolder_current_raw
```

官方数据集下载/复用目录默认 `../Code/data`，用 `--data-dir /实际路径` 改动。
彩色输入/类别数/分辨率改变后，参数量会变化，不再都是2,429；启动时打印实际参数量和完整解析后的开关。

ImageFolder数据布局：

```text
dataset/
  train/class_a/*.JPEG
  train/class_b/*.JPEG
  val/class_a/*.JPEG
  val/class_b/*.JPEG
  test/class_a/*.JPEG    # 可选
  test/class_b/*.JPEG
```

train/val/test的类别目录集合必须一致；val作为开发集，可选test只在训练结束评估。
没有test目录时 `result.json` 中test为null，不把val伪装成测试成绩。
此入口不会去重或移动图片。`--class-count 0` 选全部类别，`--train-per-class 0` 使用全部训练候选。
这两个选项也支持官方数据集。数据清单记录路径/标签或官方索引，续跑会比较清单；不校验图片内容哈希。

## 5. 训练配置、数据划分与成绩口径

默认100轮，batch128，Adam，lr=0.001，余弦衰减到初始lr的0.1倍，weight-decay=0、
label-smoothing=0、无增强、训练seed42、workers0。这些是统一的新训练默认值，不代表96.95%旧实验的已核实配方。

| 参数 | 作用 |
| --- | --- |
| `--epochs 100 --batch-size 128` | 总训练轮数与批量 |
| `--lr 0.001 --optimizer adam` | 优化器；也可选adamw |
| `--scheduler none/cosine/warmup-cosine` | 固定学习率、余弦、预热余弦 |
| `--warmup-epochs 3 --min-lr-ratio 0.1` | 预热长度与余弦下限比例 |
| `--weight-decay 0.0001 --label-smoothing 0.05` | 权重衰减与训练标签平滑；评估loss为普通交叉熵 |
| `--trace-lr-mult 0.05` | trace参数组相对主学习率的倍数 |
| `--augment / --no-augment` | 灰度用随机仿射，CIFAR用裁剪翻转，ImageFolder用随机缩放裁剪翻转 |
| `--seed 43 --split-seed 20260913` | 训练种子与独立的数据划分种子 |
| `--workers 4 --threads 4 --cpu` | 数据加载进程、CPU线程、强制CPU |

默认 `--eval-protocol dev-test`：

- MNIST/Fashion-MNIST：官方60,000张训练图中固定每类500张dev，得到55,000训练/5,000开发；官方10,000张test仅最后评估last。
- CIFAR-10：45,000训练/5,000开发/10,000测试。
- CIFAR-100：每类50张dev，同样45,000训练/5,000开发/10,000测试。
- 使用 `--dev-per-class` 调整官方数据集dev数量；训练种子不改变图片划分。
- ImageFolder直接使用train/val/可选test，不使用dev-per-class。

如需对照旧实验“官方训练集全量训练、每轮看test并报告最佳轮”的流程，显式使用：

```bash
python train.py --dataset mnist --model-version current --head linear --no-directed --normalization standardized --eval-protocol test-monitored --output-dir runs/mnist_2429_test_monitored_std
python train.py --dataset mnist --model-version current --head linear --no-directed --normalization raw --eval-protocol test-monitored --output-dir runs/mnist_2429_test_monitored_raw
```

此时不留dev，默认使用全部60,000张MNIST训练图；每轮指标明确标记 `test_monitored`，
结果包含 `best_monitor_pct`、`best_epoch` 和last的test成绩，便于记录“准确率(轮次)”。
它不再是未查看的测试集。不要把该协议的best成绩直接与dev-test协议的last成绩混比。
历史MNIST测试集已经被查看，任何模式都不能将其重新声明为全新独立最终集。

## 6. 保存、续训与旧MNISTTiny结果

每个输出目录保存 `config.json`（完整解析后的模型设置）、`split.json`、`metrics.json`、
`best.pth`、`last.pth`、`result.json`。best按监控集准确率选，结果同时显示最佳轮；默认最终test评估last。

中断后保留原命令所有模型/训练参数，加 `--resume`：

```bash
python train.py --dataset mnist --model-version current --head linear --no-directed --normalization raw --window-layout historical --epochs 100 --seed 42 --output-dir runs/mnist_2429_raw_s42 --resume
```

恢复最近完整epoch，包含优化器、scheduler、torch/Python/NumPy及数据打乱随机状态。
代码、模型、训练配置或数据清单变化会拒绝新格式续跑；新实验换输出目录。
允许调整workers/threads/CPU选择，但跨设备或改变并行方式不承诺逐位复现。

Git不会自动搬迁服务器上被忽略的旧 `MNISTTiny/runs/`。旧结果留在原处即可；
保留的 `train_v1.py` 使用原始2,295参数模型和旧checkpoint格式：

```bash
cd ~/TENet/TENetTiny
# 原始raw实验若中断，直接指定原输出目录；参数与当初保持一致
python train_v1.py --normalization raw --epochs 100 --seed 42 --output-dir ../MNISTTiny/runs/raw_s42 --resume
```

`train_v1.py` 仅用于旧tiny格式，不含新增版本/补强/数据集开关。
新 `--model-version tiny` 保持原tiny数学结构和2,295参数，但使用统一的新checkpoint格式，
不直接加载旧 `last.pth`。迁移前后都不要删除旧权重和日志。

## 7. 无需下载的快速自检

```bash
python train.py --model-version legacy --normalization raw --smoke --cpu --epochs 1 --output-dir runs/smoke_legacy
python train_datasets.py --dataset cifar10 --model-version current --normalization raw --chebyshev --spatial --smoke --cpu --epochs 1 --output-dir runs/smoke_rgb
```

合成数据仅验证运行，不代表真实准确率。测试覆盖历史快照权重/输出一致、2,429参数配置、
三种归一化、多通道排列、所有版本的每个补强开关、五类数据集流程、实际ImageFolder加载、
固定划分及中断恢复与连续训练一致性。
