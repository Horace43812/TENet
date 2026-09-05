# 服务器数据集训练

在已安装 PyTorch、torchvision 的环境中，从 `~/TENet/ViTPruning` 运行以下命令。查看全部参数：`python train.py --help`。

## 数据布局

```text
~/
├── TENet/ViTPruning/
└── dataset/
    ├── train/
    │   ├── n01440764/*.JPEG
    │   └── ...
    ├── val/
    │   ├── n01440764/*.JPEG
    │   └── ...
    └── output/                 # 不读取、不修改
```

`--data-dir` 默认根据脚本位置定位到 TENet 同级的 `dataset`，不受运行目录影响。显式相对路径则相对于运行命令的目录。本次选中的训练类别必须在验证集中存在同名目录；验证集多出的目录及未选中类别的差异不影响子集训练；支持 `.JPEG` 等 ImageFolder 图片格式，加载为 RGB。类别数自动推断，不需要设置 `--num-classes`。

## 小规模验证

先选 10 类，每类最多 100 张训练图、20 张验证图，训练 2 轮：

```bash
cd ~/TENet/ViTPruning
python train.py \
  --dataset imagefolder \
  --data-dir ../../dataset \
  --class-start 0 --class-count 10 \
  --train-per-class 100 --val-per-class 20 \
  --image-size 64 --patch-size 8 \
  --embed-dim 192 --depth 12 --num-heads 3 \
  --pruner magnitude --prune-at-block 6 --prune-keep-ratio 0.5 \
  --epochs 2 --batch-size 32 --workers 4 \
  --lr 3e-4 --weight-decay 0.05 --seed 42 \
  --output-dir checkpoints/subset_magnitude
```

这是数据流和训练可用性验证，低分辨率、少量数据和少量轮次的准确率不代表完整模型性能。如果显存不足，减小 `--batch-size`；排查数据加载问题时使用 `--workers 0`。

## 调整使用范围

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--class-start` | 文件夹名排序后的起始类别下标，从 0 开始 | 0 |
| `--class-count` | 从起点开始取多少类；0 表示剩余全部类 | 0 |
| `--train-per-class` | 每类训练图片上限；0 表示全部 | 0 |
| `--val-per-class` | 每类验证图片上限；0 表示全部 | 0 |
| `--seed` | 控制图片抽样及训练随机种子 | 42 |

例如 `--class-start 100 --class-count 50` 选择排序后的第 101～150 类。类别和图片不足时使用实际可用数量，但必须至少选中两类，每类在两个 split 中都要有可读取的图片。所选类别重新编号为 `0..类别数-1`，train/val 使用同一映射。每类图片随机抽样且不重复；保持文件列表和种子一致即可复现子集。程序仅扫描所选类别，仍需枚举这些类别的图片路径，但不会预先把全部图片读入内存。

## 全量训练

下面使用全部类别和图片，以 224 分辨率从头训练。训练轮数和超参数仍需根据实验调整。

```bash
python train.py \
  --dataset imagefolder \
  --data-dir ../../dataset \
  --class-start 0 --class-count 0 \
  --train-per-class 0 --val-per-class 0 \
  --image-size 224 --patch-size 16 \
  --embed-dim 192 --depth 12 --num-heads 3 \
  --pruner magnitude --prune-at-block 6 --prune-keep-ratio 0.5 \
  --epochs 100 --batch-size 64 --workers 8 \
  --lr 3e-4 --weight-decay 0.05 --seed 42 \
  --output-dir checkpoints/full_magnitude
```

ImageFolder 训练使用随机裁剪缩放、水平翻转及 ImageNet 归一化；验证使用等比例缩放和中心裁剪。`--no-aug` 让训练也使用验证预处理。默认自动使用可用的 CUDA；`--cpu` 强制 CPU。直接运行 `python train.py` 仍使用 fake 数据；真实数据必须明确指定 `--dataset imagefolder`。

比较剪枝方案时，保持数据参数、种子和模型尺寸一致，仅替换 `--pruner` 和输出目录：`identity` 为无剪枝对照，`magnitude` 为范数评分，`linear` 为可学习评分，`tenet_local` 为 TENet 局部图评分。

## 输出与续训

每个实验使用独立的 `--output-dir`，保存：

- `config.json`：运行参数、模型配置、所选类别及标签映射、样本数和抽样参数。
- `metrics.json`：逐轮训练/验证 loss 和 top-1 accuracy（百分比）。
- `best.pth`：验证准确率最高的模型及训练状态。
- `last.pth`：最近一轮的模型、优化器、学习率调度器及训练记录。

中断后，使用原命令并追加 `--resume checkpoints/full_magnitude/last.pth`；保留原来的 `--epochs` 总轮数以及相同的数据、模型参数。每轮结束保存，恢复时从下一轮继续；当前不保证续训随机状态与未中断训练逐位一致。改变类别、数据范围或分辨率后应开始新实验，不要使用旧 checkpoint 续训。

## 数据流测试

```bash
python -m unittest discover -s tests -v
```

测试会在临时目录创建 `.JPEG` 分类数据，检查子集规模、标签一致性、可重复抽样、全量读取、无效类别及一次 CPU 训练/验证。不会使用或修改服务器数据。

Python 兼容修复已写入 `pruning/token_pruners.py` 和 `datasets.py`：延迟类型注解求值，避免旧版 Python 导入时计算 `int | None` 或 `tuple[...]`。
