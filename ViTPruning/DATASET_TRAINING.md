# 服务器数据集训练

## 当前默认配置与结果汇总

直接运行 `python train.py` 使用 ImageFolder：TENet 同级 `dataset`，共有类别中从下标 0 开始选 10 类，每类训练/验证最多 100/20 张；输入 64、patch 8、维度 192、12 层、3 个头；magnitude 在第 6 层后保留 50% token；50 轮、batch 32、workers 4、lr 3e-4、weight decay 0.05、seed 42。默认输出目录为 `checkpoints/subset_common_magnitude`。

每次成功训练完成，自动向 `ViTPruning/checkpoints/training_results.jsonl` **末尾追加**一条 JSON 记录，不覆盖之前的记录。记录依次包含运行 ID 和 UTC 时间、本次完整配置及类别映射、汇总指标、逐轮训练/验证 loss 和准确率及学习率。准确率单位为百分比；耗时为本次程序运行耗时。续训记录包含 checkpoint 中已有的历史，并注明本次起始轮次；没有执行新轮次时不重复导出。

可用 `--results-file /path/to/results.jsonl` 更换汇总文件。JSONL 每行是一次实验的完整 JSON，便于后续读取比较。原有 `config.json`、`metrics.json`、`best.pth`、`last.pth` 仍按输出目录更新；需要保留每次的模型权重时，请使用不同 `--output-dir`。多个训练进程应使用不同的结果文件。

取消共有类别筛选使用 `--no-common-classes-only`。使用 fake/cifar10 时，未显式指定的类别筛选参数自动关闭；例如 `python train.py --dataset fake --epochs 2` 仍可用于无真实数据验证。

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

如果验证集缺少部分训练类别，简单验证时可在命令中追加 `--common-classes-only`。程序先取 train/val 同名类别的交集并排序，再应用 `--class-start` 和 `--class-count`；例如 `--common-classes-only --class-count 10` 会选前 10 个共有类别，而不是先选训练集前 10 类再删除缺失项。启动时打印共有类别数和排除数量，实际类别映射保存在 `config.json`。若共有类别不足两类则报错。

此选项不会修改数据，也不会把不同名称的类别强行配对。使用它时，数量参数全为 0 表示全部共有类别，不保证覆盖原始 1000 类。正式全量训练前应核实缺失类别的数据及标签；使用 `--no-common-classes-only` 检查所选训练类别是否都有同名验证目录。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--class-start` | 文件夹名排序后的起始类别下标，从 0 开始 | 0 |
| `--class-count` | 从起点开始取多少类；0 表示剩余全部类 | 10 |
| `--train-per-class` | 每类训练图片上限；0 表示全部 | 100 |
| `--val-per-class` | 每类验证图片上限；0 表示全部 | 20 |
| `--seed` | 控制图片抽样及训练随机种子 | 42 |

例如 `--class-start 100 --class-count 50` 选择排序后的第 101～150 类。类别和图片不足时使用实际可用数量，但必须至少选中两类，每类在两个 split 中都要有可读取的图片。所选类别重新编号为 `0..类别数-1`，train/val 使用同一映射。每类图片随机抽样且不重复；保持文件列表和种子一致即可复现子集。程序仅扫描所选类别，仍需枚举这些类别的图片路径，但不会预先把全部图片读入内存。

## 全量训练

下面使用全部类别和图片，以 224 分辨率从头训练。训练轮数和超参数仍需根据实验调整。

```bash
python train.py \
  --dataset imagefolder \
  --data-dir ../../dataset \
  --no-common-classes-only --class-start 0 --class-count 0 \
  --train-per-class 0 --val-per-class 0 \
  --image-size 224 --patch-size 16 \
  --embed-dim 192 --depth 12 --num-heads 3 \
  --pruner magnitude --prune-at-block 6 --prune-keep-ratio 0.5 \
  --epochs 100 --batch-size 64 --workers 8 \
  --lr 3e-4 --weight-decay 0.05 --seed 42 \
  --output-dir checkpoints/full_magnitude
```

ImageFolder 训练使用随机裁剪缩放、水平翻转及 ImageNet 归一化；验证使用等比例缩放和中心裁剪。`--no-aug` 让训练也使用验证预处理。默认自动使用可用的 CUDA；`--cpu` 强制 CPU。直接运行 `python train.py` 使用本文开头的真实数据子集默认配置。

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
