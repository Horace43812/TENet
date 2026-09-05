# ViTPruning：TENet 的 Transformer 剪枝应用分支

这是与 TENet 主模型隔离的独立实验子项目。它提供一个纯 PyTorch、无 timm 依赖的标准 pre-LayerNorm Vision Transformer，作为后续 Transformer 剪枝方案的统一平台。

默认训练配置使用服务器 ImageFolder 的 10 个共有类别：输入 `3x64x64`、patch size `8`（共 64 个 patch）、class token、可学习位置编码、`192` 维 token、`12` 个 Transformer block、`3` 个 attention head 和 `4x` MLP。每类最多 100 张训练图、20 张验证图，训练 50 轮。直接运行 `python train.py` 即可，完整参数见[训练指南](DATASET_TRAINING.md)。

## 训练

在本目录执行。数据尚未准备时，显式选择 `fake` 数据集生成 RGB 假数据，用于验证从 DataLoader、模型、剪枝接口到训练、验证和保存的完整链路，无需下载数据：

```bash
python train.py --dataset fake --epochs 2 --batch-size 32
```

接入 CIFAR-10 后：

```bash
python train.py --dataset cifar10 --image-size 32 --patch-size 4 --epochs 100 --batch-size 128 --lr 3e-4 --weight-decay 0.05
```

`--embed-dim` 必须能被 `--num-heads` 整除。模型、配置和逐轮指标默认写入 `checkpoints/subset_common_magnitude/`。训练完成后向 `checkpoints/training_results.jsonl` 追加本次配置、汇总指标和逐轮数据，保留已有实验记录；使用 `--results-file` 可指定其他汇总文件。

### 后续接入真实数据

| 参数 | 数据布局 |
| --- | --- |
| `--dataset cifar10` | 自动下载至 `--data-dir` |
| `--dataset imagefolder --data-dir /path/to/data` | `/path/to/data/train/<类别>/...` 与 `/path/to/data/val/<类别>/...` |

新增专用数据集时，在 `datasets.py` 中返回 `(train_dataset, eval_dataset, num_classes)`；下游训练、剪枝、评估和保存逻辑无需改动。

## 中间 Token 剪枝接口

`VisionTransformer` 在第 6 个 Transformer block 后（或 `--prune-at-block` 指定的位置）调用一次 `token_pruner`。默认方案是 `magnitude`：按每个 patch token 的 L2 特征范数筛选，保留 50% token，CLS token 永远保留。后半段 block 因 token 数变少而实际降低计算量。

```bash
# 默认：第 6 个 block 后保留 50% patch token
python train.py

# 无剪枝对照实验
python train.py --pruner identity

# 可学习线性评分对照组
python train.py --pruner linear

# TENet 2x2 局部图化评分
python train.py --pruner tenet_local

# 调整位置和保留率
python train.py --pruner magnitude --prune-at-block 4 --prune-keep-ratio 0.75
```

新方案请放入 `pruning/`，继承 `TokenPruner` 并实现 `forward(tokens)`；tokens 的形状为 `(B, 1 + N, C)`，第 0 位为 CLS token。然后在 `pruning/token_pruners.py` 的 `build_token_pruner` 注册名称即可，无需修改 ViT 主干。

`linear` 是推荐的可学习对照组：每个 token 经过 `Linear(C, 1)` 得到重要性分数，再进行 Top-K 保留。训练时采用直通式 sigmoid 门控，使线性层可由分类损失学习；评估时则执行纯硬 Top-K 剪枝。

`tenet_local` 是无参数的 TENet 局部图化基线。它将 patch token 恢复为二维网格，每个非重叠 `2x2` 窗口形成 4 个顶点与 6 个绝对差分边特征；顶点依据其在 TENet 10 节点拉普拉斯图中的局部 Dirichlet 能量贡献获得分数。它只检验局部图关系是否提供剪枝信号，不包含递归树聚合，因此不能视为完整 TENet 方案。当前要求 patch 网格边长为偶数；默认的 `8x8` 网格满足此条件。

## 适合剪枝的模块

模型遵循通用 ViT 命名，便于按模块定位、统计和替换：

| 类型 | 参数路径示例 |
| --- | --- |
| Patch embedding | `patch_embed.proj` |
| QKV 投影 | `blocks.0.attn.qkv` |
| Attention 输出投影 | `blocks.0.attn.proj` |
| MLP 两层 | `blocks.0.mlp.fc1`, `blocks.0.mlp.fc2` |
| 分类头 | `head` |

`model.iter_prunable_modules()` 会返回全部 `Conv2d` 与 `Linear` 层；请将具体剪枝算法放进 `pruning/`，避免模型本身绑定某一种 score、mask 或重参数化方法。结构化剪枝时，QKV、attention 输出投影和相邻 MLP 的输入/输出维度必须保持依赖一致；单纯置零的非结构化剪枝则没有这个限制。

## 服务器数据集训练

参见[数据集训练指南](DATASET_TRAINING.md)：包含子集选择、完整训练命令、输出文件和断点续训说明。
