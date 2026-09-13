# MNIST 2,295参数原始G实验

独立入口，不修改 TENetMechanism 已冻结实验使用的源码。

骨架沿用 `Code/history/model_arch_legacy_20260314.py` 的两层小模型：
`1×28×28 → 2×14×14 → 3×7×7 → Linear(147,10)`。
本模型使用基础G：四顶点、六绝对差边、固定关联拉普拉斯矩阵，计算
`diag(z) L diag(z)` 后直接可学习投影，默认不做图矩阵局部归一化。
投影后仍有 BatchNorm/ReLU，输入仍使用MNIST固定均值/标准差。
可训练参数：第一层206，第二层609，分类头1480，合计2295。

这不是旧模型逐项复刻：移除了可学习phi、边增强、trace及门控；
修正多通道局部窗口排列，分类头输出原始logits而不加ReLU。
因此不能将与旧成绩的所有差异归因于取消归一化；请使用本入口三种模式作同结构对照。

在已有 PyTorch、torchvision、tqdm 环境中：

```bash
cd ~/TENet/MNISTTiny
python -m unittest test_model -v
python train.py --normalization raw --epochs 100 --seed 42
```

MNIST自动下载/复用到 `../Code/data`，可用 `--data-dir` 更改。
固定从官方训练集每类划出500张dev，共55,000训练/5,000开发；不同训练种子保持划分相同。
无数据增强。Adam，lr=0.001，余弦衰减到0.0001，batch128，默认100轮。
默认使用可用CUDA，`--cpu` 强制CPU。
官方10,000张test只在全部训练结束后评估last，不用test挑选epoch。
历史曾查看MNIST测试集，因此输出不能宣称全新独立最终验证。

同结构归一化对照（参数量完全相同）：

```bash
python train.py --normalization standardized --epochs 100 --seed 42
python train.py --normalization centered --epochs 100 --seed 42
```

默认输出 `runs/<normalization>_s<seed>/`，包含配置、数据划分、每轮train/dev指标、
best/last权重、优化器和随机状态，以及最终 `result.json`（准确率以百分数表示）。
已有非空目录不覆盖。中断后原命令加 `--resume`，恢复最近完整epoch；
模型、数据路径、训练轮数等参数须保持一致。跨设备不保证逐位复现。

```bash
python train.py --normalization raw --epochs 100 --seed 42 --resume
```

无需下载数据的流程自检（合成数据成绩没有研究意义）：

```bash
python train.py --smoke --cpu --epochs 1 --output-dir runs/smoke
python train.py --smoke --cpu --epochs 1 --output-dir runs/smoke --resume
```
