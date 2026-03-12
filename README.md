# TENet

当前代码已改为使用 PlantVillage 数据集训练（39 类，目录结构为每类一个子目录）。

训练脚本：`Code/TENet_PlantVillage.py`

默认数据集路径：

`C:\Users\20429\Desktop\project\learn_pytorch\new\PlantVillage-Dataset-master`

运行方式：

```bash
python Code/TENet_PlantVillage.py
```

## MNIST 实验脚本（可选）

如果你在做当前 TENet 图化层的 MNIST 对比实验，可使用：

```bash
python Code/train.py
```

当前 `Code/train.py` 默认配置（可通过命令行覆盖）：

- `epochs=20`
- `batch_size=128`
- `lr=8e-4`
- `alpha=3`
- `trace_init=0.005`
- `trace_lr_mult=0.05`
- `phi_temp_init=1.5`
- `phi_residual_init=0.05`
- 默认开启 best checkpoint 保存（路径 `Code/checkpoints/tenet_best.pth`）

常用命令：

```bash
# 使用默认配置训练
python Code/train.py

# 关闭 best 模型保存
python Code/train.py --no-save-best

# 覆盖关键超参
python Code/train.py --alpha 2 --trace-init 0.01 --epochs 30 --lr 5e-4
```
