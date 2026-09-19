# TENetText：真实文本分类

任务是根据新闻文字判断主题，不是识别文字图片。首个数据集为AG News，4类：世界、体育、商业、科技。
官方训练集120,000条，测试集7,600条。默认从训练集按类取10%做dev，实际train/dev/test为108,000/12,000/7,600。
首次自动下载原始CSV；显示准备阶段与下载进度，连接超时30秒，失败最多尝试3次，验证行数后才发布文件。
下载重试从头开始，当前不支持断点续传；已有完整CSV不再下载。
数据来源：[AG News说明](https://huggingface.co/datasets/sh0416/ag_news)，
[CSV镜像](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)。

## 三个同时建立的对照

- `tenet`：连续4词元窗口、步长1；每个嵌入维度生成4顶点和6个绝对差值边，构造10×10的 `diag(z)Ldiag(z)`，线性投影、ReLU、掩码平均池化、分类。
- `cnn`：核宽4的一维卷积、ReLU、同样的掩码平均池化、分类。
- `mean`：词向量掩码平均、线性投影、ReLU、分类。它是简单平均基线，不是完整fastText（没有词组n-gram）。

三者相同词元化、词表、嵌入维度、嵌入初始化、数据划分与优化流程；不加载预训练模型。
词表仅用实际训练部分构建，dev/test不参与；PAD=0，UNK=1。右侧补齐符不参与池化或图归一化。
TENet是图方法的一维改版，保留有序窗口位置；不代表原图像模型可以直接用于文本。
当前只移植基础图聚合及raw/centered/standardized开关，尚不含图像版本的trace/cross/phi等补强分支。
输入默认按单词和标点切分、小写、最多128个词元；图窗口不足4个词元时对缺失节点和边做掩码。
standardized按有效矩阵位置计算总体方差，不与图像版的sample std数值完全相同。

## 安装与第一轮实验

在原PyTorch环境执行即可，新增依赖只有现有的torch/numpy/tqdm：

```bash
cd ~/TENet/TENetText
python -m pip install -r requirements.txt

# 第一轮建议三个模型都跑10轮，快速判断方向（程序默认仍为50轮）
python train.py --model-version tenet --normalization raw --epochs 10 --output-dir runs/agnews_tenet_raw_e10
python train.py --model-version cnn --target-backbone-params 25684 --epochs 10 --output-dir runs/agnews_cnn_e10
python train.py --model-version mean --target-backbone-params 25684 --epochs 10 --output-dir runs/agnews_mean_e10
```

默认TENet嵌入维度16、隐藏维度16，主干与分类头共25,684参数。
同预算开关按最接近的整数宽度匹配，CNN为隐藏宽度372、25,672参数；mean为1223、25,687参数。
总参数还包括词嵌入：若词表达到20,000上限，则另有320,000个嵌入参数。
输出和config.json分别列出 `total`、`embedding`、`backbone_and_head`；参数预算匹配不意味着计算量、隐藏维度或耗时相同。
词表实际不足上限时按实际大小计数；不要把主干25,684称为整个模型的总参数量。

也可固定三个模型相同隐藏宽度，作为另一种结构对照（此时参数量不同）：

```bash
python train.py --model-version cnn --hidden-dim 16 --epochs 10 --output-dir runs/agnews_cnn_width16_e10
python train.py --model-version mean --hidden-dim 16 --epochs 10 --output-dir runs/agnews_mean_width16_e10
```

其他实验与恢复：

```bash
# 归一化切换，仅TENet使用；不要与raw结果混用同一个输出目录
python train.py --model-version tenet --normalization centered --epochs 10 --output-dir runs/agnews_tenet_centered_e10
python train.py --model-version tenet --normalization standardized --epochs 10 --output-dir runs/agnews_tenet_standardized_e10

# 完整50轮训练需新目录；不能把10轮cosine训练直接续成50轮
python train.py --model-version tenet --normalization raw --output-dir runs/agnews_tenet_raw_e50

# 断点续训：原命令保持不变，加--resume
python train.py --model-version tenet --normalization raw --epochs 10 --output-dir runs/agnews_tenet_raw_e10 --resume

# 假数据检查安装与训练流程，不代表真实准确率
python train.py --model-version tenet --smoke --epochs 1 --max-length 16 --output-dir runs/smoke
```

优化器Adam，默认lr=0.001、batch=64、weight decay=0、dropout=0，cosine降至初始lr的10%。
支持 `--lr`、`--weight-decay`、`--dropout`、`--batch-size`、`--embedding-dim`、`--hidden-dim`、`--vocab-size`、`--max-length`、`--seed`。
如果显存不足，三个模型一起减小batch-size；改变输入长度或词表也应对全部对照使用相同设置。
整次训练一个进度条；dev选best.pth，完成后只评估一次test。恢复保存优化器、调度器和随机状态。
恢复必须配置、代码、CSV内容和划分一致；仅保证在相同设备/环境下的恢复可复现，CUDA不保证逐位确定。

## 自己的中文/英文文本数据

提供 `train.csv` 和 `test.csv`，UTF-8编码、表头严格为 `label,text`；文本含逗号时用标准CSV引号。
标签可以是字符串，所有测试标签必须在训练集中出现；dev自动从train划分。

```csv
label,text
positive,这个产品很好用
negative,使用体验很差
```

每类至少两条训练样本。中文建议先用字符切分，避免把整句当成单词：

```bash
python train.py --dataset csv --data-dir /path/to/text_data --tokenizer char --model-version tenet --normalization raw --epochs 10 --output-dir runs/chinese_tenet
python train.py --dataset csv --data-dir /path/to/text_data --tokenizer char --model-version cnn --target-backbone-params 25684 --epochs 10 --output-dir runs/chinese_cnn
python train.py --dataset csv --data-dir /path/to/text_data --tokenizer char --model-version mean --target-backbone-params 25684 --epochs 10 --output-dir runs/chinese_mean
```

## 结果与记录

每个输出目录：`config.json`、`vocab.json`、`split.json`、`best.pth`、`last.pth`、`summary.json`、`result.json`。
检查 `result.json` 的test.accuracy_pct。summary只保留最佳摘要；last checkpoint另保留最后一轮摘要用于恢复。
默认加入同一个 `../TENetTiny/records/training_records.md`，与图像实验集中存放；用dataset/model区分。
完整JSON记录另含参数拆分。也可用 `--records-dir records` 独立记录文本实验。
网络结构测试、合成训练测试不代表AG News上的实际效果；先跑上面三个10轮实验，比较准确率、参数量和训练耗时。
