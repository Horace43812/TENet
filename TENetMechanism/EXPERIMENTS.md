# 集成实验平台操作手册

统一入口是 `experiment.py`。它读取[实验方案](../docs/04_graphification_recursion_protocol.md)，提供固定数据、批量训练、自动续跑、逐层分析和汇总页面。旧 `train.py/analyze.py` 入口仍可使用。

## 1. 最常用的操作

最省操作的方式是在PyTorch环境中运行：

```text
cd TENetMechanism
python experiment.py start --data-dir ../../dataset
```

首次自动初始化默认核心清单并训练；再次执行同一命令自动继续，完成项跳过。已存在项目时沿用冻结配置，不会重置数据或训练轮数。自定义尺寸/轮数仍先用下方init命令；start支持 `--models Q,G,GQQ` 先运行指定部分，也支持 `--analyze` 训练后接着分析。默认不自动选择后续研究候选或读取final。

在装有PyTorch的Python环境中，从仓库根目录执行：

```text
cd TENetMechanism
python experiment.py doctor
python experiment.py init --data-dir ../../dataset
python experiment.py run
python experiment.py report
```

然后双击 `TENetMechanism/experiments/main/report/index.html` 查看结果。训练每轮更新页面，运行中每30秒刷新；页面包含运行进度、失败原因、跨种子汇总、配对差值、参数/运算成本和分析入口，可以按模型名称筛选；无需启动网页服务。中间轮次精度仅显示进度，不进入跨种子统计。另有CSV、Markdown及PNG/SVG曲线可导出。

训练时可在另一个终端执行 `python experiment.py status`，它只读元数据，不受训练锁影响。训练前可执行 `python experiment.py check`，核对代码、train/dev内容、任务配置和已完成权重，并显示磁盘剩余空间；不加载final。改变权重、数据或代码不会被当成正常续跑，错误原因会明确列出。

也可以只运行 `python experiment.py` 打开中文终端菜单。在没有交互终端的环境里，无参数显示帮助。

`doctor`会检查依赖并做一次真实前向/反向；缺依赖时在所选环境执行 `python -m pip install -r requirements.txt`。Windows推荐先在Anaconda Prompt中激活自己的PyTorch环境；服务器直接使用已配置的环境。平台不自动安装或替换CUDA/PyTorch。

**中断或重新开终端后，再执行相同的 `start` 或 `run` 即可。** 完成项跳过，中断项恢复到最近完整保存的轮次，失败原因保留在页面和该运行的 `state.json`，历史失败另存 `failures/`。完整性错误保留原完成记录并标记 `integrity_error`；修复回原始权重、验证通过后清除错误标记。默认遇到错误停止，修好再运行；`--keep-going`可让其他配置继续。

## 2. 初始化后自动做什么

默认选择train/val共有类别中排序前50类，每类最多500张训练图，64×64输入、100轮、batch16。初始化时：

1. 检查图片可解码，计算内容SHA-256用于后续完整性校验；不检测或拦截重复图片，各路径按原始类别保留。
2. 从原train按类别固定拆分开发集，约10%、每类至少2张；剩余池固定抽取训练图。
3. 原val只登记为final，训练/探针使用train/dev。默认将原val标记为“历史已查看或未知”，不会自动声称独立测试。
4. 保存不可变清单、配置与Python源文件快照，生成16个单种子核心运行。

标准数据目录：

```text
dataset/
  train/class_a/*.JPEG
  train/class_b/*.JPEG
  val/class_a/*.JPEG
  val/class_b/*.JPEG
```

若原val确实从未参与训练选择或分析，可在初始化时加 `--final-unseen` 作明确记录。无法通过这个标记消除历史查看事实。若有患者、场景、原图等来源分组，传 `--groups-file groups.json`：JSON将每个相对图片路径映射到同源分组字符串；平台避免组跨train/dev或原val。跨类别同源组暂拒绝自动划分，需要独立的分组划分方案。

训练期间清单和内容校验只访问train/dev；final命令才核验并加载final。初始化会读取final以检查图片可解码并登记哈希，不评估模型。

训练种子与数据抽样已分离。切换训练种子不会更换图片。为避免末批单样本在1×1特征上触发BN错误，所有模型统一在训练集末批恰好为1时丢弃该末批；每轮仍会重新打乱，dev/final不丢样本。

## 3. 首轮模型名称与预算

| 名称 | 实验 |
| --- | --- |
| Reference | 当前默认宽度3、alpha8的完整TENet |
| P / V / Q | 均值、顶点线性、顶点MLP聚合 |
| E | 四顶点＋六绝对差边直接投影 |
| G / Gplus | 基础图矩阵、增加learnable-phi与edge增强 |
| FullS | 相同恒定宽度16的完整双流TENet |
| GQQ / QGQ / QQG | 只在第1/2/3阶段图化 |
| GGQ / GQG / QGG | 在两个阶段图化 |
| G_centered / G_raw | G仅中心化、G不做局部归一化 |

`Q`就是方案的QQQ，`G`就是GGG，避免重复训练相同配置。首轮16次包含B0/B1/B2/B4；从已运行配置中预选最多5个、补43/44两个种子，则总预算最多26次。超过这些范围的扩展单独计数。

默认每个模型/种子独立目录。可以先跑三组：

```text
python experiment.py run --models Q,G,GQQ
python experiment.py status
```

再不带 `--models` 运行，就会补齐其余待运行项。默认选择CUDA（可用时），否则CPU；显式CPU用 `--cpu`。`--workers 4`增加加载进程，默认0兼容Windows；`--threads 4`控制CPU线程。实际训练预算在运行开始时打印，不自动缩短100轮。

修改数据、分辨率、训练轮数或主宽度，需要新项目：

```text
python experiment.py init --project experiments/second --data-dir ../../dataset --epochs 100 --image-size 64 --width 16
python experiment.py run --project experiments/second
```

后续所有命令都带相同 `--project`。不要手动编辑project.json、manifest.json或plan.json；配置/数据/源代码哈希不一致时平台拒绝混用。代码有改动时创建新项目，原始代码可在旧项目的 `source_snapshot/` 追溯。不同代码版本的实验不自动合并。

## 4. 查看报告、自动分析与多种子复核

```text
python experiment.py report
python experiment.py analyze --models Q,G,GQQ --seeds 42
python experiment.py report
```

`analyze`对每个完成模型的每层调用已有探针。普通聚合只观察实际存在的阶段；标准化图层额外运行A–E。每次明确使用该运行的last.pth，不自动取全工作区最新权重。分析完成项跳过，未完成目录改名保留后重试。

也可训练后立即分析：`python experiment.py run --models Q,G,GQQ --analyze`。完整分析比训练多出探针拟合开销，默认run不自动做全部探针。快速试通可用 `analyze --probe-epochs 1 --probe-dim 8`，结果不能与正式30轮、256维混用。

根据首轮结果预选对照，补齐训练种子：

```text
python experiment.py plan --batch confirm --models Q,G,GQQ --seeds 43,44
python experiment.py run
python experiment.py report
```

报告分别给各种子、均值/样本标准差和同种子差值的95% t区间。单种子只标线索；未达到3种子及协议效应门槛不标“支持改善”。dev比较均为探索性，不能把筛选最优项当独立最终验证。

## 5. 扩展实验与自动匹配预算

先选择一个实验家族，不默认全排列：

```text
python experiment.py plan --batch topology
python experiment.py plan --batch depth
python experiment.py plan --batch capacity
python experiment.py plan --batch residual
python experiment.py plan --batch schedule
```

每条命令只添加清单、不立即训练；重复模型/种子不会重复添加。`topology`含完整外积、3份固定关联错配、z标准化及z-MLP；`depth`含纯G/Q的1–4层；`capacity`将当前恒定宽度加倍；`residual`给G/Q同样的均值内容路径；`schedule`比较DDDPPP和DPDPDP。默认64输入/head-grid4支持全部深度，缩小输入的自检项目不一定支持，平台会拒绝非法形状。

自动按参数匹配：

```text
python experiment.py match-budget
python experiment.py cost
```

平台仅构造G(C=8/16/32)与Q的预设候选，按参数最近且误差≤5%挑选Q，不读取准确率；只将满足匹配条件的Q加入训练清单。结果写入 `budget_selection_parameters.json`，不满足容差的目标标记未匹配。需要运算对照时用 `--metric conv_linear_macs`，容差10%。

报告的预算匹配表和 `budget_matches.csv` 会列出对应关系；匹配成功且双方已有同种子结果时，`paired.csv` 自动加入这两个模型的直接比较。

**当前运算统计只覆盖实际执行的Conv/Linear，尚不包含完整图构造、标准差等逐元素开销。** 因此该模式称为Conv/Linear MAC匹配，不能称为完整FLOPs公平对照；成本报告始终标明不完整。每MAC按2 FLOPs换算并单独列名。`cost`不训练，支持 `--models`筛选。

`plan --batch budget`会添加完整候选池，通常优先使用 `match-budget`，避免把所有预算候选都训练一遍。

## 6. 设备实测

```text
python experiment.py benchmark --models Q,G,GQQ --seeds 42
python experiment.py report
```

使用完成的last checkpoint，在设备内预生成输入；batch1测延迟、batch16测吞吐。默认预热50次、每批200次、重复5批；GPU显式同步，CPU使用高精度时钟。包含模型内所有图构造与分支，排除加载和图片预处理。

记录dtype、线程、软件/硬件、每批中位数/P90、吞吐，以及GPU allocated/reserved峰值。CPU内存和能耗暂未测量，输出留空。每次测量独立存档，报告不会将不同设备混成同一个平均数。快速验收可缩短 `--iterations/--warmup/--repeats`，需要保留测量次数说明。

## 7. 冻结后再评估最终集

完成开发集选择后，明确指定模型与种子：

```text
python experiment.py freeze-final --models Q,G,GQQ --seeds 42,43,44
python experiment.py final
python experiment.py report
```

要求每个模型都有全部指定种子的完成checkpoint。注册保存权重哈希；此后该项目拒绝新增实验和继续训练，final评估仅使用冻结权重。重复final复用已保存结果，不重新计算；更换权重或修改注册会报错。

最终结果独立输出 `final.csv/final_paired.csv`，保留初始化时的final状态。若原val历史已使用，仍只能作为历史已查看集上的冻结复核。需要进一步修改模型时另建项目，并如实延续数据曾被查看的记录。

## 8. 全流程自检和验收

```text
python experiment.py smoke --project experiments/smoke_check
python -m unittest discover -s tests -v
```

smoke会生成小型合成图片，完成3个模型的训练、跳过检查、混合模型的普通及A–E探针、最终注册和评估、报告与图表，输出保留在指定新目录。它验证流程，不能充当真实数据成绩。目录已存在时换一个名称，不自动删除旧结果。

测试覆盖：局部窗口、矩阵公式、线性卷积等价、无闲置参数、梯度、旧模型工厂、关联错配、数据/来源组交叉、清单篡改、预算选型、单种子结论边界、CPU计时、最终注册、已完成跳过，以及中断恢复与连续训练的参数一致性。

## 9. 输出位置与当前边界

```text
experiments/main/
  project.json                 # 冻结训练设置与源代码指纹
  manifest.json                # 相对路径、标签、划分及内容哈希
  plan.json                    # 可通过plan命令追加的任务表
  source_snapshot/             # 初始化时的源码
  runs/<model>_s<seed>/
    config.json, state.json, metrics.json
    last.pth, best.pth, evaluation_dev.json
    environment.json, sessions/, cost.json
    analysis/, benchmarks/, evaluation_final.json
  report/index.html            # 双击打开
  report/*.csv, report.md, curves.png, curves.svg
  final_registration.json      # 显式冻结后创建
```

实验产物由gitignore排除，仓库提交源代码和文档；重要结果需另行备份。平台保留训练轮次和随机状态，但跨设备/库版本不承诺逐位复现。现有GPU路径需要在目标设备运行benchmark验证，CPU验收不能代替硬件收益证明。CIFAR-100专用加载、完整FLOPs计数和自动语义任务诊断不在本次实现中。
