> 2026-11-10 存储迁移已完成。当前路径、更新和下载命令见 [存储与同步指南](../docs/storage-workflow.md)；下文旧相对路径通过兼容链接继续使用，服务器信息以新指南为准。

# A1 §7.3–7.5 实验方案

脚本已经准备好，**尚未运行任何实验**。所有命令在本工作区执行：

```bash
cd /home/yifeng/projects/CS336/a1-experiments
```

使用已经安装作业依赖的 Python，例如激活 `CS336` 环境后执行 `python`。
脚本根据自己的位置查找相邻的 `assignment1-basics`，无须手动设置 PYTHONPATH。
实验模型变体放在 `scripts/model_variants.py`，不会改变作业原模型及测试适配器。

## 1. 先选规模，只预览

```bash
python scripts/run_suite.py --profile demo --section ablations
```

**没有 `--execute` 时只列计划，不加载模型、不训练，也不创建结果目录。**

| profile | 模型 d_model / 层 / 头 / d_ff | 上下文 / 批次 / 步数 | 每次训练词元数 | 用途 |
| --- | --- | --- | --- | --- |
| demo | 64 / 2 / 4 / 192 | 64 / 8 / 500 | 256,000 | 现有小样本，仅验证流程 |
| cpu | 512 / 4 / 16 / 1344 | 256 / 32 / 5000 | 40,960,000 | 讲义 CPU 低资源规模 |
| gpu | 512 / 4 / 16 / 1344 | 256 / 32 / 40000 | 327,680,000 | 讲义标准规模，需要合适 GPU |

表中是每个种子、每个配置的预算，多组运行会累加。CPU 档可能耗时很长，脚本不承诺讲义示例机器的速度。
默认以 float32 训练，不会安装软件、切换网络、下载数据、租用 GPU 或提交排行榜。
需要自定义规模时，复制 `profiles.json` 为新文件，修改其中一档，再传 `--profiles 新文件路径`。
可用 `--device cuda:0` 和 `--num-threads 8` 覆盖设备、线程数。

## 2. 准备真实数据

demo 消融使用已有的 `data/tinystories_small/`。cpu/gpu 需要提前准备正式数据：

```bash
# 先把课程训练/验证文本下载到自己选择的位置；下面路径是占位路径。
python scripts/prepare_corpus.py \
    --train-text /你的路径/TinyStoriesV2-GPT4-train.txt \
    --val-text /你的路径/TinyStoriesV2-GPT4-valid.txt \
    --vocab-size 10000 --output data/tinystories_full

# 确认路径和方案后，原命令加 --execute 才会训练 BPE 并编码。
```

OWT 对应：

```bash
python scripts/prepare_corpus.py \
    --train-text /你的路径/owt_train.txt \
    --val-text /你的路径/owt_valid.txt \
    --vocab-size 32000 --output data/owt_full
```

输出目录必须不存在，避免覆盖分词器与编号。只有训练文本参与 BPE 学习，两个集合共用同一分词器。
编码分块写盘，词表容得下时用 uint16，最终生成可内存映射的 `.npy`。
但现有 `train_bpe_indexed` 的 BPE 训练仍需读入文本及统计结构；完整 OWT 可能需要大量内存，
分块编码不意味着 BPE 训练本身已低内存化。使用更小数据时应明确标注规模，并修改 profile 的目录与词表大小。

执行训练前，方案脚本会检查所有输入路径、格式、词表大小及维度配置；缺文件会先报错，不会偷偷换成演示数据。

## 3. §7.3：四组消融

把 `--lr` 设置为 §7.2 基线实验选出的学习率。脚本中的 0.001 只是初始配置，不代表已验证最优。

```bash
# 先预览；实际执行时再加 --execute。
python scripts/run_suite.py --profile demo --section ablations \
    --lr 0.001 --seeds 0 --output results/demo_ablations_01
```

生成 7 次训练：

| 实验 | 相对基线的唯一结构变化 | 学习率 |
| --- | --- | --- |
| baseline | pre-norm + RMSNorm + RoPE + SwiGLU | lr |
| no_norm × 3 | 移除两个子层及输出端的所有 RMSNorm | lr、0.3lr、0.1lr |
| post_norm | 两个子层改为 Norm(x + F(x))，其余含输出端 Norm 保留 | lr |
| nope | 去掉 Q/K 上的 RoPE，保留因果掩码 | lr |
| silu | FFN 改为 W2 SiLU(W1 x)，隐藏维度为 4*d_model | lr |

同一 profile 的结构、训练步数、批次和种子保持一致。训练采样使用独立 NumPy 随机数生成器，
每次验证固定同一组起点，不受模型初始化、评估频率或生成采样影响。
变化后的组件仍需初始化，因此不能要求不同结构的每一项参数都相同。
记录实际总参数量及每块 FFN 参数量：cpu/gpu 的 SiLU 与 SwiGLU 近似匹配；demo 的偏差较大，应在报告中说明。

需要较可靠的结论时使用 `--seeds 0 1 2`；这会变成 21 次训练。
无归一化的较低学习率都不稳定时，修改方案继续降低，而不是把三个初始候选当成已找到最优值。
运行前每个结果目录必须不存在。脚本不会自动删旧实验或静默复用不同配置的结果。

## 4. §7.4：OpenWebText

```bash
python scripts/run_suite.py --profile cpu --section owt \
    --lr 0.001 --owt-lr 0.001 --seeds 0 \
    --output results/cpu_owt_01
```

这会计划 TinyStories 和 OWT 两个基线，模型隐藏结构、上下文、批次、步数相同，数据及相应词表不同。
§7.4 不使用 45 分钟截断，可按需要单独调整 OWT 学习率。
两种词表下的 nats/token 不能直接解释为同一尺度的文本质量；报告中需讨论词元粒度、数据复杂性和生成表现。

demo 档的 OWT 路径为 `data/owt_small/`，需自行准备，默认不会下载。
GPU 有限的自学者可按讲义低资源提示优先完成 TinyStories 上的架构对比，明确说明省略或缩减的部分。

## 5. §7.5：自己的修改及限时候选

```bash
python scripts/run_suite.py --profile gpu --section leaderboard \
    --lr 0.001 --seeds 0 --output results/gpu_tied_01
```

计划两个 OWT 运行：baseline 和 tied。tied 共享输入嵌入、输出头，并将共享矩阵用标准差 0.02 重新初始化。
它是“共享权重 + 初始化调整”的组合实验，不能仅凭结果断言收益都来自共享权重。
两者独立使用 2700 秒预算；执行器预留 60 秒保存，方案脚本用子进程硬超时兜底。
预算不足时可能提前停止或尚未来得及生成样例，状态会如实标为 time_budget/hard_timeout。
硬超时后可使用已保存的 best.pt；不会声称丢失的最终验证已完成。

**只有符合课程 B200、指定 OWT 数据及运行规则的结果才可用于正式排行榜。**
其他 GPU/CPU 上的限时结果只是本地候选实验。本脚本不自动提交排行榜，也不保证能达到目标损失。
进一步的自选架构可在 `model_variants.py` 添加并扩展方案列表。

`--section all` 合并三部分：单种子为 10 次运行；§7.4 固定步数 OWT 和 §7.5 限时 OWT 分开记录。

## 6. 结果文件和曲线

显式加 `--execute` 后，脚本顺序运行，不并发抢占设备；每组会保存：

```text
results/某次方案/
├── manifest.json                 # 方案、子进程返回码
├── 各组.json                     # 请求配置
├── 各组.console.log              # 完整控制台输出、报错
├── 各组/
│   ├── requested.json
│   ├── config.json               # 实际配置、参数量、分词器摘要、版本
│   ├── metrics.csv               # 步数、时间、训练及验证损失
│   ├── outcome.json              # 状态、最佳/最终验证损失、实际步数及时间
│   ├── best.pt                   # 验证集最佳检查点
│   ├── final.pt                  # 最后完成更新的检查点（正常/预算停止时）
│   ├── samples.txt               # 最佳检查点的 256 个新词元或到 EOS
│   └── samples.json              # 提示、温度、top-p、采样种子、实际长度
├── summary.csv
└── summary.md
```

失败或过早中断的运行可能只有其中一部分文件，summary 不会将其伪装成成功。
NaN/Inf 损失或梯度会标为 diverged，CUDA OOM 等标为 failed；失败后继续下一组。

汇总会自动生成。需要图时运行：

```bash
python scripts/summarize.py results/demo_ablations_01 --plots
```

绘图需要当前环境有 matplotlib；脚本不自动联网安装。
生成按数据集分开的验证损失曲线，分别以步数和含初始化的实际耗时为横轴，同时保存 PNG/SVG。
多种子最终验证损失给出均值与样本标准差，单种子不虚构方差。
同一数据、学习率和预算才合并统计；完整实验采用配置中的相同数据文件与分词器。

尚未跑实验时没有真实曲线或结果表。报告请使用 `docs/ablation_report_template.md`，根据实际输出填写。
