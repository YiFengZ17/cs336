> 2026-11-10 存储迁移已完成。当前路径、更新和下载命令见 [存储与同步指南](../docs/storage-workflow.md)；下文旧相对路径通过兼容链接继续使用，服务器信息以新指南为准。

# A1 实验工作区

本目录保存运行产物和实验说明，与 `../assignment1-basics/` 作业源码分开。

```text
a1-experiments/
├── data/          # 演示数据、TinyStories 文本、分词器和编号数组
├── checkpoints/   # 训练检查点
├── logs/          # 每次实验的 config.json、metrics.csv
└── docs/          # 实验操作说明
```

作业必需的训练、生成、数据加载和实验日志实现仍保留在
`assignment1-basics/cs336_basics/part4_training_loop/`，以便测试和提交。
`experiment_log.py` 随作业源码一起提交；生成的配置和 CSV 日志保存在本目录的 `logs/` 下。
准备数据、训练、生成脚本的默认路径都已经指向本目录。
原有数据和检查点已迁入，文件内容保持不变。

## 运行

在 `assignment1-basics` 目录中执行，原模块命令保持不变：

```bash
python -m cs336_basics.part4_training_loop.prepare_data
python -m cs336_basics.part4_training_loop.train
python -m cs336_basics.part4_training_loop.generate
```

默认训练仍使用演示数据；真实文本实验应指定对应的数据和模型参数，例如：

```bash
python -m cs336_basics.part4_training_loop.train \
    --train-data ../a1-experiments/data/tinystories_small/train_tokens.npy \
    --val-data ../a1-experiments/data/tinystories_small/val_tokens.npy \
    --vocab-size 1000 --context-length 64 --d-model 64 \
    --num-layers 2 --num-heads 4 --d-ff 192 \
    --num-steps 500 --warmup-iters 20 \
    --max-lr 0.001 --min-lr 0.0001 --batch-size 8 \
    --eval-interval 50 --save-interval 250 \
    --log-dir ../a1-experiments/logs/tinystories_run01 \
    --checkpoint-dir ../a1-experiments/checkpoints/tinystories_run01
```

不同实验使用不同目录。手动传入的相对路径仍相对于当前工作目录；
省略路径参数时，默认路径根据源码位置确定，不依赖当前工作目录。
日志字段说明见 [docs/training_logs.md](docs/training_logs.md)。

## §7.3–7.5 批量实验方案

消融、OWT 对比、自选改进及限时实验的脚本见 [EXPERIMENTS.md](EXPERIMENTS.md)。
这些脚本默认仅预览，必须显式添加 `--execute` 才会训练。
批量实验结果放在本目录的 `results/` 中；不会改写已有检查点或作业默认模型。

本机资源判断、上传代码、配置 GPU 环境、短跑测速、tmux/Slurm 运行及取回结果见
[导师服务器训练操作文档](docs/server_training.md)。
