> 2026-11-10 存储迁移已完成。当前路径、更新和下载命令见 [存储与同步指南](../../docs/storage-workflow.md)；下文旧相对路径通过兼容链接继续使用，服务器信息以新指南为准。

# 训练实验日志

在 `assignment1-basics` 目录运行，例如：

```bash
python -m cs336_basics.part4_training_loop.train \
    --num-steps 100 \
    --seed 0 \
    --log-dir ../a1-experiments/logs/demo_seed0 \
    --checkpoint-dir ../a1-experiments/checkpoints/demo_seed0
```

每次实验使用新目录；不指定 `--log-dir` 时，在 `../a1-experiments/logs/run_时间戳/` 下自动创建。
显式指定已经存在配置文件的目录会报错，避免覆盖旧记录。
不同实验也应指定不同的检查点目录。

## 输出文件

- `config.json`：命令行配置、实际数据路径、起始步数、优化器参数和软件版本。
- `metrics.csv`：每完成一次更新写入一行，包含下表字段。

| 字段 | 含义 |
| --- | --- |
| `step` | 全局已经完成的参数更新次数，从 1 开始 |
| `elapsed_seconds` | 本次训练循环启动后的实际耗时，包含验证和保存检查点，不包含启动和加载时间 |
| `tokens_processed` | 本次启动处理的训练输入词元数，不包含验证，也不重复计算目标词元 |
| `learning_rate` | 本次更新使用的学习率 |
| `train_loss` | 本次更新前、当前训练批次的平均交叉熵 |
| `val_loss` | 更新后评估的平均验证损失；未评估时留空 |

验证按 `--eval-interval` 执行，最后一步也执行。配置中的种子用于启动时初始化随机数，
当前检查点不保存随机数状态，因此恢复后不保证与不中断训练逐步完全一致。

## 恢复训练

恢复时保持模型、数据、优化器和学习率调度配置一致，并为本次启动指定新的日志目录：

```bash
python -m cs336_basics.part4_training_loop.train \
    --num-steps 100 \
    --seed 0 \
    --resume ../a1-experiments/checkpoints/demo_seed0/step_50.pt \
    --log-dir ../a1-experiments/logs/demo_seed0_resume50 \
    --checkpoint-dir ../a1-experiments/checkpoints/demo_seed0_resume50
```

上述例子从第 51 次更新继续。CSV 的 `step` 延续全局步数，但时间和词元计数从本次启动重新计起；
不会把中断后不计时的加载过程算进训练耗时，也不会混写旧实验的曲线。
绘制整段恢复训练曲线时，应按配置中的恢复关系连接各段，再累计各段实际耗时。

生成样例的保存尚未接入，这一步先记录训练配置和损失曲线所需的原始数据。
