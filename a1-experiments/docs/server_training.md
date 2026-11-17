> 2026-11-10 存储迁移已完成。当前路径、更新和下载命令见 [存储与同步指南](../../docs/storage-workflow.md)；下文旧相对路径通过兼容链接继续使用，服务器信息以新指南为准。

# A1 本地与导师服务器训练操作文档

本文只提供操作步骤，尚未连接服务器、上传文件或启动训练。标有“启动训练”的命令留到你准备好后执行。服务器地址、账号、调度分区均需替换为导师提供的信息。

## 1. 是否需要服务器

本机检查结果：AMD Ryzen 7 H 255，8 核 16 线程；WSL 可见内存约 9.7 GiB，检查时可用约 6.7 GiB。未找到 `nvidia-smi`；这不足以证明机器完全没有独立显卡，但目前没有确认可用的 NVIDIA CUDA 环境。AMD 核显不能直接运行本脚本的 CUDA 配置。

| 任务 | 建议 |
| --- | --- |
| 已有 64 维、2 层、500 步 demo | 本地 CPU 可以跑，之前已跑通 |
| 修改代码、检查数据、验证训练流程 | 留在本地 |
| 512 维、4 层、5000 步低资源配置 | CPU 可以尝试，但耗时和内存需实测 |
| 7 组消融、多个种子、完整 OWT 实验 | 推荐导师的 NVIDIA GPU 服务器 |
| 课程正式排行榜 | 需另行满足课程指定硬件、数据及限时规则 |

服务器是否值得迁移，关键是它是否提供可用 GPU；只有 CPU 的服务器不一定比本机快。没有实测之前，不承诺训练需要几小时。

当前程序使用 float32，未实现混合精度（mixed precision）、梯度累积（gradient accumulation）或多卡并行。多张 GPU 不会自动一起训练。OWT 配置中，仅 `(32, 256, 32000)` 的一份 float32 logits 就约占 0.98 GiB，反向传播和中间张量还需额外内存；不能仅按模型权重大小判断能否训练。

## 2. 先向导师确认这些信息

- SSH 地址、用户名、端口，是否需要校园网/VPN或跳板机。
- GPU 型号、显存及你获准使用的设备；主机内存、存储目录和配额。
- 使用 Slurm 等调度系统，还是直接登录指定 GPU 机器；作业最长时限。
- 是否已有 Python 3.12、PyTorch GPU 环境或课程容器。
- 是否已有课程 TinyStories/OWT 数据及共享位置。

有调度系统时，在分配到的计算节点检查 GPU、运行训练；登录节点仅做上传、编辑和提交作业。

## 3. 从本地上传代码和小数据

在**本地 WSL**设置 SSH 别名，编辑 `~/.ssh/config`，追加以下内容，保留已有配置：

```sshconfig
Host mentor-a1
    HostName 替换成服务器地址
    User 替换成你的账号
    Port 22
```

如实际端口不是 22，修改 Port。跳板机按实验室提供的配置添加。随后执行：

```bash
# 本地 WSL
ssh mentor-a1
# 登录后退出，回到本地进行上传
exit

ssh mentor-a1 'mkdir -p ~/projects/CS336'
cd /home/yifeng/projects/CS336
rsync -av --progress \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  assignment1-basics/ mentor-a1:~/projects/CS336/assignment1-basics/

rsync -av --progress \
  --exclude='results/' --exclude='checkpoints/' --exclude='logs/' \
  --exclude='__pycache__/' \
  a1-experiments/ mentor-a1:~/projects/CS336/a1-experiments/
```

保留两个目录相邻，脚本通过这个关系定位作业源码。上传的是当前磁盘文件，包括尚未提交 Git 的实现；仅在服务器克隆课程原仓库会缺少你自己的实现。命令不使用 `--delete`，不会删除服务器额外文件；重新上传同名源码会更新它们，训练过程中不要同步修改中的源码。

无需上传本地 Conda 环境。大型数据准备好后可以单独传输，或使用服务器已有数据。当前小数据只适合 demo。

## 4. 配置服务器环境

以下在**服务器**运行。优先使用实验室已经提供且兼容本项目的环境。项目 `pyproject.toml` 要求 Python >=3.12、<3.14，并指定 PyTorch 2.11 系列；旧的 CS231n Python 3.8 环境不能直接使用。

如果服务器已安装 `uv`，使用项目锁文件：

```bash
cd ~/projects/CS336/assignment1-basics
uv sync --locked --python 3.12
source .venv/bin/activate
python --version
python -m pip --version
```

`uv` 创建的环境可能没有 pip；最后一条失败不代表环境损坏，之后可用 `uv pip`。如果没有 uv，先按实验室环境管理方式安装，或参考 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)。`--locked` 会要求依赖声明与锁文件一致，避免静默改变版本，参见 [uv 同步文档](https://docs.astral.sh/uv/concepts/projects/sync/)。

检查 GPU（有调度系统时先申请计算资源）：

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('wheel CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('memory GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
```

这只是环境检查，不训练模型。必须确认 `CUDA available: True`。如失败，检查是否分配到 GPU、安装包是否支持 CUDA，以及驱动与 PyTorch 构建是否兼容。不要随意升级实验室驱动。根据实际驱动采用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 或 [uv 的 PyTorch 配置说明](https://docs.astral.sh/uv/guides/integration/pytorch/)，不能在未知服务器上预先固定一个 CUDA 安装命令。变更依赖后记录具体版本与安装方式。

联网安装前先确认网络：境外官方源通常适合日本 VPN；清华/USTC、hf-mirror 等国内源应走国内网络。服务器网络可能与本机不同。优先用实验室已有环境，避免对境外源反复多连接下载。

## 5. 先做预览，再做短跑

激活环境后，在服务器实验目录执行：

```bash
cd ~/projects/CS336/a1-experiments
# 只预览：不会加载模型或训练
python scripts/run_suite.py --profile demo --section ablations --device cuda:0
```

确认无缺文件后，先用已有小数据做**单个 baseline、100 步短跑**。下面第一段仅生成配置，不训练：

```bash
mkdir -p plans
python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
from run_suite import make_jobs
profile = json.loads(Path('profiles.json').read_text())['demo']
profile.update(device='cuda:0', num_steps=100, warmup_iters=10,
               eval_interval=50, eval_batches=5)
_, config = make_jobs(profile, 'ablations', [0], 0.001)[0]
Path('plans/demo_pilot.json').write_text(json.dumps(config, indent=2))
PY

# 仍只预览配置
python scripts/train_experiment.py \
  --config plans/demo_pilot.json --output results/demo_pilot_01

# 启动训练：准备好后才执行
python -u scripts/train_experiment.py \
  --config plans/demo_pilot.json --output results/demo_pilot_01 --execute
```

目录已存在时换新名称，脚本不会覆盖。这个短跑仅验证服务器流程，不能测出正式模型的速度。

准备好正式数据后，再生成**正式模型 100 步短跑**：复制上面的生成配置代码，将 `['demo']` 改为 `['cpu']`，并把配置文件名改为 `plans/formal_pilot.json`。保留 `device='cuda:0'`、100 步、warmup 10；这样测试的是正式模型、批次和上下文，训练预算仍很小。执行时使用新的输出目录 `results/formal_pilot_01`。

`cpu` profile 的名字表示较低的训练词元预算；加上 CUDA 设备后也能用 GPU 训练。OWT 显存需求不同，如需跑 OWT，将生成配置时的 `'ablations'` 改为 `'owt'` 并把返回列表索引 `[0]` 改为 `[1]`，另存配置和输出目录。

从 `metrics.csv` 选取不跨验证/保存操作的一段连续训练步，用时间差除以更新步数，估算每步耗时；乘 5000 或 40000，再加验证、保存和初始化时间。多组消融还需累加，不能用小模型速度外推。短跑的 `outcome.json` 会记录峰值 CUDA 分配内存，结合 `nvidia-smi` 观察实际显存占用。短跑最终 loss 不用于正式实验结论。

## 6. 准备正式数据

完整准备步骤见 [EXPERIMENTS.md](../EXPERIMENTS.md)。TinyStories 正式配置要求词表 10000；OWT 要求 32000。每个数据目录必须包含：

```text
train_tokens.npy
val_tokens.npy
tokenizer.json
```

它们应来自同一个分词器；不要只拷贝编号数组再重新训练一个分词器。数据可通过上述 rsync 单独上传，或在服务器用 `scripts/prepare_corpus.py` 准备。该程序仍需内存训练 BPE，完整 OWT 不能保证适合本机约 10 GiB 的 WSL 内存。

若共享数据在其他目录，复制 `profiles.json` 为 `profiles.server.json`，修改对应数据 `directory` 为服务器绝对路径。后续命令都添加 `--profiles profiles.server.json`。预览会显示缺失文件，不会自动下载或换用小数据。

## 7. 正式运行：先单种子，再增加重复实验

建议先完成一个低资源预算的消融套件：**正式模型 + 5000 步 + GPU**。它含 7 次训练；`--lr` 应采用你选定的基线学习率，下面 0.001 只是初始候选。

```bash
# 只预览
python scripts/run_suite.py --profile cpu --device cuda:0 \
  --section ablations --lr 0.001 --seeds 0 \
  --output results/server_ablations_01

# 启动训练
python -u scripts/run_suite.py --profile cpu --device cuda:0 \
  --section ablations --lr 0.001 --seeds 0 \
  --output results/server_ablations_01 --execute
```

`--profile gpu` 是同一模型的 40000 步预算，单次词元数是 cpu 档的 8 倍。根据短跑速度和分配时限决定是否使用。需要多个种子时用 `--seeds 0 1 2`，7 次会变成 21 次。OWT 用 `--section owt`；限时改进用 `--section leaderboard`，具体解释见实验方案。先不要直接运行 `--section all`。

显存不足时可复制 profile 减小 `batch_size`；当前没有梯度累积，批次减半会使每步训练词元减半。如希望保留总词元预算，可相应增加步数，但优化行为仍发生变化，应统一调整所有对比组、重新选学习率并在报告记录。不要只给某一变体换批次后声称严格可比。

## 8. SSH 断开后继续运行

**直接 GPU 机器**：在获准使用的机器和 GPU 上用 tmux：

```bash
tmux new -s a1
cd ~/projects/CS336/assignment1-basics
source .venv/bin/activate
cd ../a1-experiments
# 在此执行第 7 节的启动训练命令
```

按 `Ctrl+b`，松开，再按 `d` 脱离会话；重新登录后执行 `tmux attach -t a1`。它不能抵抗服务器重启，也不能延长调度作业时限。

**Slurm 集群**：将下面内容保存为服务器的 `a1-experiments/submit_a1.sh`。`gpu` 分区、资源和时长只是模板，按导师提供的规则修改；不要在登录节点直接启动长训练。

```bash
#!/bin/bash
#SBATCH --job-name=cs336-a1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=a1-%j.out
set -euo pipefail
cd "$HOME/projects/CS336/assignment1-basics"
source .venv/bin/activate
cd ../a1-experiments
python -u scripts/run_suite.py --profile cpu --device cuda:0 \
  --section ablations --lr 0.001 --seeds 0 \
  --output "results/slurm_${SLURM_JOB_ID}" --execute
```

在服务器实验目录执行 `sbatch submit_a1.sh` 提交，`squeue -u "$USER"` 查看，`scancel 作业编号` 停止。12 小时不是训练耗时预测，须依据短跑估算并符合分区上限。调度器通常会设置可见 GPU，使用 `cuda:0` 指向已分配设备，不要擅自覆盖 `CUDA_VISIBLE_DEVICES`。

## 9. 查看、汇总并取回结果

服务器上查看某组进度：

```bash
tail -f results/server_ablations_01/tinystories_baseline_lr0.001_seed0.console.log
```

套件结束后自动生成 summary。绘图须环境装有 matplotlib；可以在项目环境用 `uv pip install matplotlib` 安装，并记录额外版本（再次精确同步环境时可能移除额外包）。

```bash
# 服务器，已有结果才执行，不会重新训练
python scripts/summarize.py results/server_ablations_01 --plots
```

当前 `manifest.json` 中记录的是服务器绝对路径，**应在服务器完成汇总和绘图，再取回产物**。复制到本地后可直接查看 CSV、Markdown 和图；原样在本地重跑汇总会找不到服务器路径，需先更新清单的 `config` 和 `directory` 路径。

在**本地 WSL**取回结果：

```bash
mkdir -p /home/yifeng/projects/CS336/a1-experiments/results/server_download
rsync -av --progress \
  mentor-a1:~/projects/CS336/a1-experiments/results/server_ablations_01/ \
  /home/yifeng/projects/CS336/a1-experiments/results/server_download/server_ablations_01/
```

检查 `summary.md`、`summary.csv`、各组 `outcome.json`、`metrics.csv`、`samples.txt` 及曲线。磁盘紧张可先在 rsync 添加 `--exclude='*.pt'` 只取日志和图；生成文本或恢复模型仍需对应权重。

当前批量实验执行器**没有自动断点续训**，不要直接重复同一输出目录命令来“续跑”。异常退出后保留已有日志和检查点，查看 `.console.log`；修复后使用新输出目录重跑。单独的作业训练接口有 checkpoint 支持，但不能据此认为批量套件已实现精确恢复采样和进度。

推荐顺序：本地调试 → 服务器环境检查 → demo 短跑 → 正式数据和正式模型短跑 → 单种子消融 → 资源允许时多种子与 OWT。
