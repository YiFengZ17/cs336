# 本地与导师服务器：代码、数据、实验目录

本机 WSL 为主要代码来源，mentor-gpu 负责正式训练。2026-11-10 完成迁移。

| 内容 | 本地 WSL | mentor-gpu |
|---|---|---|
| 代码 | /home/yifeng/projects/CS336 | /home/zyf/projects/CS336 |
| 数据 | /home/yifeng/datasets/CS336 | /home/zyf/datasets/CS336 |
| 结果 | /home/yifeng/experiments/CS336 | /home/zyf/experiments/CS336 |
| 基础模型 | /home/yifeng/models/CS336 | /home/zyf/models/CS336 |

默认根据当前用户 home 定位。可选环境变量：CS336_DATA_ROOT、CS336_RUN_ROOT、CS336_MODEL_ROOT；不设置也可运行。模型根目录目前是存储约定，A5 需在自己的模型加载配置中传入实际路径。

## 平时更新代码（本地 WSL 执行）

```bash
cd /home/yifeng/projects/CS336
bash scripts/sync-code.sh --dry-run
bash scripts/sync-code.sh
```

只上传内容有变化的文件，不删除远端文件；跳过 Git 元数据、虚拟环境和实验大文件。课程自带 data、tests/fixtures、prompts 保留同步。覆盖的远端文件备份到 experiments/CS336/_maintenance/sync-时间戳。
检测到训练入口进程时拒绝同步，但检测无法覆盖自定义启动方式；应始终在实验结束后同步。远端临时改过的代码须先手动取回合并；备份不等于自动合并。
服务器使用用户目录 /home/zyf/.local/bin/rsync，不需要管理员安装。

## 上传一份数据

```bash
bash scripts/push-data.sh tinystories/processed/bpe1000-small-v1 --dry-run
bash scripts/push-data.sh tinystories/processed/bpe1000-small-v1
```

数据版本目录发布后保持不变。改变切分、清洗或分词器时新建版本，更新 profiles.json。大型数据传输与代码传输分开。迁移数据具有 dataset-manifest.json；新 prepare_corpus 输出的 preparation.json 包含输入与输出 SHA-256。

## A1 运行（服务器终端）

激活合适的项目环境后：

```bash
cd /home/zyf/projects/CS336/a1-experiments
python scripts/run_suite.py --profile demo --section ablations
```

这是预览，不训练。默认结果位置为 ~/experiments/CS336/a1/suites/；指定 --output 可覆盖。profiles.json 的 datasets: 前缀相对 CS336_DATA_ROOT；普通相对路径兼容旧配置。
正式 TinyStories bpe10000 和 OWT bpe32000 数据尚未准备，不能直接跑完整 profile。
独立 train 模块默认把 config、metrics 和 checkpoints 放在一次 standalone/run_时间戳 内；显式 --log-dir、--checkpoint-dir 仍可用。
prepare_data 默认拒绝覆盖已有目录，可用 --output 指定新数据版本目录。
generate 支持 --checkpoint、--tokenizer、--config、--device、--prompt；不传 config 时保留原小模型尺寸，传 config 时使用训练配置的尺寸，仅支持 baseline。

## 下载结果（本地 WSL 执行）

```bash
bash scripts/pull-results.sh a1/suites/某次实验 --dry-run
bash scripts/pull-results.sh a1/suites/某次实验
# 同时取回模型权重：
bash scripts/pull-results.sh a1/suites/某次实验 --with-checkpoints
```

下载至 ~/experiments/CS336/server-download/，不会覆盖本地独立实验。新 manifest 使用相对路径，下载后可以重新运行 summarize.py；旧的绝对路径 manifest 仍按旧语义读取。

## 历史数据和兼容性

a1-experiments/data 的小数据链接指向外部版本目录；完整数据的链接在数据准备前可能是悬空链接。checkpoints、logs、results 也指向外部目录。根目录旧 data 和 checkpoints 归档到 a1/legacy/root-data 与 root-checkpoints。新代码不依赖这些兼容链接，链接主要用于旧命令。
两端 _maintenance/ 下有迁移前源码快照、每个迁移文件的 SHA-256 和后续同步备份。尚未删除历史重复数据。

## 可复现记录和限制

批量执行器记录项目与 A1 提交号、Git 状态、补丁、源文件哈希、输入数据 SHA-256、分词器哈希及实际配置。同步不上传 .git；服务器 Git 元数据可能滞后，因此以同步生成的 code-sync-manifest.json 和源文件哈希为准。本地修改提交后，重新同步可更新来源信息。
没有创建或推送 Git 提交。实验脚本、配置、文档现已允许 Git 跟踪；数据、输出继续忽略。
批量执行器尚无自动断点续训，普通 --resume 不恢复完整随机状态；本次未改变训练算法。恢复实验必须记录父运行并使用新输出目录。
A2-A5 仅预建外部实验目录，未改课程实现、测试资源和提交结构。
