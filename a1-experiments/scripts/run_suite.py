"""默认只展示方案；只有 --execute 才会顺序启动训练。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE.parent / "assignment1-basics"))
from cs336_basics.part4_training_loop.storage_paths import RUN_ROOT, dataset_directory


def make_jobs(profile, section, seeds, lr, owt_lr=None):
    owt_lr = lr if owt_lr is None else owt_lr
    common = {k: v for k, v in profile.items() if k not in ("tinystories", "owt")}
    common.update({
        "max_lr": lr, "min_lr": lr / 10, "rope_theta": 10000,
        "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1,
        "max_grad_norm": 1.0, "eval_seed": 12345,
        "temperature": 0.8, "top_p": 0.9, "tied_init_std": 0.02,
    })
    cases = []
    if section in ("ablations", "all"):
        cases += [("tinystories", "baseline", lr, None)]
        # 无归一化必须同时观察基线学习率和降低学习率后的稳定性。
        cases += [("tinystories", "no_norm", lr * factor, None) for factor in (1, 0.3, 0.1)]
        cases += [("tinystories", name, lr, None) for name in ("post_norm", "nope", "silu")]
    if section in ("owt", "all"):
        cases += [("tinystories", "baseline", lr, None), ("owt", "baseline", owt_lr, None)]
    if section in ("leaderboard", "all"):
        cases += [("owt", name, owt_lr, 2700) for name in ("baseline", "tied")]

    jobs = []
    for dataset, variant, learning_rate, budget in dict.fromkeys(cases):
        data = profile[dataset]
        for seed in seeds:
            config = dict(common)
            config.update({
                "dataset": dataset, "variant": variant, "seed": seed,
                "max_lr": learning_rate, "min_lr": learning_rate / 10,
                "vocab_size": data["vocab_size"],
                "train_data": str(dataset_directory(data["directory"], WORKSPACE) / "train_tokens.npy"),
                "val_data": str(dataset_directory(data["directory"], WORKSPACE) / "val_tokens.npy"),
                "tokenizer": str(dataset_directory(data["directory"], WORKSPACE) / "tokenizer.json"),
                "prompt": "Once upon a time" if dataset == "tinystories" else "The purpose of science is",
                # §7.4 按相同步数比较；§7.5 是单独的 45 分钟预算实验。
                "max_seconds": budget,
            })
            name = f"{dataset}_{variant}_lr{learning_rate:g}_seed{seed}"
            if budget:
                name += f"_budget{budget}s"
            jobs.append((name, config))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["demo", "cpu", "gpu"], default="demo")
    parser.add_argument("--profiles", type=Path, default=WORKSPACE / "profiles.json")
    parser.add_argument("--section", choices=["ablations", "owt", "leaderboard", "all"], default="ablations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--lr", type=float, default=1e-3, help="先由 §7.2 确定的基线学习率；不是自动搜索的最优值")
    parser.add_argument("--owt-lr", type=float, default=None, help="可单独调整 OWT 学习率；默认与 --lr 一致")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="明确启动实验；默认仅预览")
    args = parser.parse_args()
    if args.lr <= 0 or (args.owt_lr is not None and args.owt_lr <= 0):
        parser.error("学习率必须为正数")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 1001 for s in args.seeds):
        parser.error("种子不可重复，且必须在 [0, 2**32-1001) 内")
    profile = json.loads(args.profiles.read_text(encoding="utf-8"))[args.profile]
    if args.device:
        profile["device"] = args.device
    if args.num_threads:
        profile["num_threads"] = args.num_threads
    jobs = make_jobs(profile, args.section, args.seeds, args.lr, args.owt_lr)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or RUN_ROOT / "a1/suites" / f"{args.profile}_{args.section}_{stamp}").resolve()
    print(profile["description"])
    print(f"计划 {len(jobs)} 次训练，输出：{output}")
    missing = set()
    for name, config in jobs:
        tokens = config["batch_size"] * config["context_length"] * config["num_steps"]
        print(f"  {name}: {config['num_steps']} steps, {tokens:,} tokens, device={config['device']}")
        for key in ("train_data", "val_data", "tokenizer"):
            if not Path(config[key]).is_file():
                missing.add(config[key])
    if missing:
        print("缺少输入文件：\n  " + "\n  ".join(sorted(missing)))
    if not args.execute:
        print("仅预览：未创建结果目录，也未加载模型或启动训练。加 --execute 才执行。")
        return
    if missing:
        parser.error("请先准备全部输入文件；本脚本不下载数据")

    # 先检查所有配置与数据，再开始第一组训练，避免中途才发现路径错误。
    from train_experiment import validate_config
    for _, config in jobs:
        validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for name, config in jobs:
        plan = output / f"{name}.json"
        plan.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        run_dir = output / name
        manifest.append({"name": name, "config": plan.name, "directory": run_dir.name})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for item, (_, config) in zip(manifest, jobs):
        print(f"开始：{item['name']}", flush=True)
        command = [sys.executable, str(Path(__file__).with_name("train_experiment.py")),
                   "--config", str(output / item["config"]), "--output", str(output / item["directory"]), "--execute"]
        # 45 分钟方案使用子进程超时兜底；提前保存的检查点不会丢失。
        with (output / f"{item['name']}.console.log").open("x", encoding="utf-8") as console:
            try:
                result = subprocess.run(command, stdout=console, stderr=subprocess.STDOUT,
                                        timeout=config["max_seconds"])
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        # CUDA OOM、发散、超时等均保留日志；不伪装成完成的实验。
        item["exit_code"] = code
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"结束：{item['name']}，exit_code={code}", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), str(output)], check=True)
    print(f"结果已保存：{output}；可用 summarize.py --plots 绘图。")
    if any(item["exit_code"] != 0 for item in manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
