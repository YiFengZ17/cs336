"""汇总已有结果；--plots 可额外生成损失曲线，不会启动训练。"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plots", action="store_true", help="需要 matplotlib；不自动安装")
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest is None:
        parser.error("目录中没有 manifest.json，请指定一次 run_suite 的结果目录")
    rows, curves = [], []
    for item in manifest:
        run_dir = directory / item["directory"]
        plan = directory / item["config"]
        config = read_json(run_dir / "config.json", read_json(plan, {}))
        outcome = read_json(run_dir / "outcome.json", {})
        exit_code = item.get("exit_code")
        status = outcome.get("status", "not_started" if exit_code is None else "incomplete")
        if exit_code == 124:
            status = "hard_timeout"
        rows.append({
            "name": item["name"], "dataset": config.get("dataset"),
            "variant": config.get("variant"), "seed": config.get("seed"),
            "max_lr": config.get("max_lr"), "max_seconds": config.get("max_seconds"),
            "status": status, "exit_code": exit_code,
            "completed_steps": outcome.get("completed_steps", 0),
            "tokens_processed": outcome.get("tokens_processed", 0),
            "num_parameters": config.get("num_parameters"),
            "ffn_parameters_per_block": config.get("ffn_parameters_per_block"),
            "best_val_loss": outcome.get("best_val_loss"),
            "final_val_loss": outcome.get("final_val_loss"),
            "last_recorded_val_loss": None,
            "training_wall_seconds": outcome.get("training_wall_seconds"),
            "peak_cuda_bytes": outcome.get("peak_cuda_bytes"),
            "reason": outcome.get("reason", ""),
        })
        metrics = run_dir / "metrics.csv"
        if metrics.exists():
            with metrics.open(encoding="utf-8", newline="") as file:
                recorded = list(csv.DictReader(file))
            points = [r for r in recorded if finite(r["val_loss"])]
            # 硬超时可能来不及写 outcome.json；CSV 仍能说明实际完成的更新数。
            # 最后一次验证不一定对应最后一步，因此不冒充 final_val_loss。
            if recorded:
                rows[-1]["completed_steps"] = max(rows[-1]["completed_steps"], int(recorded[-1]["step"]))
                rows[-1]["tokens_processed"] = max(rows[-1]["tokens_processed"], int(recorded[-1]["tokens_processed"]))
            if points:
                rows[-1]["last_recorded_val_loss"] = float(points[-1]["val_loss"])
                if rows[-1]["best_val_loss"] is None:
                    rows[-1]["best_val_loss"] = min(float(p["val_loss"]) for p in points)
            curves.append((config, item["name"], points))

    if not rows:
        parser.error("清单中没有实验")
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 只聚合同一数据集、变体、学习率及预算下成功完成或达到预算的运行。
    groups = {}
    for row in rows:
        if row["status"] in ("completed", "time_budget") and finite(row["final_val_loss"]):
            key = (row["dataset"], row["variant"], row["max_lr"], row["max_seconds"])
            groups.setdefault(key, []).append(float(row["final_val_loss"]))
    report = ["# 实验结果汇总", "", "原始逐次运行结果见 summary.csv。以下为最终验证损失（不是训练损失）。", "",
              "| 数据集 | 变体 | 最大学习率 | 时间预算 | 有效种子数 | 均值 | 样本标准差 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for (dataset, variant, lr, budget), values in sorted(groups.items(), key=lambda x: str(x[0])):
        std = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "单种子，未估计"
        report.append(f"| {dataset} | {variant} | {lr:g} | {budget or '步数预算'} | {len(values)} | {statistics.mean(values):.6f} | {std} |")
    report += ["", "失败、发散和硬超时未计入均值，请结合 summary.csv 与 console.log 分析。",
               "不同词表/分词器的数据集，其每词元交叉熵不能直接视为难度或质量的公平对比。",
               "每组 samples.txt 供人工评估流畅度；脚本不会自动编造实验结论。",
               "无归一化实验请同时对照原学习率和较低学习率；选择最优学习率后应再用其他种子确认。",
               "时间预算实验和固定步数实验分开统计。demo 仅用于流程验证。"]
    (directory / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            parser.error("CSV 和 Markdown 已写出；绘图需要当前环境安装 matplotlib")
        for dataset in sorted({c["dataset"] for c, _, _ in curves}):
            for axis in ("step", "wall_seconds"):
                fig, ax = plt.subplots(figsize=(10, 6))
                for config, name, points in curves:
                    if config["dataset"] != dataset or not points:
                        continue
                    xs = [int(p["step"]) if axis == "step" else
                          float(p["elapsed_seconds"]) + config.get("setup_seconds", 0) for p in points]
                    ax.plot(xs, [float(p["val_loss"]) for p in points], marker=".", label=name)
                ax.set(xlabel="Completed optimizer steps" if axis == "step" else "Wall time including setup (s)",
                       ylabel="Validation cross-entropy (nats/token)", title=f"{dataset}: validation curves")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7)
                fig.tight_layout()
                for extension in ("png", "svg"):
                    fig.savefig(directory / f"{dataset}_{axis}.{extension}", dpi=160)
                plt.close(fig)
    print(f"汇总已写入：{directory / 'summary.csv'} 和 summary.md")


if __name__ == "__main__":
    main()
