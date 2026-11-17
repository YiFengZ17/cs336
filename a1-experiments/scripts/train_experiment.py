"""单次实验执行器；没有 --execute 时只读配置并预览，不加载模型。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

WORKSPACE = Path(__file__).resolve().parents[1]
ASSIGNMENT = WORKSPACE.parent / "assignment1-basics"
sys.path.insert(0, str(ASSIGNMENT))


def write_outcome(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_tokenizer(path):
    from cs336_basics.part1_tokenizer.tokenizer import Tokenizer
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {int(k): bytes(v) for k, v in state["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in state["merges"]]
    return Tokenizer(vocab, merges, state["special_tokens"])


def validate_config(c):
    import numpy as np
    import torch
    for key in ("d_model", "num_layers", "num_heads", "d_ff", "context_length",
                "batch_size", "num_steps", "eval_interval", "eval_batches", "num_threads", "sample_tokens"):
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f"{key} 必须是正整数")
    if c["d_model"] % c["num_heads"] or (c["d_model"] // c["num_heads"]) % 2:
        raise ValueError("d_model 必须整除 num_heads，且每头维度必须为偶数")
    if not 0 <= c["warmup_iters"] < c["num_steps"]:
        raise ValueError("要求 0 <= warmup_iters < num_steps")
    if not 0 <= c["min_lr"] <= c["max_lr"] or c["max_lr"] <= 0:
        raise ValueError("学习率范围无效")
    if c["max_seconds"] is not None and c["max_seconds"] <= 60:
        raise ValueError("时间预算需超过 60 秒，以便预留保存时间")
    device = torch.device(c["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("配置要求 CUDA，但当前环境没有可用 CUDA")
    tokenizer = load_tokenizer(c["tokenizer"])
    if len(tokenizer.vocab) != c["vocab_size"] or set(tokenizer.vocab) != set(range(c["vocab_size"])):
        raise ValueError("模型词表大小必须与分词器完全一致，编号应连续")
    for key in ("train_data", "val_data"):
        data = np.load(c[key], mmap_mode="r", allow_pickle=False)
        if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
            raise ValueError(f"{key} 必须是一维整数 .npy 数组")
        if len(data) <= c["context_length"]:
            raise ValueError(f"{key} 长度必须大于 context_length")
        # 快速抽查头尾；完整编号范围由数据准备脚本保证。
        for sample in (data[:4096], data[-4096:]):
            if sample.min() < 0 or sample.max() >= c["vocab_size"]:
                raise ValueError(f"{key} 包含词表范围外的编号")


def sample_batch(data, c, rng):
    import numpy as np
    import torch
    length = c["context_length"]
    starts = rng.integers(0, len(data) - length, size=c["batch_size"])
    x = np.stack([data[i:i + length] for i in starts])
    y = np.stack([data[i + 1:i + length + 1] for i in starts])
    return (torch.tensor(x, dtype=torch.long, device=c["device"]),
            torch.tensor(y, dtype=torch.long, device=c["device"]))


def evaluate(model, data, c):
    import numpy as np
    import torch
    from cs336_basics.part3_training.cross_entropy import cross_entropy
    # 每次评估使用同一组验证起点；不消耗训练采样器的随机状态。
    rng = np.random.default_rng(c["eval_seed"])
    mode = model.training
    model.eval()
    losses = []
    try:
        with torch.no_grad():
            for _ in range(c["eval_batches"]):
                x, y = sample_batch(data, c, rng)
                losses.append(cross_entropy(model(x), y).item())
    finally:
        model.train(mode)
    return sum(losses) / len(losses)


def save_state(path, model, optimizer, step):
    from cs336_basics.part4_training_loop.checkpoint import save_checkpoint
    # 临时文件写完后替换，尽量避免中断时留下半个检查点。
    temporary = path.with_suffix(".tmp")
    save_checkpoint(model, optimizer, step, temporary)
    temporary.replace(path)


def save_samples(model, c, output):
    import torch
    from cs336_basics.part4_training_loop.generate import generate
    tokenizer = load_tokenizer(c["tokenizer"])
    prompt = torch.tensor([tokenizer.encode(c["prompt"])], dtype=torch.long, device=c["device"])
    eos = tokenizer.byte_to_id.get(b"<|endoftext|>")
    torch.manual_seed(c["seed"] + 1)
    result = generate(model, prompt, c["sample_tokens"], c["context_length"],
                      temperature=c["temperature"], eos_token_id=eos, top_p=c["top_p"])
    new_ids = result[0, prompt.shape[1]:].tolist()
    sample = {
        "checkpoint": "best.pt", "prompt": c["prompt"],
        "temperature": c["temperature"], "top_p": c["top_p"],
        "seed": c["seed"] + 1, "new_token_count": len(new_ids),
        "ended_with_eos": bool(new_ids and new_ids[-1] == eos),
        "text": tokenizer.decode(result[0].tolist()),
    }
    (output / "samples.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "samples.txt").write_text(
        f"checkpoint=best.pt, temperature={c['temperature']}, top_p={c['top_p']}\n"
        f"new_tokens={len(new_ids)}, ended_with_eos={sample['ended_with_eos']}\n\n{sample['text']}\n",
        encoding="utf-8",
    )


def run(c, output):
    started = time.perf_counter()
    import numpy as np
    import torch
    from model_variants import build_model
    from cs336_basics.part3_training.optimizer import AdamW
    from cs336_basics.part3_training.cross_entropy import cross_entropy
    from cs336_basics.part3_training.lr_schedule import get_lr_cosine_schedule
    from cs336_basics.part3_training.gradient_clipping import gradient_clipping
    from cs336_basics.part4_training_loop.experiment_log import ExperimentLog

    output.mkdir(parents=True, exist_ok=False)
    (output / "requested.json").write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    outcome = {"status": "running", "completed_steps": 0, "best_val_loss": None}
    model = optimizer = None
    try:
        validate_config(c)
        torch.set_num_threads(c["num_threads"])
        torch.manual_seed(c["seed"])
        np.random.seed(c["seed"])
        if torch.device(c["device"]).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(c["device"]))
        model = build_model(c, c["device"])
        optimizer = AdamW(model.parameters(), lr=c["max_lr"], betas=tuple(c["betas"]),
                          eps=c["eps"], weight_decay=c["weight_decay"])
        train_data = np.load(c["train_data"], mmap_mode="r", allow_pickle=False)
        val_data = np.load(c["val_data"], mmap_mode="r", allow_pickle=False)
        # 专用训练采样器：模型初始化、评估和生成都不会改变批次序列。
        train_rng = np.random.default_rng(c["seed"] + 1000)
        from provenance import sha256, capture
        metadata = dict(c)
        metadata.update({
            "code_state": capture(WORKSPACE.parent, output),
            "setup_seconds": time.perf_counter() - started,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "ffn_parameters_per_block": sum(p.numel() for p in model.layers[0].swiglu.parameters()),
            "torch_version": str(torch.__version__),
            "device_name": torch.cuda.get_device_name(torch.device(c["device"])) if torch.device(c["device"]).type == "cuda" else "cpu",
            "tokenizer_sha256": hashlib.sha256(Path(c["tokenizer"]).read_bytes()).hexdigest(),
            "input_files": {key: {"bytes": Path(c[key]).stat().st_size,
                                   "mtime_ns": Path(c[key]).stat().st_mtime_ns,
                                   "sha256": sha256(c[key])}
                            for key in ("train_data", "val_data")},
            "metric_notes": "固定验证样本；tokens_processed 仅计训练输入；CSV 耗时从训练循环开始，包含验证及保存；预算从执行器开始计时。",
        })
        model.train()
        best = math.inf
        status = "completed"
        last_val_step = 0
        with ExperimentLog(output, metadata) as logger:
            for step in range(c["num_steps"]):
                # 留 60 秒用于最终评估、检查点及生成；外层还有硬超时。
                if c["max_seconds"] and time.perf_counter() - started >= c["max_seconds"] - 60:
                    status = "time_budget"
                    break
                lr = get_lr_cosine_schedule(step, c["max_lr"], c["min_lr"],
                                            c["warmup_iters"], c["num_steps"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                x, y = sample_batch(train_data, c, train_rng)
                optimizer.zero_grad()
                loss = cross_entropy(model(x), y)
                if not torch.isfinite(loss).item():
                    status = "diverged"
                    outcome["reason"] = f"第 {step + 1} 步损失不是有限值"
                    break
                loss.backward()
                norm = torch.sqrt(sum(p.grad.float().square().sum() for p in model.parameters() if p.grad is not None))
                if not torch.isfinite(norm).item():
                    status = "diverged"
                    outcome["reason"] = f"第 {step + 1} 步梯度范数不是有限值"
                    break
                gradient_clipping(model.parameters(), c["max_grad_norm"])
                optimizer.step()
                completed = step + 1
                outcome["completed_steps"] = completed
                val_loss = None
                if completed % c["eval_interval"] == 0 or completed == c["num_steps"]:
                    val_loss = evaluate(model, val_data, c)
                    last_val_step = completed
                    outcome["final_val_loss"] = val_loss if math.isfinite(val_loss) else None
                    if not math.isfinite(val_loss):
                        status = "diverged"
                        outcome["reason"] = "验证损失不是有限值"
                    elif val_loss < best:
                        best = val_loss
                        outcome["best_val_loss"] = best
                        save_state(output / "best.pt", model, optimizer, completed)
                logger.record(completed, completed * c["batch_size"] * c["context_length"],
                              lr, loss.item(), val_loss)
                print(f"step={completed}, loss={loss.item():.4f}, val={val_loss}", flush=True)
                if status == "diverged":
                    break

            completed = outcome["completed_steps"]
            if completed and status != "diverged":
                # 时间预算触发在非评估步时，也记录真正的最终验证结果。
                if last_val_step != completed:
                    val_loss = evaluate(model, val_data, c)
                    outcome["final_val_loss"] = val_loss if math.isfinite(val_loss) else None
                    if math.isfinite(val_loss) and val_loss < best:
                        best = val_loss
                        outcome["best_val_loss"] = best
                        save_state(output / "best.pt", model, optimizer, completed)
                    if not math.isfinite(val_loss):
                        status = "diverged"
                    # 单独的最终验证记录可能与上一行同 step；汇总器按 step 去重取最后验证值。
                    logger.record(completed, completed * c["batch_size"] * c["context_length"], lr, None, val_loss)
                save_state(output / "final.pt", model, optimizer, completed)
        outcome["status"] = status
        outcome["training_wall_seconds"] = time.perf_counter() - started
        if torch.device(c["device"]).type == "cuda":
            outcome["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(torch.device(c["device"]))
        # 生成样例前先落盘训练结果，避免生成阶段超时丢失训练状态说明。
        write_outcome(output / "outcome.json", outcome)
        if (output / "best.pt").exists():
            # 展示验证集最佳检查点的文本，明确区别于 final.pt。
            state = torch.load(output / "best.pt", map_location=c["device"], weights_only=True)
            model.load_state_dict(state["model"])
            save_samples(model, c, output)
        return 2 if status == "diverged" else 0
    except Exception as error:
        outcome["status"] = "failed"
        outcome["reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        return 1
    finally:
        outcome["total_wall_seconds"] = time.perf_counter() - started
        outcome["tokens_processed"] = outcome["completed_steps"] * c["batch_size"] * c["context_length"]
        write_outcome(output / "outcome.json", outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("仅预览；没有 --execute 不会训练。")
        return
    raise SystemExit(run(config, args.output.resolve()))


if __name__ == "__main__":
    main()
