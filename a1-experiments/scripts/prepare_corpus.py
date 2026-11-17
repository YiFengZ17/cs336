"""使用用户已下载的训练/验证文本准备数据；默认预览，--execute 才训练 BPE。"""

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE.parent / "assignment1-basics"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--val-text", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="新的输出目录，不覆盖已有目录")
    parser.add_argument("--chunk-tokens", type=int, default=1000000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.vocab_size < 257 or args.chunk_tokens <= 0:
        parser.error("词表至少 257 项，分块大小必须为正")
    print(f"BPE 训练：{args.train_text}；验证：{args.val_text}；词表：{args.vocab_size}；输出：{args.output}")
    if not args.execute:
        print("仅预览；未读取语料或训练 BPE。")
        return
    if not args.train_text.is_file() or not args.val_text.is_file():
        parser.error("两个文本文件必须已经存在；脚本不下载数据")
    if args.train_text.resolve() == args.val_text.resolve():
        parser.error("训练集和验证集不能是同一个文件")

    import numpy as np
    from cs336_basics.part1_tokenizer.BPE import train_bpe_indexed
    from cs336_basics.part1_tokenizer.tokenizer import Tokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    specials = ["<|endoftext|>"]
    vocab, merges = train_bpe_indexed(args.train_text, args.vocab_size, specials)
    tokenizer = Tokenizer(vocab, merges, specials)
    state = {"vocab": {str(k): list(v) for k, v in tokenizer.vocab.items()},
             "merges": [[list(a), list(b)] for a, b in merges], "special_tokens": specials}
    (args.output / "tokenizer.json").write_text(json.dumps(state), encoding="utf-8")
    info = {"requested_vocab_size": args.vocab_size, "actual_vocab_size": len(tokenizer.vocab),
            "bpe_seconds": time.perf_counter() - started,
            "train_text": str(args.train_text.resolve()), "val_text": str(args.val_text.resolve())}
    dtype = np.uint16 if max(tokenizer.vocab) <= 65535 else np.uint32
    for name, text_path in [("train", args.train_text), ("val", args.val_text)]:
        # 分块写临时二进制文件，不把所有词元编号一次放入内存。
        raw_path = args.output / f"{name}.tokens.tmp"
        count = 0
        with text_path.open(encoding="utf-8") as source, raw_path.open("xb") as destination:
            stream = iter(tokenizer.encode_iterable(source))
            while True:
                chunk = list(islice(stream, args.chunk_tokens))
                if not chunk:
                    break
                if min(chunk) < 0 or max(chunk) >= len(tokenizer.vocab):
                    raise ValueError("分词器返回了词表范围外的编号")
                np.asarray(chunk, dtype=dtype).tofile(destination)
                count += len(chunk)
        if count == 0:
            raise ValueError(f"{name} 编码后为空")
        # .npy 包含 dtype 与形状，可直接被训练脚本 mmap_mode='r' 打开。
        raw = np.memmap(raw_path, mode="r", dtype=dtype, shape=(count,))
        output = np.lib.format.open_memmap(args.output / f"{name}_tokens.npy", mode="w+", dtype=dtype, shape=(count,))
        for offset in range(0, count, args.chunk_tokens):
            output[offset:offset + args.chunk_tokens] = raw[offset:offset + args.chunk_tokens]
        output.flush()
        del raw, output
        raw_path.unlink()
        info[f"{name}_tokens"] = count
        print(f"{name}: {count:,} tokens")
    from provenance import sha256
    info["source_sha256"] = {str(p): sha256(p) for p in [args.train_text, args.val_text]}
    info["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                     for p in args.output.iterdir() if p.is_file()}
    info["total_seconds"] = time.perf_counter() - started
    (args.output / "preparation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"完成：{args.output}；实际词表大小 {len(tokenizer.vocab)}")


if __name__ == "__main__":
    main()
