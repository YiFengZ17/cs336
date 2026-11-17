"""实验模型变体：复用作业模块，不修改作业的默认实现。"""

import torch

from cs336_basics.part2_model.linear import Linear
from cs336_basics.part2_model.silu import Silu
from cs336_basics.part2_model.transformer_lm import TransformerLM


class SiLUFeedForward(torch.nn.Module):
    def __init__(self, d_model, device):
        super().__init__()
        # 按讲义设置为 4*d_model，近似匹配 SwiGLU 的参数量。
        self.w1 = Linear(d_model, 4 * d_model, device=device)
        self.w2 = Linear(4 * d_model, d_model, device=device)
        self.silu = Silu()

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)))


class PostNormBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        # 复用已经初始化的子模块，保持与基线相同的参数名和初始权重。
        self.rmsnorm1 = block.rmsnorm1
        self.rmsnorm2 = block.rmsnorm2
        self.rope = block.rope
        self.Attention = block.Attention
        self.swiglu = block.swiglu

    def forward(self, x, token_positions=None):
        y = self.rmsnorm1(x + self.Attention(x, token_positions))
        return self.rmsnorm2(y + self.swiglu(y))


def build_model(config, device):
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
    )
    variant = config["variant"]
    if variant == "no_norm":
        # 讲义要求移除所有 RMSNorm，包括模型末尾的归一化。
        for block in model.layers:
            block.rmsnorm1 = torch.nn.Identity()
            block.rmsnorm2 = torch.nn.Identity()
        model.rmsnorm_final = torch.nn.Identity()
    elif variant == "post_norm":
        model.layers = torch.nn.ModuleList(PostNormBlock(b) for b in model.layers)
    elif variant == "nope":
        # 只取消旋转位置编码，保留原来的因果掩码。
        for block in model.layers:
            block.Attention.rope = None
            block.rope = None
    elif variant == "silu":
        for block in model.layers:
            block.swiglu = SiLUFeedForward(config["d_model"], device)
    elif variant == "tied":
        # 自选改进：输入嵌入与输出头共享参数，并使用小尺度初始化。
        # 这是“权重共享 + 共享矩阵初始化”的组合实验，不能归因于单一因素。
        model.mpl.W = model.embed.vocab
        std = config.get("tied_init_std", 0.02)
        torch.nn.init.trunc_normal_(model.embed.vocab, std=std, a=-3 * std, b=3 * std)
    elif variant != "baseline":
        raise ValueError(f"未知变体：{variant}")
    return model
