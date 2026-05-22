import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# Day 14: 字符级 GPT（Decoder-only Transformer）
#
# 架构（自下往上）：
#   token_ids (B, T)
#     └─ Token Embedding + Positional Embedding  → (B, T, d_model)
#     └─ N × TransformerBlock
#            ├─ LayerNorm → Multi-Head Attention(带 causal mask) → 加残差
#            └─ LayerNorm → FFN (d_model → 4*d_model → d_model)  → 加残差
#     └─ 最后 LayerNorm
#     └─ Linear head → (B, T, vocab_size)  ← 每个位置预测下一个字符
#
# 关键新概念：causal mask
#   语言模型是"看前面猜后面"，不能让位置 i 看到位置 j>i 的信息。
#   做法：在 softmax 之前，把上三角（不含对角线）的 scores 设成 -inf。
#   softmax(-inf) = 0，相当于完全遮挡未来位置。
# ============================================================


class MultiHeadAttention(nn.Module):
    """
    和 Day 13 几乎一样，新增 causal mask 支持。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, num_heads, T, d_k)

    def forward(self, x, mask=None):
        B, T, _ = x.size()

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        # Q, K, V shape: (B, num_heads, T, d_k)

        # --- 任务 1: 算 scores 并加 causal mask ---
        # 提示：
        #   scores = Q @ K^T / sqrt(d_k)  shape: (B, num_heads, T, T)
        #   如果 mask 不为 None，把 mask==0 的位置设成 -inf
        #     用法：scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, num_heads, T, d_k)

        # 拼回多头 (和 Day 13 一样)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)
        return out


class FeedForward(nn.Module):
    """
    Transformer 的 FFN：两层 Linear，中间 ReLU（或 GELU）。
    维度变化：d_model → 4*d_model → d_model
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    一个 Transformer Decoder Block（GPT 风格，Pre-LN）：
        x = x + MHA(LN(x), mask)
        x = x + FFN(LN(x))
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, mask=None):
        # --- 任务 2: 实现 Pre-LN 残差结构 ---
        # 步骤：
        #   a) attention 支路：先 LayerNorm，过 attention，再加回 x
        #      x = x + self.attn(self.ln1(x), mask)
        #   b) FFN 支路：先 LayerNorm，过 FFN，再加回 x
        #      x = x + self.ffn(self.ln2(x))
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))

        return x


class GPT(nn.Module):
    """
    字符级 GPT。
    """
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size  # 最长能看多少个 token

        # Token 嵌入 + 位置嵌入（这里用可学习的位置嵌入，比正弦式简单）
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # N 个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Causal mask：下三角矩阵，shape (1, 1, block_size, block_size)
        # 1 表示"可以看"，0 表示"遮住"
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer('causal_mask', mask)

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) 整数序列
        targets: (B, T) 每个位置的下一个字符 id；如果给了就顺便算 loss
        """
        B, T = idx.size()
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"

        # token + 位置
        pos = torch.arange(T, device=idx.device)  # (T,)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, d_model)
        x = self.drop(x)

        # 取出前 T×T 的 mask
        mask = self.causal_mask[:, :, :T, :T]

        # 过每一层
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # 算 cross-entropy loss
        # 把 (B, T, V) 压成 (B*T, V)，targets 压成 (B*T,)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成：从 idx 开始，每次预测下一个 token 并续到末尾。
        idx: (B, T) 起始上下文
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 只保留最后 block_size 个 token
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            # 取最后一个位置的 logits：(B, vocab_size)
            logits = logits[:, -1, :] / temperature

            # top-k 过滤（只从概率最高的 k 个里采样）
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)

        self.train()
        return idx


if __name__ == "__main__":
    # 小测试：随便造点数据，看模型能不能跑通前向
    torch.manual_seed(42)

    vocab_size = 100
    model = GPT(vocab_size=vocab_size, d_model=64, num_heads=4, num_layers=2, block_size=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    x = torch.randint(0, vocab_size, (2, 16))
    y = torch.randint(0, vocab_size, (2, 16))
    logits, loss = model(x, targets=y)
    print(f"logits shape: {logits.shape}  (期望: (2, 16, {vocab_size}))")
    print(f"loss: {loss.item():.4f}  (期望: 约 ln({vocab_size}) ≈ {math.log(vocab_size):.4f})")

    # 生成测试
    context = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(context, max_new_tokens=10)
    print(f"生成结果 shape: {out.shape}  (期望: (1, 11))")
