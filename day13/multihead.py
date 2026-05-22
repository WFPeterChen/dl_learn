import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# Day 13: Multi-Head Attention + 位置编码
#
# 对比：
#   Day 12 SimpleAttention         Day 13 MultiHeadAttention
#   ───────────────────            ─────────────────────────
#   一组 Q/K/V                     h 组 Q/K/V 并行
#   d_model 维全部一起算           切成 h 个头，每头 d_k = d_model/h 维
#   捕捉一种模式                   捕捉 h 种不同的关注模式
#   没有位置信息                   加位置编码
# ============================================================


def scaled_dot_product_attention(Q, K, V):
    """
    Day 12 写的函数，直接复用。
    注意：Q/K/V 可以是任意多维度，只要最后两维是 (seq_len, d_k)。
    Multi-head 的时候 shape 是 (batch, num_heads, seq_len, d_k)，照样能跑。
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 检查能整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        # --- 任务 1: 保存参数 ---
        # 提示：需要记住 num_heads 和每个头的维度 d_k = d_model / num_heads
        self.num_heads = num_heads   
        self.d_k = d_model // num_heads         
        self.d_model = d_model

        # 三个投影层（和 Day 12 一样）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # --- 任务 2: 输出投影 ---
        # 拼回所有头后，再过一个 Linear 混合信息。维度 d_model -> d_model
        self.W_o = self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        输入 x shape: (batch, seq_len, d_model)
        输出 shape:  (batch, num_heads, seq_len, d_k)

        关键操作：先 view 把 d_model 分成 (num_heads, d_k)，再 transpose 把 num_heads 提前
        """
        batch_size, seq_len, _ = x.size()

        # --- 任务 3: 先 view/reshape ---
        # 把最后一维 d_model 拆成 (num_heads, d_k)
        # 提示：x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # 结果 shape: (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # --- 任务 4: transpose 调整维度顺序 ---
        # 我们希望 num_heads 在 batch 后面，这样每个头能独立跑 attention
        # 提示：x.transpose(1, 2) 交换第 1 维和第 2 维
        # 结果 shape: (batch, num_heads, seq_len, d_k)
        x = x.transpose(1, 2)

        return x

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # 投影生成 Q, K, V（和 Day 12 一样，shape 都是 (batch, seq_len, d_model)）
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # --- 任务 5: 把 Q, K, V 切成多头 ---
        # 提示：调用 self.split_heads(...)
        # 切完 shape: (batch, num_heads, seq_len, d_k)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 跑 attention（Day 12 的函数直接用，它能处理任意 batch 维度）
        # 这里每个头独立跑 —— 本质是 num_heads 个并行的 attention
        output, weights = scaled_dot_product_attention(Q, K, V)
        # output shape: (batch, num_heads, seq_len, d_k)
        # weights shape: (batch, num_heads, seq_len, seq_len)

        # --- 任务 6: 把多头拼回来 ---
        # 步骤 a: transpose 把 seq_len 挪回第 1 位
        #         shape: (batch, seq_len, num_heads, d_k)
        # 步骤 b: contiguous() 让内存连续（transpose 后不连续，view 会报错）
        # 步骤 c: view 把最后两维合并成 d_model
        #         shape: (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous()
        output = output = output.view(batch_size, seq_len, self.d_model)
        # --- 任务 7: 过输出投影 ---
        output = self.W_o(output)

        return output, weights


# ============================================================
# 位置编码（正弦式）—— 这部分我直接给你，重点看思想
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建一个 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # 除数项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数位置用 sin，奇数位置用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer: 不是训练参数，但要跟着模型移到 GPU
        # 加 batch 维度变成 (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # 截取前 seq_len 个位置的编码，加到输入上
        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    print("输入 x shape:", x.shape)

    # 先加位置编码
    pos_enc = PositionalEncoding(d_model)
    x_with_pos = pos_enc(x)
    print("加位置编码后 shape:", x_with_pos.shape)
    print("位置编码前 x[0, 0]:", x[0, 0])
    print("位置编码后 x[0, 0]:", x_with_pos[0, 0])
    print("(可以看到每个位置加了一个固定的 sin/cos 向量)")
    print()

    # 送进 multi-head attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mha(x_with_pos)

    print("输出 shape:", output.shape)
    print("注意力权重 shape:", weights.shape)  # 应该是 (batch, num_heads, seq_len, seq_len)
    print()

    # 展示每个头的注意力权重
    print("=" * 60)
    print(f"头 0 的 attention 矩阵 ({seq_len}x{seq_len}):")
    print(weights[0, 0])
    print(f"\n头 1 的 attention 矩阵 ({seq_len}x{seq_len}):")
    print(weights[0, 1])
    print("\n两个头学的是不同的关注模式（权重分布不一样）")
