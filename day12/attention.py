import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Day 12: 手写 Scaled Dot-Product Attention
#
# 今天的目标：理解 Q/K/V 怎么互动
# 不训练、不用真实数据，只让一个 attention 层跑一次，看输出
#
# Toy 场景：
#   一句"句子"有 4 个词（seq_len=4）
#   每个词的 embedding 是 8 维（d_model=8）
#   输入 shape: (batch=1, seq_len=4, d_model=8)
# ============================================================


def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V shape: (batch, seq_len, d_k)
    返回: (output, attention_weights)
        output shape: (batch, seq_len, d_k)  -- 加权后的 V
        attention_weights shape: (batch, seq_len, seq_len)  -- 每个位置对其他位置的注意力
    """
    # --- 任务 1: 算 Q 和 K 的点积 ---
    # 提示：torch.matmul(Q, K.transpose(-2, -1))
    #   Q shape: (batch, seq_len, d_k)
    #   K.transpose(-2, -1) shape: (batch, d_k, seq_len)
    #   结果 shape: (batch, seq_len, seq_len)
    # transpose(-2, -1) 把最后两个维度交换，保留 batch 维度
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # --- 任务 2: 缩放 ---
    # 提示：除以 sqrt(d_k)
    #   d_k 就是 Q 的最后一维大小：Q.size(-1)
    #   开根号：d_k ** 0.5 或 torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    d_k = Q.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # --- 任务 3: Softmax ---
    # 提示：F.softmax(scores, dim=-1)
    #   dim=-1 表示在最后一维上做 softmax（每一行和为 1）
    #   为什么是 -1？因为 scores[i][j] 表示"位置 i 对位置 j 的关注度"
    #   我们希望"位置 i 对所有位置的关注度加起来 = 1"，所以在 j 那一维（最后一维）做 softmax
    attention_weights = F.softmax(scores, dim=-1)

    # --- 任务 4: 加权 V ---
    # 提示：torch.matmul(attention_weights, V)
    #   attention_weights shape: (batch, seq_len, seq_len)
    #   V shape: (batch, seq_len, d_k)
    #   结果 shape: (batch, seq_len, d_k)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class SimpleAttention(nn.Module):
    """
    一个极简的 self-attention 层。
    输入 X -> 用三个 Linear 生成 Q, K, V -> 送进 attention
    """
    def __init__(self, d_model):
        super().__init__()
        # --- 任务 5: 创建三个 Linear 层，把输入映射到 Q, K, V ---
        # 提示：都是 d_model -> d_model 的映射
        #   self.W_q = nn.Linear(d_model, d_model)
        #   self.W_k, self.W_v 同理
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # --- 任务 6: 用三个 Linear 生成 Q, K, V ---
        # 提示：Q = self.W_q(x) 就行，Linear 会自动处理 batch 和 seq 维度
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights


if __name__ == "__main__":
    torch.manual_seed(42)

    # Toy 输入：1 个 batch，4 个 token，每个 8 维
    batch_size = 1
    seq_len = 4
    d_model = 8

    # 随机生成 4 个词的 embedding
    x = torch.randn(batch_size, seq_len, d_model)
    print("输入 x shape:", x.shape)
    print("x[0] (4 个词的 embedding):")
    print(x[0])
    print()

    # 建立 attention 层
    attn = SimpleAttention(d_model)
    output, weights = attn(x)

    print("输出 shape:", output.shape)
    print("注意力权重 shape:", weights.shape)
    print()
    print("=" * 60)
    print("注意力权重矩阵 (4x4):")
    print("  行 = 每个位置的 Query")
    print("  列 = 每个位置的 Key")
    print("  值 = '位置 i 对位置 j 的关注度'（每行和=1）")
    print("=" * 60)
    print(weights[0])  # (4, 4) 矩阵
    print()
    print("每行加起来（应该都 ≈ 1.0）:")
    print(weights[0].sum(dim=-1))
