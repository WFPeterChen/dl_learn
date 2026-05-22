import torch
import pickle
import time
from model import GPT

# ============================================================
# Day 14 Step 3: 训练 GPT 生成斗破苍穹
#
# 训练思路：
#   1. 每步从 train.bin 里随机抽 batch_size 个起点
#   2. 每个起点取 block_size 个字作为 x，后移 1 位作为 y
#   3. 正常 forward + backward
#   4. 每隔 eval_interval 步算一次 val loss 并生成一段文本看效果
# ============================================================

# --- 超参 ---
DATA_DIR = '/home/peter/dl_learn/day14'
OUT_FILE = f'{DATA_DIR}/gpt.pt'

batch_size = 64
block_size = 192       # 每个样本最多看 192 个字（上下文更长，生成更连贯）
d_model = 256
num_heads = 8
num_layers = 6
dropout = 0.15         # 数据量变大，过拟合风险降低，可以少 dropout

learning_rate = 3e-4
max_iters = 20000      # 数据翻 3 倍，步数也加一倍
eval_interval = 500
eval_iters = 100
warmup_iters = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")


# --- 加载数据 ---
train_data = torch.load(f'{DATA_DIR}/train.bin', weights_only=True)
val_data = torch.load(f'{DATA_DIR}/val.bin', weights_only=True)
with open(f'{DATA_DIR}/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi, itos = meta['stoi'], meta['itos']
print(f"vocab_size: {vocab_size}")
print(f"train: {len(train_data):,} | val: {len(val_data):,}")


def get_batch(split):
    """随机采一个 batch: x (B, T), y (B, T)"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """train/val 各跑 eval_iters 个 batch，取平均 loss"""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def generate_sample(model, start_text="萧炎", max_new_tokens=200):
    """从 start_text 开始续写"""
    ids = [stoi[c] for c in start_text if c in stoi]
    if not ids:
        ids = [0]
    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    return ''.join(itos[i] for i in out[0].tolist())


# --- 建模型 ---
model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
    dropout=dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)


# --- 训练循环 ---
print("\n开始训练...\n")
t0 = time.time()
best_val = float('inf')

for it in range(max_iters + 1):
    # warmup 学习率
    if it < warmup_iters:
        lr = learning_rate * (it + 1) / warmup_iters
        for g in optimizer.param_groups:
            g['lr'] = lr

    # 评估
    if it % eval_interval == 0 or it == max_iters:
        losses = estimate_loss(model)
        elapsed = time.time() - t0
        print(f"[{it:>5d}/{max_iters}] train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}s")

        if losses['val'] < best_val:
            best_val = losses['val']
            torch.save({'model': model.state_dict(), 'meta': meta}, OUT_FILE)

        print("  ── 采样 ──")
        sample = generate_sample(model, start_text="萧炎", max_new_tokens=150)
        print(f"  {sample}")
        print()

    # 训练一步
    x, y = get_batch('train')
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print(f"\n训练结束，最好 val loss: {best_val:.4f}")
print(f"模型已保存到 {OUT_FILE}")
