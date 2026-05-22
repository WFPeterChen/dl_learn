import torch
import pickle
import time
import math
from model import GPT

# ============================================================
# 从 gpt.pt 继续训练，加 cosine 学习率衰减
# 让 lr 从 1e-4 平滑降到 1e-5，把最后几个小数点榨出来
# ============================================================

DATA_DIR = '/home/peter/dl_learn/day14'
CKPT_FILE = f'{DATA_DIR}/gpt.pt'

# 超参必须和之前训练一致！
batch_size = 64
block_size = 192
d_model = 256
num_heads = 8
num_layers = 6
dropout = 0.15

# --- 继续训练的新超参 ---
lr_max = 1e-4          # 起点比原来的 3e-4 低一点
lr_min = 1e-5          # 终点
extra_iters = 8000     # 再训 8000 步
eval_interval = 500
eval_iters = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 加载数据 ---
train_data = torch.load(f'{DATA_DIR}/train.bin', weights_only=True)
val_data = torch.load(f'{DATA_DIR}/val.bin', weights_only=True)
with open(f'{DATA_DIR}/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi, itos = meta['stoi'], meta['itos']
print(f"vocab_size: {vocab_size} | train: {len(train_data):,} | val: {len(val_data):,}")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
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


def generate_sample(model, start_text="萧炎", max_new_tokens=150):
    ids = [stoi[c] for c in start_text if c in stoi]
    if not ids:
        ids = [0]
    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    return ''.join(itos[i] for i in out[0].tolist())


def cosine_lr(it, total):
    """从 lr_max 按余弦曲线平滑降到 lr_min"""
    progress = it / total
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * coeff


# --- 加载模型 ---
ckpt = torch.load(CKPT_FILE, weights_only=False)
model = GPT(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, block_size=block_size, dropout=dropout).to(device)
model.load_state_dict(ckpt['model'])
print(f"已加载 {CKPT_FILE}")

# 先评估一次，确认从 2.66 附近继续
losses = estimate_loss(model)
print(f"起点 loss: train {losses['train']:.4f} | val {losses['val']:.4f}")
best_val = losses['val']

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max,
                              betas=(0.9, 0.95), weight_decay=0.1)

print(f"\n继续训练 {extra_iters} 步，lr: {lr_max} → {lr_min}\n")
t0 = time.time()

for it in range(1, extra_iters + 1):
    # cosine lr schedule
    lr = cosine_lr(it, extra_iters)
    for g in optimizer.param_groups:
        g['lr'] = lr

    if it % eval_interval == 0 or it == extra_iters:
        losses = estimate_loss(model)
        elapsed = time.time() - t0
        print(f"[+{it:>5d}/{extra_iters}] train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        if losses['val'] < best_val:
            best_val = losses['val']
            torch.save({'model': model.state_dict(), 'meta': meta}, CKPT_FILE)
            print(f"  ✓ 保存新的最好模型 (val {best_val:.4f})")

        print("  ── 采样 ──")
        print(f"  {generate_sample(model)}\n")

    x, y = get_batch('train')
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print(f"\n继续训练结束，最好 val loss: {best_val:.4f}")
