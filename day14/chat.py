import torch
import pickle
from model import GPT

# ============================================================
# 和你训好的 GPT 聊天 —— 给个开头，它接着写
#
# 命令：
#   /t 0.8        设置 temperature (默认 0.8)
#   /k 40         设置 top_k (默认 40)
#   /n 200        设置一次生成多少字 (默认 200)
#   /q            退出
#   其他任何文本  = prompt，模型从这里续写
# ============================================================

DATA_DIR = '/home/peter/dl_learn/day14'
CKPT = f'{DATA_DIR}/gpt.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载
ckpt = torch.load(CKPT, weights_only=False)
meta = ckpt['meta']
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']

model = GPT(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=6,
            block_size=192, dropout=0.15).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"模型加载完毕: {n_params/1e6:.2f}M 参数, device={device}")
print(f"词表大小: {vocab_size}")

# 默认参数
temp = 0.8
top_k = 40
n_tokens = 200

print("\n提示：输入开头让模型续写。命令：/t 温度  /k top_k  /n 字数  /q 退出\n")

while True:
    try:
        line = input("📝 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见")
        break

    if not line:
        continue

    # 命令处理
    if line.startswith('/'):
        parts = line.split()
        cmd = parts[0]
        if cmd == '/q':
            print("再见")
            break
        elif cmd == '/t' and len(parts) == 2:
            temp = float(parts[1])
            print(f"  temperature = {temp}")
        elif cmd == '/k' and len(parts) == 2:
            top_k = int(parts[1])
            print(f"  top_k = {top_k}")
        elif cmd == '/n' and len(parts) == 2:
            n_tokens = int(parts[1])
            print(f"  n_tokens = {n_tokens}")
        else:
            print("  未知命令。可用：/t <温度>  /k <top_k>  /n <字数>  /q")
        continue

    # 检查输入的字符都在词表里
    unknown = [c for c in line if c not in stoi]
    if unknown:
        print(f"  (跳过词表外字符: {''.join(set(unknown))})")
    ids = [stoi[c] for c in line if c in stoi]
    if not ids:
        print("  输入里一个有效字符都没有，换一个试试")
        continue

    # 生成
    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(context, max_new_tokens=n_tokens, temperature=temp, top_k=top_k)
    text = ''.join(itos[i] for i in out[0].tolist())
    print(f"\n{text}\n{'─' * 60}")
