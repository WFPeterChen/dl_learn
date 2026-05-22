import torch
import pickle
from model import GPT

# 加载模型和原文
DATA_DIR = '/home/peter/dl_learn/day14'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(f'{DATA_DIR}/doupo.txt', 'r', encoding='utf-8') as f:
    corpus = f.read().replace('\r\n', '\n').replace('\r', '\n')

ckpt = torch.load(f'{DATA_DIR}/gpt.pt', weights_only=False)
meta = ckpt['meta']
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']

model = GPT(vocab_size=vocab_size, d_model=192, num_heads=6, num_layers=4,
            block_size=128, dropout=0.2).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

# 生成一段
start = "萧炎"
ids = torch.tensor([stoi[c] for c in start], device=device).unsqueeze(0)
out = model.generate(ids, max_new_tokens=300, temperature=0.8, top_k=40)
text = ''.join(itos[i] for i in out[0].tolist())
print(f"=== 生成文本 ===\n{text}\n")

# 从生成文本里切滑动窗口，找最长能在原文里搜到的片段
print("=== 原文匹配检查 ===")
max_len_found = 0
max_str = ""
for start_i in range(len(text) - 3):
    # 从每个起点往后伸长，找能在原文匹配的最长子串
    for end_i in range(start_i + 4, min(start_i + 60, len(text))):
        sub = text[start_i:end_i]
        if sub in corpus:
            if len(sub) > max_len_found:
                max_len_found = len(sub)
                max_str = sub
        else:
            break

print(f"生成文本里最长能在原文找到的连续片段: {max_len_found} 字")
print(f"片段内容: 「{max_str}」")
print()
print("解读：")
print("  < 8 字  → 完全自己编的（常用词组偶然重合）")
print("  8-15 字 → 学到了常见搭配，仍算泛化")
print("  > 20 字 → 有明显的背书成分")
