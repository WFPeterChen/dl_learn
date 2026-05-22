import torch
import pickle
import glob
import os

# ============================================================
# Day 14 Step 1: 数据预处理
#
# 读 corpus/ 目录下所有 .txt，拼在一起建字符表。
# 如果 corpus/ 不存在，退回到读单个 doupo.txt。
# ============================================================

CORPUS_DIR = '/home/peter/dl_learn/day14/corpus'
FALLBACK_FILE = '/home/peter/dl_learn/day14/doupo.txt'
TRAIN_FILE = '/home/peter/dl_learn/day14/train.bin'
VAL_FILE = '/home/peter/dl_learn/day14/val.bin'
META_FILE = '/home/peter/dl_learn/day14/meta.pkl'


def main():
    # 找输入文件列表
    if os.path.isdir(CORPUS_DIR):
        files = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.txt')))
        print(f"从 {CORPUS_DIR} 读到 {len(files)} 个文件:")
        for f in files:
            print(f"  - {os.path.basename(f)} ({os.path.getsize(f)/1e6:.1f} MB)")
    else:
        files = [FALLBACK_FILE]
        print(f"corpus/ 目录不存在，使用 {FALLBACK_FILE}")

    # 依次读取并拼起来，每本书之间加个分隔
    parts = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            t = f.read().replace('\r\n', '\n').replace('\r', '\n')
        parts.append(t.strip())
    text = '\n\n'.join(parts)

    print(f"\n原始文本: {len(text):,} 字符")

    # 建字符表
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size}")

    # char -> int 和 int -> char
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 编码整篇文本
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"编码后: {data.shape}")

    # 90% 训练，10% 验证
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"训练集: {len(train_data):,} | 验证集: {len(val_data):,}")

    # 保存
    torch.save(train_data, TRAIN_FILE)
    torch.save(val_data, VAL_FILE)
    with open(META_FILE, 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)

    print(f"\n已保存到 {TRAIN_FILE}, {VAL_FILE}, {META_FILE}")

    # 预览：前 200 个 token 解码回来
    sample = train_data[:200].tolist()
    decoded = ''.join(itos[i] for i in sample)
    print(f"\n前 200 个 token 解码:\n{decoded}")


if __name__ == '__main__':
    main()
