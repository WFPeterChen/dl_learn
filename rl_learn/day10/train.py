"""day10: 31 离散 action (-3..+3 每 0.2N) — 复用 day6 训练 + 末尾导 TorchScript."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__) or ".", "..", "day6"))
import train_dqn as t

# patch
t.N_ACTIONS = 31
t.ACTION_VALUES = list(np.linspace(-3.0, 3.0, 31))
t.TOTAL_STEPS = 200000       # 31 action 比 3 action 探索更难, 翻倍训练
t.EPS_DECAY_STEPS = 100000

THIS = os.path.abspath(os.path.dirname(__file__) or ".")
t.__file__ = os.path.join(THIS, "train.py")   # 让产物落到 day10 目录

if __name__ == "__main__":
    t.main()
    # 训完 trace + 导 TorchScript
    net = t.QNet().to(t.DEVICE)
    net.load_state_dict(torch.load(os.path.join(THIS, "trained_dqn.pt"), map_location=t.DEVICE))
    net.eval()
    ex = torch.zeros(1, 4, dtype=torch.float32, device=t.DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(net, ex)
    out = os.path.join(THIS, "trained_dqn.script.pt")
    traced.save(out)
    print(f"✓ TorchScript -> {out}")
