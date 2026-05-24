"""day11: 31 action + reward shaping ("钉钉子") + 加大训练.

基于 day10:
  - reward shaping: 越接近 (θ=0, θ_dot=0) 奖励越大, terminate=0
  - 训练步数 200k → 400k
  - viewer 复用 day10 binary, 只是加载 day11 的 .script.pt
"""
import os, sys
import numpy as np
import torch
import gymnasium as gym


# ===== Reward shaping wrapper =====
class ShapeReward(gym.Wrapper):
    """r = max(0, 1 - 10θ² - 0.1·θ_dot²);  terminate 时 r = 0.

    设计:
      θ=0, θ_dot=0:      r = 1.0  (完美直立 + 静止)
      θ=0.1 (~5.7°):     r = 0.9
      θ=0.15:            r ≈ 0.77
      θ=0.2 (terminate): r ≈ 0  (杆倒)
      → 持续逼近 θ=0 才能拿到 max reward
    """
    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        if term:
            r = 0.0
        else:
            _, theta, _, theta_dot = obs
            r = max(0.0, 1.0 - 10.0 * theta**2 - 0.1 * theta_dot**2)
        return obs, r, term, trunc, info


# Monkey-patch gym.make 让 day6 创建的 env 自动套 ShapeReward
_orig_make = gym.make
def _patched_make(*args, **kw):
    return ShapeReward(_orig_make(*args, **kw))
gym.make = _patched_make


# 复用 day6 训练逻辑
sys.path.insert(0, os.path.join(os.path.dirname(__file__) or ".", "..", "day6"))
import train_dqn as t

t.N_ACTIONS = 31
t.ACTION_VALUES = list(np.linspace(-3.0, 3.0, 31))
t.TOTAL_STEPS = 400000           # 加倍 day10 的 200k
t.EPS_DECAY_STEPS = 150000

THIS = os.path.abspath(os.path.dirname(__file__) or ".")
t.__file__ = os.path.join(THIS, "train.py")


if __name__ == "__main__":
    t.main()
    # export TorchScript
    net = t.QNet().to(t.DEVICE)
    net.load_state_dict(torch.load(os.path.join(THIS, "trained_dqn.pt"), map_location=t.DEVICE))
    net.eval()
    ex = torch.zeros(1, 4, dtype=torch.float32, device=t.DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(net, ex)
    out = os.path.join(THIS, "trained_dqn.script.pt")
    traced.save(out)
    print(f"✓ TorchScript -> {out}")
