"""day8 / Python: 把 day6 训好的 DQN 导出为 TorchScript, 供 C++ 加载.

为什么需要这一步:
  PyTorch 的 .pt (state_dict 形式) 不能被 C++ 直接 load.
  必须先 trace 或 script 转换成 TorchScript 中间格式, C++ 才能 load.

两种转换方式:
  torch.jit.trace(net, example):  跑一遍 example 输入, 录下计算图  ⭐ 简单
  torch.jit.script(net):          静态分析整个 nn.Module             — 复杂模型用

  我们的 QNet 是简单 MLP, trace 完全够用.

Steps:
  1. 加载 day6 state_dict → 重建 QNet
  2. torch.jit.trace 用 dummy state 录计算图
  3. save 成 .script.pt → C++ 用
  4. 用同样几个 test state 跑 forward, 打印 Q 值
     (用于跟 C++ 输出 1:1 比对验证 LibTorch 加载正确)
"""
import os
import sys
import torch

# import day6 的 QNet 定义
sys.path.insert(0, os.path.join(os.path.dirname(__file__) or ".", "..", "day6"))
from train_dqn import QNet, DEVICE


def main():
    base = os.path.dirname(__file__) or "."
    pt_path     = os.path.join(base, "..", "day6", "trained_dqn.pt")
    script_path = os.path.join(base, "trained_dqn.script.pt")

    # ---- 1. 加载 day6 state_dict ----
    net = QNet().to(DEVICE)
    net.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Loaded {pt_path}")
    print(f"  device: {DEVICE}, params: {n_params}")

    # ---- 2. Trace + save TorchScript ----
    example = torch.zeros(1, 4, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(net, example)
    traced.save(script_path)
    file_size_kb = os.path.getsize(script_path) / 1024
    print(f"\n✓ TorchScript saved → {script_path}  ({file_size_kb:.1f} KB)")

    # ---- 3. 跑 test states, 打印 Q 值 (后面跟 C++ 对比) ----
    test_states = [
        [0.0,  0.00, 0.0, 0.0],     # 完美初始
        [0.0,  0.05, 0.0, 0.0],     # 轻微右倾
        [0.0, -0.05, 0.0, 0.0],     # 轻微左倾
        [0.0,  0.15, 0.0, 0.0],     # 接近 terminate
        [0.0, -0.15, 0.0, 0.0],     # 接近 terminate (反向)
    ]
    print(f"\n=== Python reference Q values (跟 C++ 输出对比) ===")
    print(f"  state -> [Q(a=-3), Q(a=0), Q(a=+3)]  argmax")
    for s in test_states:
        st = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q = traced(st)
        a = int(q.argmax(dim=1).item())
        print(f"  {s}  ->  "
              f"[{q[0,0].item():+8.4f}, {q[0,1].item():+8.4f}, {q[0,2].item():+8.4f}]"
              f"  argmax={a} (force={[-3, 0, 3][a]:+d})")


if __name__ == "__main__":
    main()
