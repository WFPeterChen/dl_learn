"""day7 / 02: 用纯 mujoco API + day6 trained DQN policy 跑控制 loop.

意义:
  这是 day10 C++ 部署的 Python 原型.
  把 gym 那层包装剥掉, 直接看 "state → NN forward → ctrl → mj_step" 这条数据链.
  C++ 翻译 = 把这个 loop 改写成 C++ 调用 LibTorch + MuJoCo C++ API.

对比 day6 (gym 包装):
  gym 内部 已经做了:
    1. obs = concat(qpos, qvel)
    2. ctrl = ACTION_VALUES[a]  (DiscreteActionWrapper)
    3. 调 mj_step
    4. r = ... (CartPole reward)
    5. term = |θ| > 0.2  (gym 自动判)

  我们现在自己做这 5 步, 不通过 gym.
"""
import os
import sys
import numpy as np
import torch
import mujoco
import matplotlib.pyplot as plt

# import day6 训好的 net 定义 + 加载 .pt
sys.path.insert(0, os.path.join(os.path.dirname(__file__) or ".", "..", "day6"))
from train_dqn import QNet, ACTION_VALUES, DEVICE

import gymnasium.envs.mujoco
MJCF_PATH = os.path.join(
    os.path.dirname(gymnasium.envs.mujoco.__file__),
    "assets", "inverted_pendulum.xml",
)
PT_PATH = os.path.join(os.path.dirname(__file__) or ".", "..", "day6", "trained_dqn.pt")


# ====================================================================
# 4 步把 mujoco state → DQN action → mujoco ctrl
# ====================================================================
def dqn_action(net, qpos, qvel):
    """day6 DQN policy 的 mujoco-friendly wrapper.

    输入 mujoco state (qpos, qvel)
    输出 mujoco ctrl (单 float)
    """
    # ① state: concat 成 4 维 obs (跟 day6 训练时一致)
    obs = np.concatenate([qpos, qvel]).astype(np.float32)

    # ② tensor: 加 batch 维 + 上 GPU
    s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # ③ greedy: argmax NN forward
    with torch.no_grad():
        a_idx = int(net(s_t).argmax(dim=1).item())

    # ④ 离散 action → 连续 ctrl
    return ACTION_VALUES[a_idx]


# ====================================================================
# Pure mujoco loop with DQN
# ====================================================================
def run_dqn_episode(model, data, net, max_steps=1000, init_theta=0.05):
    mujoco.mj_resetData(model, data)
    data.qpos[1] = init_theta
    mujoco.mj_forward(model, data)

    history = {"t": [], "x": [], "theta": [], "x_dot": [], "theta_dot": [], "u": []}
    for step in range(max_steps):
        # 自己的 4 步循环, 不通过 gym
        u = dqn_action(net, data.qpos.copy(), data.qvel.copy())
        data.ctrl[0] = u
        mujoco.mj_step(model, data)

        history["t"].append(data.time)
        history["x"].append(data.qpos[0])
        history["theta"].append(data.qpos[1])
        history["x_dot"].append(data.qvel[0])
        history["theta_dot"].append(data.qvel[1])
        history["u"].append(u)

        # 自己判 terminate (gym 内部就是这两条)
        if abs(data.qpos[1]) > 0.2:
            print(f"  terminate: |θ|={abs(data.qpos[1]):.3f} > 0.2 at t={data.time:.2f}s")
            break

    return history


def plot_compare_with_pd(hist_dqn, hist_pd_path_unused, savepath):
    """画 DQN 跑的 4 维 state + ctrl."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    t = hist_dqn["t"]

    axes[0, 0].plot(t, hist_dqn["theta"], color='blue')
    axes[0, 0].axhline(0.2,  ls='--', c='r', alpha=0.5)
    axes[0, 0].axhline(-0.2, ls='--', c='r', alpha=0.5)
    axes[0, 0].set_xlabel("t (s)"); axes[0, 0].set_ylabel("theta (rad)")
    axes[0, 0].set_title("Pole angle"); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, hist_dqn["x"], color='green')
    axes[0, 1].axhline(1,  ls='--', c='r', alpha=0.5, label='slider limit')
    axes[0, 1].axhline(-1, ls='--', c='r', alpha=0.5)
    axes[0, 1].set_xlabel("t (s)"); axes[0, 1].set_ylabel("x (m)")
    axes[0, 1].set_title("Cart position"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(t, hist_dqn["theta_dot"], color='purple')
    axes[0, 2].set_xlabel("t (s)"); axes[0, 2].set_ylabel("theta_dot (rad/s)")
    axes[0, 2].set_title("Pole angular velocity"); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(t, hist_dqn["x_dot"], color='orange')
    axes[1, 0].set_xlabel("t (s)"); axes[1, 0].set_ylabel("x_dot (m/s)")
    axes[1, 0].set_title("Cart velocity"); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].step(t, hist_dqn["u"], color='navy', where='post')
    axes[1, 1].axhline(3,  ls='--', c='r', alpha=0.4)
    axes[1, 1].axhline(-3, ls='--', c='r', alpha=0.4)
    axes[1, 1].axhline(0,  ls='--', c='gray', alpha=0.4)
    axes[1, 1].set_xlabel("t (s)"); axes[1, 1].set_ylabel("ctrl")
    axes[1, 1].set_title("DQN ctrl (discrete: -3/0/+3)"); axes[1, 1].grid(alpha=0.3)

    axes[1, 2].axis('off')
    axes[1, 2].text(0.05, 0.7,
        f"DQN policy in pure mujoco loop\n"
        f"  steps survived: {len(t)}\n"
        f"  final t: {t[-1]:.2f}s\n"
        f"  final theta: {hist_dqn['theta'][-1]:+.4f} rad\n"
        f"  final x:     {hist_dqn['x'][-1]:+.3f} m\n\n"
        f"vs PD (day7/01):\n"
        f"  PD survived 6.28s (cart 撞墙)\n"
        f"  DQN should survive 20s (max episode)\n",
        fontsize=10, family='monospace', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(savepath, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    # ---- 1. 加载 mujoco model ----
    print(f"Loading MJCF: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data  = mujoco.MjData(model)

    # ---- 2. 加载 day6 DQN ----
    print(f"Loading DQN:  {PT_PATH}")
    net = QNet().to(DEVICE)
    net.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  device: {DEVICE}, n_params: {n_params}")

    # ---- 3. 跑 DQN 控制 loop in pure mujoco ----
    print(f"\n=== Running DQN policy in pure mujoco loop ===")
    history = run_dqn_episode(model, data, net, max_steps=1000, init_theta=0.05)

    print(f"\n  survived: {len(history['t'])} steps ({history['t'][-1]:.2f}s)")
    print(f"  final theta: {history['theta'][-1]:+.4f} rad")
    print(f"  final x:     {history['x'][-1]:+.3f} m")

    # ---- 4. 出图 ----
    figs_dir = os.path.join(os.path.dirname(__file__) or ".", "figs")
    os.makedirs(figs_dir, exist_ok=True)
    plot_compare_with_pd(history, None, os.path.join(figs_dir, "02_dqn_in_mujoco.png"))
    print(f"\n✓ figs → figs/02_dqn_in_mujoco.png")


if __name__ == "__main__":
    main()
