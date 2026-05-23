"""录制 DQN 训好的 greedy policy 视频.

用 trained_dqn.pt 加载 main_net, ε=0 跑 5 个 seed, 保存 mp4.
"""
import os
import numpy as np
import torch
import gymnasium as gym
import imageio

from train_dqn import (
    QNet, DiscreteActionWrapper,
    ACTION_VALUES, DEVICE, ENV_ID,
)


def main():
    # ---- 加载训好的 main net ----
    net = QNet().to(DEVICE)
    base = os.path.dirname(__file__) or "."
    pt_path = os.path.join(base, "trained_dqn.pt")
    net.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    net.eval()

    # ---- env (带 rgb 渲染) ----
    env = DiscreteActionWrapper(
        gym.make(ENV_ID, render_mode="rgb_array"),
        ACTION_VALUES,
    )

    videos_dir = os.path.join(base, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    seeds = [10000, 10001, 10002, 10003, 10004]
    print(f"=== Recording DQN policy on {ENV_ID} ===\n")

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        frames = [env.render()]
        total = 0.0
        length = 0
        for t in range(1000):
            with torch.no_grad():
                s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a = int(net(s_t).argmax(dim=1).item())
            obs, r, term, trunc, _ = env.step(a)
            frames.append(env.render())
            total += r
            length += 1
            if term or trunc:
                break

        # gif: 每 4 帧取一帧 (1000 → 250 frames), 12.5 fps 重放 ≈ 20s
        frames_sub = frames[::4]
        out_path = os.path.join(videos_dir, f"dqn_seed{seed}_len{length:04d}.gif")
        imageio.mimsave(out_path, frames_sub, fps=12.5, loop=0)
        print(f"  seed={seed}: 撑了 {length:>4d} 步 ({length/50:.1f}s), "
              f"return={total:>5.0f}  →  {os.path.basename(out_path)}")

    env.close()


if __name__ == "__main__":
    main()
