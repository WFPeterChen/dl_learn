"""录制 iter / nail 版的 greedy policy 视频 (用细化离散化 + 1000 步 max).

跟 record.py 的差别:
  - BINS 用 train_q_fqi_iter / nail 的细化版 (5 × 5 × 11 × 11)
  - env max_episode_steps = 1000 (允许撑满 20 秒)
"""
import os
import numpy as np
import gymnasium as gym
import imageio


# ----------------------------------------------------------------------
# 细化离散化 (跟 train_q_fqi_iter.py / train_q_fqi_nail.py 一致)
# ----------------------------------------------------------------------
X_BINS         = np.array([-1.0, -0.3, 0.3, 1.0])
X_DOT_BINS     = np.array([-1.0, -0.3, 0.3, 1.0])
THETA_BINS     = np.array([-0.15, -0.08, -0.04, -0.02, -0.005,
                            0.005, 0.02, 0.04, 0.08, 0.15])
THETA_DOT_BINS = np.array([-1.5, -0.8, -0.4, -0.15, -0.05,
                            0.05, 0.15, 0.4, 0.8, 1.5])

N_X         = len(X_BINS)         + 1   # 5
N_X_DOT     = len(X_DOT_BINS)     + 1   # 5
N_THETA     = len(THETA_BINS)     + 1   # 11
N_THETA_DOT = len(THETA_DOT_BINS) + 1   # 11

MAX_STEPS = 1000


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


# ----------------------------------------------------------------------
# 加载 Q
# ----------------------------------------------------------------------
ALGO_TAG = "q_fqi_iter"   # "q_fqi_iter" / "q_fqi_nail"

Q_PATH = os.path.join(os.path.dirname(__file__), f"trained_Q_{ALGO_TAG}.npy")
Q = np.load(Q_PATH)


def greedy_policy(obs):
    return int(np.argmax(Q[discretize(obs)]))


# ----------------------------------------------------------------------
# Run episode + record
# ----------------------------------------------------------------------
def run_episode(env, policy_fn, seed=0):
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    length = 0
    for t in range(MAX_STEPS):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return frames, total_reward, length


def main():
    videos_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(videos_dir, exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    seeds = [10000, 10001, 10002, 10003, 10004]
    print(f"=== 录制 [{ALGO_TAG}] 的 greedy policy 在 {len(seeds)} 个 seed 上的表现 ===\n")

    for seed in seeds:
        frames, ret, length = run_episode(env, greedy_policy, seed=seed)
        out_path = os.path.join(
            videos_dir, f"{ALGO_TAG}_seed{seed}_len{length:04d}.mp4"
        )
        imageio.mimsave(out_path, frames, fps=50)
        print(f"  seed={seed}: 撑了 {length:>4d} 步 ({length/50:.1f}s), "
              f"原始 return={ret:>5.0f}  →  {os.path.basename(out_path)}")

    env.close()


if __name__ == "__main__":
    main()
