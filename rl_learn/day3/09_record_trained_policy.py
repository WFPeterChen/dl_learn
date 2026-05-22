"""用训练好的 MC 策略（ε=0）运行一个 episode 并录制视频。

前提：已经运行过训练脚本，生成了 trained_Q.npy
"""
import os
import numpy as np
import gymnasium as gym
import imageio


# ----------------------------------------------------------------------
# 与训练脚本完全一致的状态离散化（必须一致，否则 Q 表索引对不上）
# ----------------------------------------------------------------------
X_BINS         = np.array([-0.8, 0.8])
X_DOT_BINS     = np.array([-0.5, 0.5])
THETA_BINS     = np.array([-0.105, -0.017, 0.0, 0.017, 0.105])
THETA_DOT_BINS = np.array([-0.87, -0.5, 0.0, 0.5, 0.87])

N_X         = len(X_BINS) + 1
N_X_DOT     = len(X_DOT_BINS) + 1
N_THETA     = len(THETA_BINS) + 1
N_THETA_DOT = len(THETA_DOT_BINS) + 1


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


# ----------------------------------------------------------------------
# 加载训练好的 Q 表
# ----------------------------------------------------------------------
Q_PATH = os.path.join(os.path.dirname(__file__), "trained_Q.npy")
if not os.path.exists(Q_PATH):
    raise FileNotFoundError(
        f"找不到 {Q_PATH}，请先运行训练脚本（day3/08 等）生成 trained_Q.npy"
    )
Q = np.load(Q_PATH)


def greedy_policy(obs):
    """确定性贪心策略（ε=0）：永远选择当前状态下 Q 值最大的动作。"""
    box = discretize(obs)
    return int(np.argmax(Q[box]))


# ----------------------------------------------------------------------
# 采集 episode 并保存视频
# ----------------------------------------------------------------------
def run_episode(env, policy_fn, seed=0):
    """跑一个 episode，返回 (frames, total_reward, length)。"""
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]          # 第 0 帧：初始状态
    total_reward = 0.0
    length = 0
    for t in range(1000):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return frames, total_reward, length


def main():
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames, ret, length = run_episode(env, greedy_policy, seed=0)

    out_path = os.path.join(figs_dir, "trained_policy.mp4")
    imageio.mimsave(out_path, frames, fps=50)

    print(f"训练好的贪心策略 (ε=0): 撑了 {length} 步, return={ret:.0f}, "
          f"保存 {len(frames)} 帧 → {out_path}")

    env.close()


if __name__ == "__main__":
    main()
