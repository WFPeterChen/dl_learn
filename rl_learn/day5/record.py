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
ALGO_TAG = "q_fqi"    # 改这个切换不同算法的 Q

Q_PATH = os.path.join(os.path.dirname(__file__), f"trained_Q_{ALGO_TAG}.npy")
if not os.path.exists(Q_PATH):
    raise FileNotFoundError(
        f"找不到 {Q_PATH}，请先运行训练脚本生成它"
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
    videos_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(videos_dir, exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # 录 5 个不同 seed 的 episode (seed 与 evaluate.py 一致, 方便对照)
    seeds = [10000, 10001, 10002, 10003, 10004]
    print(f"=== 录制 [{ALGO_TAG}] 的 greedy policy 在 {len(seeds)} 个 seed 上的表现 ===\n")

    for seed in seeds:
        frames, ret, length = run_episode(env, greedy_policy, seed=seed)
        out_path = os.path.join(
            videos_dir, f"{ALGO_TAG}_seed{seed}_len{length:03d}_ret{int(ret):03d}.mp4"
        )
        imageio.mimsave(out_path, frames, fps=50)
        print(f"  seed={seed}: 撑了 {length:>3d} 步, return={ret:>4.0f}  →  {os.path.basename(out_path)}")

    env.close()


if __name__ == "__main__":
    main()
