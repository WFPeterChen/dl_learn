"""day3 / 02：把 CartPole 录成 mp4，在 VS Code 里直接看物理。

为啥要录视频：
  - 这台机器没有显示器（headless server），不能用 render_mode="human"
  - 但 RL 看的是"轨迹的形状"——agent 学到的是 sequential decision，
    单帧静态图说明不了什么，必须看连续的运动
  - 录 mp4 是 headless 工作流的标配：训完模型录几条 episode 看行为

实现思路（手写版，不用 RecordVideo wrapper）：
  - render_mode="rgb_array" 让 env.render() 每次返回一帧 (H, W, 3) uint8 array
  - 我们手动收集 frames 列表，最后用 imageio.mimsave 写 mp4
  - 比 wrapper 透明：你能清楚看到 frame 是什么、什么时候采样
"""
import os
import gymnasium as gym
import numpy as np
import imageio


# ----------------------------------------------------------------------
# 关键 helper：跑一个 episode，把每一帧画面收集起来
# ----------------------------------------------------------------------
def run_episode(env, policy_fn, seed=0):
    """跑一个 episode，返回 (frames, total_reward, length)。

    policy_fn(obs) -> action
        每步喂一个 obs，吐一个 action。
        我们把"怎么决策"抽出来，后面要换 random / hand-crafted / DQN 都不用动这个函数。
    """
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]              # 第 0 帧：reset 之后的初始画面
    total_reward = 0.0
    length = 0
    for t in range(1000):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())      # 每步存一帧
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return frames, total_reward, length


# ----------------------------------------------------------------------
# 三种 policy 对照
# ----------------------------------------------------------------------
def random_policy(env):
    """每步均匀随机 push 0 或 1。"""
    return lambda obs: env.action_space.sample()


def always_left_policy():
    """永远向左推。CartPole 在不可控时是不稳定的，会很快倒。"""
    return lambda obs: 0


def hand_crafted_policy():
    """4 行规则的 PD 控制器（不学，纯人工启发式）。

    obs = [x, x_dot, theta, theta_dot]

    思路：杆子向右倒（theta > 0）就向右推，把车追到杆子下方让它"扶正"。
    再加 theta_dot 项做微分阻尼，否则会过冲振荡。
    系数 1.0 / 0.5 是拍脑袋的，调一下能撑更久。

    这是 Sutton 1983 的 "POLE" 系统原型解法，比任何随机搜索都强；
    但它不是"学"出来的，给我们当 baseline——RL 算法应该至少打平这个。
    """
    def pi(obs):
        x, x_dot, theta, theta_dot = obs
        # 一个简单线性组合：theta 主导，theta_dot 次要
        score = 1.0 * theta + 0.5 * theta_dot
        return 1 if score > 0 else 0     # 朝杆倒的方向推
    return pi


# ----------------------------------------------------------------------
# 主流程：分别录三个 policy 各一个 episode
# ----------------------------------------------------------------------
def main():
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # 注意：录视频必须 render_mode="rgb_array"，否则 env.render() 返回 None
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    policies = {
        "random":       random_policy(env),
        "always_left":  always_left_policy(),
        "hand_crafted": hand_crafted_policy(),
    }

    for name, pi in policies.items():
        # 同一个 seed 让三种 policy 从同一初始状态出发，对比公平
        frames, ret, length = run_episode(env, pi, seed=0)

        # imageio 写 mp4：fps=50 是 CartPole 物理仿真的真实步频（dt=0.02s）
        out_path = os.path.join(figs_dir, f"{name}.mp4")
        imageio.mimsave(out_path, frames, fps=50)

        print(f"{name:>13s}: 撑了 {length:>3d} 步, return={ret:.0f}, "
              f"saved {len(frames)} frames → {out_path}")

    env.close()


if __name__ == "__main__":
    main()
