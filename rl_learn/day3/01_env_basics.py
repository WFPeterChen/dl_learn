"""day3 / 01：gymnasium CartPole 入门 —— 把环境 API 摸一遍。

CartPole 是 RL 教科书里的 "hello world"：
  - 一辆小车 (cart) 在 1D 轨道上
  - 车上立着一根杆子 (pole)，初始几乎竖直
  - 你每步推车向左 / 向右，让杆子不倒下
  - 杆倒（angle 超阈）或车出界 → episode 结束

物理 4 维状态：
    s = (x, x_dot, theta, theta_dot)
       x       : 车的位置
       x_dot   : 车的速度
       theta   : 杆的角度（垂直为 0）
       theta_dot: 杆的角速度

动作 2 维（离散）：
    a = 0 → 向左推一个固定大小的力
    a = 1 → 向右推一个固定大小的力

奖励：每多撑一步 +1，所以总 reward = 撑住的步数。
"""
import gymnasium as gym
import numpy as np


def main():
    # ---- 1. 创建环境 ----------------------------------------------------
    # gym.make 接收环境 id；CartPole-v1 是 v0 的修订版（最大步数 500，奖励上限也是 500）
    # render_mode 控制可视化：
    #   None         : 不渲染，最快（训练时用）
    #   "rgb_array"  : 每次调 env.render() 返回一个 (H, W, 3) 的 uint8 numpy 数组（录视频用）
    #   "human"      : 弹一个 pygame 窗口实时看（headless server 上用不了）
    env = gym.make("CartPole-v1", render_mode=None)

    # ---- 2. 看清楚观测 / 动作空间 ---------------------------------------
    # observation_space 和 action_space 是 gym.spaces.* 对象
    #   Box(low, high, shape, dtype): 连续向量空间
    #   Discrete(n)                 : 0..n-1 的离散整数
    # 这两个对象是 RL 算法的"接线图"——后面写 DQN / PPO 都靠这里读维度
    print("observation_space :", env.observation_space)
    print("  low  :", env.observation_space.low)
    print("  high :", env.observation_space.high)
    print("  shape:", env.observation_space.shape)
    print("action_space      :", env.action_space)
    print("  n    :", env.action_space.n)
    print()

    # ---- 3. reset：拿初始状态 s_0 ---------------------------------------
    # gymnasium 0.26+ 的新 API：reset 返回 (obs, info) 二元组
    #   obs : 初始观测 s_0，numpy array
    #   info: dict，存额外调试信息（这里基本是空的）
    # seed 控制初始扰动的随机性，复现实验时务必 seed
    obs, info = env.reset(seed=42)
    print(f"初始 obs s_0 = {obs}")
    print(f"info        = {info}")
    print()

    # ---- 4. step：往前推一步 --------------------------------------------
    # gymnasium step 返回 5 元组（这是 0.26 版本的 breaking change，老教程很多还在用 4 元组，注意区分）：
    #   obs        : s_{t+1}
    #   reward     : r_{t+1}
    #   terminated : bool —— "自然结束"（杆倒了 / 车出界），任务真的完成
    #   truncated  : bool —— "时间到了"（episode 跑满 500 步被截断），任务还没完
    #   info       : dict
    #
    # 为什么要把 done 拆成 terminated 和 truncated？
    #   做 bootstrapping 时（Q(s,a) ← r + γ·max_a' Q(s', a')），
    #   只有 terminated=True 才能把 γ·V(s') 砍成 0；truncated=True 时 s' 后面其实还有未来价值，
    #   不该当成 0 处理。老的 4 元组 done 把两件事混在一起，是 RL bug 重灾区。
    action = env.action_space.sample()        # 随机一个动作（这里就是 0 或 1）
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"action a_0       = {action}")
    print(f"next obs s_1     = {obs}")
    print(f"reward r_1       = {reward}")
    print(f"terminated       = {terminated}")
    print(f"truncated        = {truncated}")
    print()

    # ---- 5. 完整一个 episode：random policy ------------------------------
    # 跑 5 个 episode，看 random policy 平均能撑多久
    n_episodes = 5
    lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        ep_return = 0.0
        for t in range(1000):                 # 兜底上限，正常 500 步内会 truncated
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        lengths.append(ep_return)
        print(f"  episode {ep}: 撑了 {int(ep_return):>3d} 步, "
              f"结束原因 = {'terminated' if terminated else 'truncated'}")
    print(f"\nrandom policy 平均回报 = {np.mean(lengths):.1f} (满分 500)")

    # ---- 6. 收尾 --------------------------------------------------------
    # env.close() 释放渲染窗口、子进程等资源；headless 模式下其实可省，但养成好习惯
    env.close()


if __name__ == "__main__":
    main()
