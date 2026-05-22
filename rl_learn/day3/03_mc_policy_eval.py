"""day3 / 03：Monte Carlo policy evaluation —— 第一次"不靠 P"估值。

任务：估 v_π(s_0)，给两个 policy（random / hand_crafted），用样本平均做。

理论 baseline（CartPole-v1，γ=0.99）：
  - 撑满 500 步：v_π = (1 - 0.99^500)/(1 - 0.99) ≈ 99.34
  - 撑 22 步   ：v_π = (1 - 0.99^22)/(1 - 0.99) ≈ 19.85

对照"撑住的步数"：
  - hand_crafted 几乎每次 500（θ=0 附近线性化稳定区）
  - random      平均 ~22

数学：MC = 用样本均值代替期望
  v_π(s_0) ≈ (1/N) Σ G_0^{(i)},   G_0^{(i)} = Σ_t γ^t r_{t+1}^{(i)}

收敛速率：标准误差 SE ≈ σ/√N
  - 想把 SE 缩到 0.1：需要 N ≈ (σ/0.1)²
  - σ 越大（policy 越不一致）→ 需要更多 episode
"""
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 两个对照 policy（和 02 里一样）
# ----------------------------------------------------------------------
def hand_crafted_policy(obs):
    x, x_dot, theta, theta_dot = obs
    return 1 if (1.0 * theta + 0.5 * theta_dot) > 0 else 0


def random_policy(obs, rng):
    return int(rng.integers(0, 2))


# ----------------------------------------------------------------------
# 核心：算单个 episode 的 G_0
# ----------------------------------------------------------------------
def episode_return(env, policy_fn, gamma, seed):
    """跑一个 episode，返回 G_0 = Σ_t γ^t · r_{t+1}。

    实现技巧：从后往前累加比从前往后乘 γ^t 更稳（避免 γ^t 数值下溢）：
        G ← 0
        for r in reversed(rewards):
            G ← r + γ·G
    最终 G = r_1 + γr_2 + γ²r_3 + ... 正是 G_0。
    这个反向迭代叫 "backward computation of returns"，后面 PG / Actor-Critic 也用这个套路。
    """
    obs, _ = env.reset(seed=seed)
    rewards = []
    for t in range(1000):
        a = policy_fn(obs)
        obs, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        if term or trunc:
            break
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G, len(rewards)


# ----------------------------------------------------------------------
# MC eval：跑 N 个 episode，返回所有 G_0
# ----------------------------------------------------------------------
def mc_eval(env, policy_fn, gamma, n_episodes, seed_base=0):
    Gs = np.zeros(n_episodes)
    Ts = np.zeros(n_episodes, dtype=int)
    for i in range(n_episodes):
        Gs[i], Ts[i] = episode_return(env, policy_fn, gamma, seed=seed_base + i)
    return Gs, Ts


# ----------------------------------------------------------------------
# 可视化：收敛曲线 + 分布直方图
# ----------------------------------------------------------------------
def plot_convergence(Gs_dict, gamma, savepath):
    """两 policy 并列画：左 = 收敛曲线，右 = 回报分布。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for row, (name, Gs) in enumerate(Gs_dict.items()):
        N = len(Gs)
        n_arr = np.arange(1, N + 1)

        # 累计平均（running mean）：第 i 步的估计 = G_1..G_i 的平均
        running_mean = np.cumsum(Gs) / n_arr

        # 累计标准误（standard error of the mean）：σ / √N
        # 用 unbiased std；前 1 个样本时 std=nan，画图时跳过
        running_std = np.array([Gs[:i].std(ddof=1) if i >= 2 else 0.0 for i in n_arr])
        running_se  = running_std / np.sqrt(n_arr)

        # 左图：收敛曲线 + ±2 SE 置信带
        ax = axes[row, 0]
        ax.plot(n_arr, running_mean, lw=1.5, label="running mean")
        ax.fill_between(n_arr,
                        running_mean - 2 * running_se,
                        running_mean + 2 * running_se,
                        alpha=0.3, label="±2 SE")
        ax.set_xlabel("episode #")
        ax.set_ylabel(f"estimate of v_π(s_0)")
        ax.set_title(f"{name} —— MC convergence (γ={gamma})")
        ax.legend()
        ax.grid(alpha=0.3)

        # 右图：所有 episode return 的分布
        ax = axes[row, 1]
        ax.hist(Gs, bins=40, edgecolor="gray", alpha=0.8)
        ax.axvline(Gs.mean(), color="red", linestyle="--",
                   label=f"mean={Gs.mean():.2f}, σ={Gs.std():.2f}")
        ax.set_xlabel("episode return G_0")
        ax.set_ylabel("count")
        ax.set_title(f"{name} —— return distribution")
        ax.legend()

    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    GAMMA = 0.99
    N_EPISODES = 1000

    env = gym.make("CartPole-v1")

    # 用各自独立的 RNG 喂 random_policy，保证可复现
    rng_random = np.random.default_rng(seed=42)
    pi_random = lambda obs: random_policy(obs, rng_random)
    pi_hand   = hand_crafted_policy

    print(f"=== MC policy evaluation, γ={GAMMA}, N={N_EPISODES} ===\n")

    Gs_dict = {}
    for name, pi in [("random_policy", pi_random),
                     ("hand_crafted_policy", pi_hand)]:
        Gs, Ts = mc_eval(env, pi, GAMMA, N_EPISODES, seed_base=0)
        mean = Gs.mean()
        std  = Gs.std(ddof=1)
        se   = std / np.sqrt(N_EPISODES)
        print(f"{name}:")
        print(f"  v_π(s_0) ≈ {mean:.3f} ± {se:.3f}   (mean ± SE)")
        print(f"  σ (per-episode std) = {std:.3f}")
        print(f"  episode length: mean={Ts.mean():.1f}, max={Ts.max()}, min={Ts.min()}")
        # 理论值：撑满 T 步 = (1-γ^T)/(1-γ)
        T_med = int(np.median(Ts))
        v_theory = (1 - GAMMA**T_med) / (1 - GAMMA)
        print(f"  理论参考 (γ^T 公式, T=median={T_med}): {v_theory:.3f}\n")
        Gs_dict[name] = Gs

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    out = os.path.join(figs_dir, "mc_policy_eval.png")
    plot_convergence(Gs_dict, GAMMA, out)
    print(f"图保存到: {out}")

    env.close()


if __name__ == "__main__":
    main()
