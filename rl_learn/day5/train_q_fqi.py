"""day5 / Fitted Q-Iteration (FQI):
两阶段 - 先用 generative model + 高斯采样收集固定数据集,
然后在数据集上反复 batch Bellman update 直到 Q 收敛.

跟 train_q_generative.py 的关键区别:
  generative (1-step):  data 一边收集一边 update Q (stochastic)
  FQI:                  data 收集完冻结, Q 在固定 data 上反复迭代 (batch)

理论对应:
  FQI 等价于 model-based value iteration 的 sample 版本
  当 data 充分时, sample-mean ≈ expected value, 收敛到 Q*
"""
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ====================================================================
# 离散化
# ====================================================================
X_BINS         = np.array([-0.8, 0.8])
X_DOT_BINS     = np.array([-0.5, 0.5])
THETA_BINS     = np.array([-0.105, -0.017, 0.0, 0.017, 0.105])
THETA_DOT_BINS = np.array([-0.87, -0.5, 0.0, 0.5, 0.87])
N_X         = len(X_BINS)         + 1
N_X_DOT     = len(X_DOT_BINS)     + 1
N_THETA     = len(THETA_BINS)     + 1
N_THETA_DOT = len(THETA_DOT_BINS) + 1
N_BOXES     = N_X * N_X_DOT * N_THETA * N_THETA_DOT
N_ACTIONS   = 2


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


def sample_state(rng, mu, sigma):
    s = rng.normal(mu, sigma)
    s[0] = float(np.clip(s[0], -2.3,  2.3))     # 放宽到接近 terminal 边界 (-2.4, 2.4)
    s[2] = float(np.clip(s[2], -0.22, 0.22))    # 放宽: 允许 |θ| 越过 0.21 → 单步后 terminate
    return s.astype(np.float32)


def evaluate_greedy(env, Q, n_episodes=20, seed_base=10000):
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total = 0.0
        for t in range(1000):
            a = int(np.argmax(Q[discretize(obs)]))
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        returns[ep] = total
    return returns.mean()


# ====================================================================
# Phase 1: 收集数据集 (generative + Gaussian sampling)
# ====================================================================
def collect_dataset(env, rng, n_transitions, mu, sigma):
    """收集 N 个独立 transitions (s, a, r, s', done). 返回 numpy 数组方便向量化."""
    s_boxes      = np.zeros(n_transitions, dtype=np.int64)
    actions      = np.zeros(n_transitions, dtype=np.int64)
    rewards      = np.zeros(n_transitions, dtype=np.float64)
    s_next_boxes = np.zeros(n_transitions, dtype=np.int64)
    dones        = np.zeros(n_transitions, dtype=bool)

    env.reset(seed=0)
    for i in range(n_transitions):
        s = sample_state(rng, mu, sigma)
        env.reset()
        env.unwrapped.state = s
        a = int(rng.integers(0, N_ACTIONS))
        obs, r, term, trunc, _ = env.step(a)
        s_boxes[i]      = discretize(s)
        actions[i]      = a
        rewards[i]      = r
        s_next_boxes[i] = discretize(obs)
        dones[i]        = term
    return s_boxes, actions, rewards, s_next_boxes, dones


# ====================================================================
# Phase 2: FQI iteration (batch Bellman update)
# ====================================================================
def fqi_iterate(Q, s_boxes, actions, rewards, s_next_boxes, dones, gamma):
    """一轮 FQI: 用固定 dataset 上的 sample mean 估 Bellman target."""
    # 计算所有 transitions 的 target
    next_max = Q[s_next_boxes].max(axis=1)                # (N,)
    targets  = rewards + gamma * next_max * (~dones)      # (N,)

    # 用 sample mean 估每个 (s, a) 的新 Q 值
    sums   = np.zeros_like(Q)
    counts = np.zeros_like(Q)
    np.add.at(sums,   (s_boxes, actions), targets)
    np.add.at(counts, (s_boxes, actions), 1.0)

    Q_new = Q.copy()
    mask = counts > 0
    Q_new[mask] = sums[mask] / counts[mask]
    return Q_new


# ====================================================================
# 主流程
# ====================================================================
def main():
    rng = np.random.default_rng(seed=42)
    env_collect = gym.make("CartPole-v1")
    env_eval    = gym.make("CartPole-v1")

    # 超参
    N_TRANSITIONS = 50000     # Phase 1 数据集大小 (5× 提升, 改善 (s,a) 覆盖率)
    N_ITERATIONS  = 100       # Phase 2 迭代次数
    EVAL_EVERY    = 5
    EVAL_EPISODES = 20
    GAMMA         = 0.99

    # 高斯采样参数
    MU    = np.array([0.0, 0.0, 0.0,  0.0])
    SIGMA = np.array([0.5, 0.5, 0.10, 0.5])    # σ_θ 增大: 0.05 → 0.10, 让一些样本越过 terminal 边界

    print(f"=== Fitted Q-Iteration (FQI) + generative model ===")
    print(f"  Phase 1: collect {N_TRANSITIONS} transitions (Gaussian, μ=0)")
    print(f"  Phase 2: {N_ITERATIONS} batch Bellman iterations")
    print(f"  γ = {GAMMA},  eval every {EVAL_EVERY} iter (× {EVAL_EPISODES} episodes)")
    print()

    # ---- Phase 1 ----
    print("Phase 1: collecting dataset ...")
    s_boxes, actions, rewards, s_next_boxes, dones = collect_dataset(
        env_collect, rng, N_TRANSITIONS, MU, SIGMA
    )
    unique_sa = len(set(zip(s_boxes.tolist(), actions.tolist())))
    print(f"  数据集覆盖 unique (s, a): {unique_sa} / {N_BOXES * N_ACTIONS}")
    print(f"  terminal transitions:    {dones.sum()} / {N_TRANSITIONS}")
    print()

    # ---- Phase 2: FQI ----
    print("Phase 2: FQI iterations ...")
    Q = np.zeros((N_BOXES, N_ACTIONS))

    iter_hist = [0]
    eval_hist = [evaluate_greedy(env_eval, Q, EVAL_EPISODES)]

    for k in range(1, N_ITERATIONS + 1):
        Q_new = fqi_iterate(Q, s_boxes, actions, rewards, s_next_boxes, dones, GAMMA)
        delta = np.abs(Q_new - Q).max()
        Q = Q_new

        if k % EVAL_EVERY == 0 or k == 1:
            eval_mean = evaluate_greedy(env_eval, Q, EVAL_EPISODES)
            iter_hist.append(k)
            eval_hist.append(eval_mean)
            print(f"  iter {k:>3d}: ‖ΔQ‖∞ = {delta:>7.4f},  greedy eval mean = {eval_mean:>6.1f}")

    # ---- 保存 ----
    base = os.path.dirname(__file__)
    Q_path = os.path.join(base, "trained_Q_q_fqi.npy")
    np.save(Q_path, Q)

    # ---- 出图 ----
    figs_dir = os.path.join(base, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iter_hist, eval_hist, marker='o', lw=2, color='purple',
            label=f'FQI on {N_TRANSITIONS} Gaussian transitions')
    ax.axhline(500, color='green',  ls='--', lw=1.5, label='MC / Q-uniform 上限')
    ax.axhline(271, color='orange', ls=':',  lw=1.5, label='Sarsa ε-greedy 80k')
    ax.axhline(140, color='red',    ls=':',  lw=1.5, label='Q-generative 1-step 50k')
    ax.set_xlabel('FQI iteration')
    ax.set_ylabel('greedy eval mean return (20 episodes)')
    ax.set_title('FQI convergence: batch Bellman iteration on fixed dataset')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(figs_dir, "q_fqi_convergence.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print()
    print(f"  Q     → {Q_path}")
    print(f"  曲线  → {plot_path}")
    print(f"\n  最终 greedy eval mean = {eval_hist[-1]:.1f}")
    reached = [it for it, r in zip(iter_hist, eval_hist) if r >= 499]
    print(f"  最早达到 500 的 iteration: {reached[0] if reached else 'never'}")

    env_collect.close()
    env_eval.close()


if __name__ == "__main__":
    main()
