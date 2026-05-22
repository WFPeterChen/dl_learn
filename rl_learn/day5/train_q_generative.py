"""day5 / Q-learning + generative model + Gaussian state sampling.

不 follow trajectory, 而是每次任意 reset env 到高斯采样的状态, 走 1 步,
用 Q-learning update 这个单步 transition (s, a, r, s').

对应 generative model setting (Kearns & Singh 2002 / Azar et al. 2013).

用法:
    python train_q_generative.py
输出:
    trained_Q_q_generative.npy, trained_N_q_generative.npy
    figs/q_generative_sample_efficiency.png
"""
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ====================================================================
# 离散化 (和其他脚本一致)
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


# ====================================================================
# 高斯状态采样
# ====================================================================
def sample_state(rng, mu, sigma):
    """高斯采样 state, 截断到 env 合法范围 (避免 reset 就 terminate)."""
    s = rng.normal(mu, sigma)
    # CartPole-v1 terminate 条件: |x| > 2.4 or |θ| > 0.21
    s[0] = float(np.clip(s[0], -2.0,  2.0))    # 留 margin
    s[2] = float(np.clip(s[2], -0.18, 0.18))   # 留 margin (远离 |θ|=0.21 边界)
    return s.astype(np.float32)


# ====================================================================
# Greedy evaluation (用于 checkpoint)
# ====================================================================
def evaluate_greedy(env, Q, n_episodes=20, seed_base=10000):
    """从 env.reset() 开始的标准 greedy eval, 取 mean return."""
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
# 主流程
# ====================================================================
def main():
    rng = np.random.default_rng(seed=42)
    env_train = gym.make("CartPole-v1")
    env_eval  = gym.make("CartPole-v1")
    env_train.reset(seed=0)   # 必须先 reset 一次让内部 state 初始化

    # 超参
    N_TOTAL          = 50000      # 总 transitions (比 uniform Q-learning 的 80k episodes × 22 step ≈ 1.8M 少很多)
    CHECKPOINT_EVERY = 1000
    EVAL_EPISODES    = 20
    ALPHA            = 0.05
    GAMMA            = 0.99

    # 高斯采样参数 (以最优状态 (0,0,0,0) 为中心)
    MU    = np.array([0.0, 0.0, 0.0,  0.0])
    SIGMA = np.array([0.5, 0.5, 0.05, 0.5])    # θ 维窄 (因为合法范围 ±0.21 小)

    print(f"=== Q-learning + Generative Model + Gaussian state sampling ===")
    print(f"  N_TOTAL          = {N_TOTAL}")
    print(f"  γ                = {GAMMA}, α = {ALPHA}")
    print(f"  state sampling   = N(μ={MU.tolist()}, σ={SIGMA.tolist()})")
    print(f"  checkpoint every = {CHECKPOINT_EVERY} transitions")
    print(f"  eval episodes    = {EVAL_EPISODES} per checkpoint")
    print(f"  N_BOXES          = {N_BOXES}")
    print()

    Q = np.zeros((N_BOXES, N_ACTIONS))
    N = np.zeros((N_BOXES, N_ACTIONS), dtype=int)

    eval_steps   = []
    eval_returns = []

    for step in range(1, N_TOTAL + 1):
        # 高斯采样目标 state
        s = sample_state(rng, MU, SIGMA)
        s_box = discretize(s)

        # 重置 env 并强行覆盖 state
        env_train.reset()
        env_train.unwrapped.state = s        # gymnasium CartPole 直接接受 np.array

        # 均匀采样 action
        a = int(rng.integers(0, N_ACTIONS))

        # 单步 transition
        obs, r, term, trunc, _ = env_train.step(a)
        s_next_box = discretize(obs)
        done = term       # trunc 单步后不可能 (counter 刚清零)

        # Q-learning update
        target = r if done else r + GAMMA * Q[s_next_box].max()
        N[s_box, a] += 1
        Q[s_box, a] += ALPHA * (target - Q[s_box, a])

        # Checkpoint
        if step % CHECKPOINT_EVERY == 0:
            mean_ret = evaluate_greedy(env_eval, Q, EVAL_EPISODES)
            eval_steps.append(step)
            eval_returns.append(mean_ret)
            both_visited = ((N[:, 0] > 0) & (N[:, 1] > 0)).sum()
            print(f"  step {step:>6d}: greedy eval mean = {mean_ret:>6.1f}  "
                  f"(both_visited = {both_visited}/{N_BOXES})")

    # ---- 保存 Q / N ----
    base = os.path.dirname(__file__)
    Q_path = os.path.join(base, "trained_Q_q_generative.npy")
    N_path = os.path.join(base, "trained_N_q_generative.npy")
    np.save(Q_path, Q)
    np.save(N_path, N)

    # ---- 画 sample efficiency 曲线 ----
    figs_dir = os.path.join(base, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_steps, eval_returns, marker='o', lw=2, color='steelblue',
            label='Q-learning generative (Gaussian state sampling)')
    ax.axhline(500, color='green',  ls='--', lw=1.5,
               label='MC / Q-uniform / hand-crafted upper bound (500)')
    ax.axhline(271, color='orange', ls=':',  lw=1.5,
               label='Sarsa ε-greedy 80k ep (271)')
    ax.axhline(137, color='red',    ls=':',  lw=1.5,
               label='Q-learn ε-greedy 80k ep (137)')
    ax.set_xlabel('transitions')
    ax.set_ylabel('greedy eval mean return (20 episodes)')
    ax.set_title('Sample efficiency: generative model + Gaussian state sampling')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(figs_dir, "q_generative_sample_efficiency.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print()
    print(f"  Q & N    → {Q_path}, {N_path}")
    print(f"  曲线     → {plot_path}")
    print(f"\n  最终 greedy eval mean = {eval_returns[-1]:.1f}")
    print(f"  最早达到 500 的 checkpoint: ", end="")
    reached = [s for s, r in zip(eval_steps, eval_returns) if r >= 499]
    print(f"{reached[0] if reached else 'never'}")

    env_train.close()
    env_eval.close()


if __name__ == "__main__":
    main()
