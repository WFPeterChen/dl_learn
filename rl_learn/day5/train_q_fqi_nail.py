"""day5 / Q-FQI nail-it: task difficulty 升级.

目标三件事:
  ① reward shaping: 居中 + 静止 (vs 默认 "活着就 +1")
  ② max episode = 1000 steps = 20 秒 (vs 默认 500 = 10 秒)
  ③ 细化离散化: 3025 boxes (vs 324), 重点加细 θ / θ_dot 中心区

依然是 FQI: phase 1 generative + Gaussian sampling, phase 2 batch Bellman iter.
"""
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ====================================================================
# 细化离散化: 5 × 5 × 11 × 11 = 3025 boxes
# ====================================================================
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
N_BOXES     = N_X * N_X_DOT * N_THETA * N_THETA_DOT  # 3025
N_ACTIONS   = 2

# Episode 上限 (20s @ 50Hz)
MAX_STEPS = 1000


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


# ====================================================================
# Reward shaping
# ====================================================================
def shaped_reward(obs, term):
    """杆倒 → 0; 否则 = 0.5 r_position + 0.5 r_static, ∈ [0, 1]."""
    if term:
        return 0.0
    x, x_dot, theta, theta_dot = obs
    r_position = max(0.0, 1.0 - (x / 2.4) ** 2)
    r_static   = max(0.0, 1.0 - 0.5 * (x_dot / 3.0) ** 2 - 0.5 * (theta_dot / 2.0) ** 2)
    return 0.5 * r_position + 0.5 * r_static


# ====================================================================
# 高斯采样
# ====================================================================
def sample_state(rng, mu, sigma):
    s = rng.normal(mu, sigma)
    s[0] = float(np.clip(s[0], -2.3,  2.3))
    s[2] = float(np.clip(s[2], -0.22, 0.22))
    return s.astype(np.float32)


# ====================================================================
# Greedy evaluation
# ====================================================================
def evaluate_greedy(env, Q, n_episodes=20, seed_base=10000):
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total = 0.0
        length = 0
        for t in range(MAX_STEPS):
            a = int(np.argmax(Q[discretize(obs)]))
            obs, _, term, trunc, _ = env.step(a)
            length += 1
            total += shaped_reward(obs, term)
            if term or trunc:
                break
        returns[ep] = total
        lengths[ep] = length
    return returns.mean(), returns.std(), lengths.mean()


# ====================================================================
# Phase 1: 收集数据集 (用 shaped reward)
# ====================================================================
def collect_dataset(env, rng, n_transitions, mu, sigma):
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
        obs, _, term, trunc, _ = env.step(a)
        s_boxes[i]      = discretize(s)
        actions[i]      = a
        rewards[i]      = shaped_reward(obs, term)   # ★ 用 shaped reward
        s_next_boxes[i] = discretize(obs)
        dones[i]        = term
    return s_boxes, actions, rewards, s_next_boxes, dones


# ====================================================================
# Phase 2: FQI iteration
# ====================================================================
def fqi_iterate(Q, s_boxes, actions, rewards, s_next_boxes, dones, gamma):
    next_max = Q[s_next_boxes].max(axis=1)
    targets  = rewards + gamma * next_max * (~dones)
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
    env_collect = gym.make("CartPole-v1", max_episode_steps=MAX_STEPS)
    env_eval    = gym.make("CartPole-v1", max_episode_steps=MAX_STEPS)

    N_TRANSITIONS = 200000
    N_ITERATIONS  = 200
    EVAL_EVERY    = 10
    EVAL_EPISODES = 20
    GAMMA         = 0.99

    MU    = np.array([0.0, 0.0, 0.0,  0.0])
    SIGMA = np.array([0.5, 0.5, 0.10, 0.5])

    print(f"=== Q-FQI nail-it: 细化离散化 + reward shaping + 20s episode ===")
    print(f"  N_BOXES          = {N_BOXES}  (5 × 5 × 11 × 11)")
    print(f"  MAX_STEPS        = {MAX_STEPS} (= 20 秒 @ 50Hz)")
    print(f"  Phase 1: collect {N_TRANSITIONS} transitions (Gaussian)")
    print(f"  Phase 2: {N_ITERATIONS} batch Bellman iterations")
    print(f"  γ = {GAMMA},  eval every {EVAL_EVERY} iter")
    print()

    # ---- Phase 1 ----
    print("Phase 1: collecting dataset ...")
    s_boxes, actions, rewards, s_next_boxes, dones = collect_dataset(
        env_collect, rng, N_TRANSITIONS, MU, SIGMA
    )
    unique_sa = len(set(zip(s_boxes.tolist(), actions.tolist())))
    print(f"  unique (s, a):       {unique_sa} / {N_BOXES * N_ACTIONS} "
          f"({100 * unique_sa / (N_BOXES * N_ACTIONS):.1f}%)")
    print(f"  terminal:            {dones.sum()} / {N_TRANSITIONS}")
    print(f"  mean reward:         {rewards.mean():.3f}")
    print(f"  reward range:        [{rewards.min():.3f}, {rewards.max():.3f}]")
    print()

    # ---- Phase 2: FQI ----
    print("Phase 2: FQI iterations ...")
    Q = np.zeros((N_BOXES, N_ACTIONS))

    iter_hist     = [0]
    eval_mean     = [0.0]
    eval_std      = [0.0]
    eval_length   = [0.0]

    for k in range(1, N_ITERATIONS + 1):
        Q_new = fqi_iterate(Q, s_boxes, actions, rewards, s_next_boxes, dones, GAMMA)
        delta = np.abs(Q_new - Q).max()
        Q = Q_new

        if k % EVAL_EVERY == 0 or k == 1:
            m, s, l = evaluate_greedy(env_eval, Q, EVAL_EPISODES)
            iter_hist.append(k)
            eval_mean.append(m)
            eval_std.append(s)
            eval_length.append(l)
            print(f"  iter {k:>3d}: ‖ΔQ‖∞={delta:>7.4f}  "
                  f"return={m:>7.2f} ± {s:>6.2f}  ep_len={l:>6.1f}")

    # ---- 保存 ----
    base = os.path.dirname(__file__)
    Q_path = os.path.join(base, "trained_Q_q_fqi_nail.npy")
    np.save(Q_path, Q)

    # ---- 出图 ----
    figs_dir = os.path.join(base, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(iter_hist, eval_mean, marker='o', lw=2, color='purple', label='shaped return')
    ax1.fill_between(iter_hist,
                     np.array(eval_mean) - np.array(eval_std),
                     np.array(eval_mean) + np.array(eval_std),
                     alpha=0.2, color='purple')
    ax1.axhline(1000, color='green', ls='--', lw=1.5, label='theoretical max (1000)')
    ax1.set_xlabel('FQI iteration')
    ax1.set_ylabel('mean shaped return (20 eval episodes)')
    ax1.set_title('Shaped return convergence')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)

    ax2.plot(iter_hist, eval_length, marker='o', lw=2, color='navy')
    ax2.axhline(MAX_STEPS, color='green', ls='--', lw=1.5,
                label=f'max episode = {MAX_STEPS} steps (20s)')
    ax2.set_xlabel('FQI iteration')
    ax2.set_ylabel('mean episode length (steps)')
    ax2.set_title('Episode length convergence')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(figs_dir, "q_fqi_nail_convergence.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print()
    print(f"  Q     → {Q_path}")
    print(f"  曲线  → {plot_path}")
    print(f"\n  最终 return  = {eval_mean[-1]:.2f} (上限 {MAX_STEPS})")
    print(f"  最终 ep_len  = {eval_length[-1]:.1f}")
    reached_full = [it for it, l in zip(iter_hist, eval_length) if l >= MAX_STEPS - 1]
    print(f"  最早撑满 20s 的 iter: {reached_full[0] if reached_full else 'never'}")

    env_collect.close()
    env_eval.close()


if __name__ == "__main__":
    main()
