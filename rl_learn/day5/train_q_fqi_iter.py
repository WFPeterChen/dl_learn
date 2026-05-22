"""day5 / Iterated FQI: 混合 Gaussian + trajectory data, 解决 distribution mismatch.

外循环 k = 0, 1, ..., K_OUTER-1:
  outer 0: 用初始 Gaussian buffer 跑 FQI K_INNER 轮
  outer k>0: 用当前 Q 跑 ε-greedy trajectory 收集新 data 加入 buffer, 再跑 FQI

对比 train_q_fqi_nail.py (单纯加细 + reward shape):
  关键加强: trajectory data 让训练分布 → 部署分布对齐 (解决 covariate shift)
"""
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ====================================================================
# 离散化 (跟 nail 版一致)
# ====================================================================
X_BINS         = np.array([-1.0, -0.3, 0.3, 1.0])
X_DOT_BINS     = np.array([-1.0, -0.3, 0.3, 1.0])
THETA_BINS     = np.array([-0.15, -0.08, -0.04, -0.02, -0.005,
                            0.005, 0.02, 0.04, 0.08, 0.15])
THETA_DOT_BINS = np.array([-1.5, -0.8, -0.4, -0.15, -0.05,
                            0.05, 0.15, 0.4, 0.8, 1.5])
N_X         = len(X_BINS)         + 1
N_X_DOT     = len(X_DOT_BINS)     + 1
N_THETA     = len(THETA_BINS)     + 1
N_THETA_DOT = len(THETA_DOT_BINS) + 1
N_BOXES     = N_X * N_X_DOT * N_THETA * N_THETA_DOT
N_ACTIONS   = 2
MAX_STEPS   = 1000


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


def shaped_reward(obs, term):
    if term:
        return 0.0
    x, x_dot, theta, theta_dot = obs
    r_position = max(0.0, 1.0 - (x / 2.4) ** 2)
    r_static   = max(0.0, 1.0 - 0.5 * (x_dot / 3.0) ** 2 - 0.5 * (theta_dot / 2.0) ** 2)
    return 0.5 * r_position + 0.5 * r_static


def sample_state(rng, mu, sigma):
    s = rng.normal(mu, sigma)
    s[0] = float(np.clip(s[0], -2.3,  2.3))
    s[2] = float(np.clip(s[2], -0.22, 0.22))
    return s.astype(np.float32)


# ====================================================================
# Data collection: Gaussian / Trajectory
# ====================================================================
def collect_gaussian(env, rng, n_trans, mu, sigma):
    s_box  = np.zeros(n_trans, dtype=np.int64)
    act    = np.zeros(n_trans, dtype=np.int64)
    rew    = np.zeros(n_trans, dtype=np.float64)
    sn_box = np.zeros(n_trans, dtype=np.int64)
    done   = np.zeros(n_trans, dtype=bool)
    env.reset(seed=0)
    for i in range(n_trans):
        s = sample_state(rng, mu, sigma)
        env.reset()
        env.unwrapped.state = s
        a = int(rng.integers(0, N_ACTIONS))
        obs, _, term, trunc, _ = env.step(a)
        s_box[i]  = discretize(s)
        act[i]    = a
        rew[i]    = shaped_reward(obs, term)
        sn_box[i] = discretize(obs)
        done[i]   = term
    return s_box, act, rew, sn_box, done


def collect_trajectory(env, Q, rng, n_trans, epsilon=0.1):
    """用 current Q 的 ε-greedy policy 跑 trajectories, 收集 n_trans transitions."""
    s_box  = np.zeros(n_trans, dtype=np.int64)
    act    = np.zeros(n_trans, dtype=np.int64)
    rew    = np.zeros(n_trans, dtype=np.float64)
    sn_box = np.zeros(n_trans, dtype=np.int64)
    done   = np.zeros(n_trans, dtype=bool)

    obs, _ = env.reset()
    i = 0
    while i < n_trans:
        s_b = discretize(obs)
        if rng.random() < epsilon:
            a = int(rng.integers(0, N_ACTIONS))
        else:
            a = int(np.argmax(Q[s_b]))
        next_obs, _, term, trunc, _ = env.step(a)
        s_box[i]  = s_b
        act[i]    = a
        rew[i]    = shaped_reward(next_obs, term)
        sn_box[i] = discretize(next_obs)
        done[i]   = term
        i += 1
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return s_box, act, rew, sn_box, done


# ====================================================================
# FQI iteration + evaluate (跟之前一致)
# ====================================================================
def fqi_iterate(Q, s_box, act, rew, sn_box, done, gamma):
    next_max = Q[sn_box].max(axis=1)
    targets  = rew + gamma * next_max * (~done)
    sums   = np.zeros_like(Q)
    counts = np.zeros_like(Q)
    np.add.at(sums,   (s_box, act), targets)
    np.add.at(counts, (s_box, act), 1.0)
    Q_new = Q.copy()
    mask = counts > 0
    Q_new[mask] = sums[mask] / counts[mask]
    return Q_new


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
# 主流程
# ====================================================================
def main():
    rng = np.random.default_rng(seed=42)
    env_collect = gym.make("CartPole-v1", max_episode_steps=MAX_STEPS)
    env_eval    = gym.make("CartPole-v1", max_episode_steps=MAX_STEPS)

    N_INITIAL    = 100000
    N_TRAJ       = 100000
    K_OUTER      = 3      # 3 = outer 0/1/2 (最终 Q = outer 2 的, 撑满 1000 的版本)
    K_INNER      = 100
    EVAL_EPISODES = 20
    GAMMA        = 0.99
    EPSILON_TRAJ = 0.1     # trajectory collection 时 ε-greedy 的 ε

    MU    = np.array([0.0, 0.0, 0.0,  0.0])
    SIGMA = np.array([1.0, 1.0, 0.15, 1.0])    # σ 增大覆盖更广

    print(f"=== Iterated FQI: Gaussian + trajectory mixed buffer ===")
    print(f"  N_BOXES          = {N_BOXES}")
    print(f"  MAX_STEPS        = {MAX_STEPS} (20s)")
    print(f"  K_OUTER × K_INNER = {K_OUTER} × {K_INNER} = {K_OUTER*K_INNER} 总 FQI iters")
    print(f"  N_INITIAL Gaussian = {N_INITIAL}")
    print(f"  N_TRAJ per outer  = {N_TRAJ}")
    print(f"  σ = {SIGMA.tolist()}")
    print(f"  γ = {GAMMA},  ε_traj = {EPSILON_TRAJ}")
    print()

    # ---- 初始 Gaussian buffer ----
    print(f"[init] collecting {N_INITIAL} Gaussian transitions ...")
    s_buf, a_buf, r_buf, sn_buf, d_buf = collect_gaussian(env_collect, rng, N_INITIAL, MU, SIGMA)
    unique_sa = len(set(zip(s_buf.tolist(), a_buf.tolist())))
    print(f"        unique (s, a): {unique_sa}/{N_BOXES*N_ACTIONS} "
          f"({100*unique_sa/(N_BOXES*N_ACTIONS):.1f}%)")
    print(f"        terminal:      {d_buf.sum()}/{N_INITIAL}")
    print()

    # ---- 外循环 ----
    Q = np.zeros((N_BOXES, N_ACTIONS))
    history = []   # (outer, buffer_size, return_mean, return_std, ep_len)

    for outer in range(K_OUTER):
        # 从 outer 1 开始, 用 current Q 收集 trajectory 数据加入 buffer
        if outer > 0:
            print(f"[outer {outer}] collecting {N_TRAJ} trajectory transitions (ε={EPSILON_TRAJ}) ...")
            s_t, a_t, r_t, sn_t, d_t = collect_trajectory(env_collect, Q, rng, N_TRAJ, EPSILON_TRAJ)
            s_buf  = np.concatenate([s_buf,  s_t])
            a_buf  = np.concatenate([a_buf,  a_t])
            r_buf  = np.concatenate([r_buf,  r_t])
            sn_buf = np.concatenate([sn_buf, sn_t])
            d_buf  = np.concatenate([d_buf,  d_t])
            unique_sa = len(set(zip(s_buf.tolist(), a_buf.tolist())))
            print(f"             buffer size: {len(s_buf)}  unique (s, a): {unique_sa} "
                  f"({100*unique_sa/(N_BOXES*N_ACTIONS):.1f}%)  terminal in buf: {d_buf.sum()}")

        # FQI 内循环
        print(f"[outer {outer}] FQI {K_INNER} iterations ...")
        for k_inner in range(1, K_INNER + 1):
            Q_new = fqi_iterate(Q, s_buf, a_buf, r_buf, sn_buf, d_buf, GAMMA)
            Q = Q_new

        # 评估
        m, s, l = evaluate_greedy(env_eval, Q, EVAL_EPISODES)
        history.append((outer, len(s_buf), m, s, l))
        print(f"             eval: return = {m:>7.2f} ± {s:>6.2f},  ep_len = {l:>6.1f} / {MAX_STEPS}")
        print()

    # ---- 保存 ----
    base = os.path.dirname(__file__)
    Q_path = os.path.join(base, "trained_Q_q_fqi_iter.npy")
    np.save(Q_path, Q)

    # ---- 出图 ----
    figs_dir = os.path.join(base, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    outers, buf_sizes, mns, sds, lens = zip(*history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(outers, mns, yerr=sds, marker='o', lw=2, color='purple', capsize=5,
                 label='Iterated FQI')
    ax1.axhline(1000, color='green', ls='--', lw=1.5, label='max (1000)')
    ax1.axhline(441, color='orange', ls=':', lw=1.5, label='nail (no traj) final')
    ax1.set_xlabel('outer iteration')
    ax1.set_ylabel('eval return (mean ± std)')
    ax1.set_title('Iterated FQI return convergence')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)

    ax2.plot(outers, lens, marker='o', lw=2, color='navy', label='ep_len')
    ax2.axhline(MAX_STEPS, color='green', ls='--', lw=1.5, label='max (1000)')
    ax2.axhline(518, color='orange', ls=':', lw=1.5, label='nail (no traj) final')
    ax2.set_xlabel('outer iteration')
    ax2.set_ylabel('mean episode length')
    ax2.set_title('Episode length convergence')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(figs_dir, "q_fqi_iter_convergence.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"  Q     → {Q_path}")
    print(f"  曲线  → {plot_path}")
    print(f"\n  最终 return = {mns[-1]:.2f} (vs nail 单 Gaussian: 441.44)")
    print(f"  最终 ep_len = {lens[-1]:.1f}  (vs nail: 518.6)")

    env_collect.close()
    env_eval.close()


if __name__ == "__main__":
    main()
