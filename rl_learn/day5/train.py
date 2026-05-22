"""day3 / 08: Step 5 —— 训练循环 + ε 衰减.

(包含 Step 1+2+3+4 的代码 inline)

把前面所有积木装在一起：
    for ep in range(N_EPISODES):
        ε = schedule(ep)                          # 线性衰减
        traj, Gs = run_episode(env, Q, ε, ...)    # 用当前 Q 采样
        mc_update_every_visit(Q, N, traj, Gs)     # 灌进 Q

输出：
    1. 学习曲线图（G_0 vs episode + ε schedule）
    2. 覆盖率图
    3. trained_Q.npy （Step 6 评估用）
"""
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ====================================================================
# 来自 Step 1+2+3+4
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
ALPHA       = 0.05    # None = 用 1/N(s,a) 自适应步长 (RM 风格, 模仿 MC); 否则常数 α


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


def init_QN():
    Q = np.zeros((N_BOXES, N_ACTIONS))
    N = np.zeros((N_BOXES, N_ACTIONS), dtype=int)
    return Q, N


def epsilon_greedy(Q, box, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(Q[box]))


def run_episode_sarsa(env, Q, N, eps, alpha, gamma, rng, seed=None):
    obs, _ = env.reset(seed=seed)
    box = discretize(obs)  
    a   = epsilon_greedy(Q, box, eps, rng)
    G_0, discount = 0.0, 1.0


    for t in range(1000):
        obs, r, term, trunc, _ = env.step(a)
        next_box = discretize(obs)
        done = term or trunc
        G_0 += discount * r
        discount *= gamma
        if done:
            target = r    
        else:
            next_a = epsilon_greedy(Q, next_box, eps, rng)   
            target = r + gamma * Q[next_box, next_a]
        N[box, a] += 1
        step = (1.0 / N[box, a]) if alpha is None else alpha
        Q[box, a] += step * (target - Q[box, a])
        if done:
             break
        box, a = next_box, next_a
    return G_0




# ====================================================================
# Step 5 新增：ε schedule + 训练循环
# ====================================================================
def epsilon_schedule(ep, eps_start, eps_min, decay_episodes):
    """线性衰减：前 decay_episodes 个 episode 内从 eps_start → eps_min。"""
    if ep >= decay_episodes:
        return eps_min
    return eps_start - (eps_start - eps_min) * (ep / decay_episodes)


def train(env, n_episodes, gamma, eps_start, eps_min, decay_episodes, rng):
    """完整训练循环。返回 Q, N 和各种 history。"""
    Q, N = init_QN()
    G_history        = np.zeros(n_episodes)
    eps_history      = np.zeros(n_episodes)
    coverage_history = []     # [(ep, n_boxes_with_both_actions), ...]

    for ep in range(n_episodes):
        eps = epsilon_schedule(ep, eps_start, eps_min, decay_episodes)
        G_0 = run_episode_sarsa(env, Q, N, eps=eps, alpha=ALPHA, gamma=gamma, rng=rng, seed=ep)

        G_history[ep]   = G_0
        eps_history[ep] = eps

        if ep % 100 == 0 or ep == n_episodes - 1:
            both_visited = ((N[:, 0] > 0) & (N[:, 1] > 0)).sum()
            coverage_history.append((ep, both_visited))

    return Q, N, G_history, eps_history, coverage_history


# ====================================================================
# 可视化
# ====================================================================
def plot_training(G_history, eps_history, coverage_history, savepath,
                  hand_crafted=99.34, random_baseline=19.85):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    n = len(G_history)
    eps_arr = np.arange(n)

    # ---- 左上：学习曲线 ----
    ax = axes[0, 0]
    ax.scatter(eps_arr, G_history, alpha=0.15, s=3, c='steelblue', label='raw G_0')
    # rolling mean
    window = max(100, n // 50)
    smooth = np.convolve(G_history, np.ones(window) / window, mode='valid')
    ax.plot(np.arange(window - 1, n), smooth, color='navy', lw=2,
            label=f'rolling mean (w={window})')
    ax.axhline(hand_crafted, color='green', linestyle='--', lw=1.5,
               label=f'hand_crafted ({hand_crafted})')
    ax.axhline(random_baseline, color='red', linestyle=':', lw=1.5,
               label=f'random ({random_baseline})')
    ax.set_xlabel('episode')
    ax.set_ylabel('G_0')
    ax.set_title('Learning curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # ---- 右上：ε schedule ----
    ax = axes[0, 1]
    ax.plot(eps_arr, eps_history, color='orange', lw=2)
    ax.set_xlabel('episode')
    ax.set_ylabel('ε')
    ax.set_title('ε schedule')
    ax.grid(alpha=0.3)

    # ---- 左下：覆盖率 ----
    ax = axes[1, 0]
    eps_cov, cov = zip(*coverage_history)
    ax.plot(eps_cov, cov, color='purple', lw=2)
    ax.axhline(N_BOXES, color='gray', linestyle='--', label=f'all {N_BOXES} boxes')
    ax.set_xlabel('episode')
    ax.set_ylabel('boxes with both actions visited')
    ax.set_title('Coverage growth')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # ---- 右下：早/中/晚 G_0 分布对比 ----
    ax = axes[1, 1]
    n3 = n // 3
    early = G_history[:n3]
    mid   = G_history[n3:2*n3]
    late  = G_history[2*n3:]
    ax.hist([early, mid, late], bins=30,
            label=[f'early (1-{n3})', f'mid ({n3+1}-{2*n3})', f'late ({2*n3+1}-{n})'],
            stacked=False, alpha=0.7)
    ax.axvline(hand_crafted, color='green', linestyle='--', lw=1.5)
    ax.set_xlabel('G_0')
    ax.set_ylabel('count')
    ax.set_title('Return distribution by phase')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ====================================================================
# 主流程
# ====================================================================
def main():
    rng = np.random.default_rng(seed=42)
    env = gym.make("CartPole-v1")

    # 超参
    GAMMA            = 0.99
    N_EPISODES       = 80000
    EPS_START        = 1.0
    EPS_MIN          = 0.05
    DECAY_EPISODES   = 4000

    alpha_desc = "1/N(s,a) 自适应 (RM 风格)" if ALPHA is None else f"常数 {ALPHA}"
    print(f"=== Sarsa ε-greedy 训练 CartPole ===")
    print(f"  N_EPISODES       = {N_EPISODES}")
    print(f"  γ                = {GAMMA}")
    print(f"  α schedule       = {alpha_desc}")
    print(f"  ε schedule       = linear {EPS_START} → {EPS_MIN} over {DECAY_EPISODES} ep")
    print(f"  N_BOXES          = {N_BOXES}")
    print()

    Q, N, G_hist, eps_hist, cov_hist = train(
        env, N_EPISODES, GAMMA, EPS_START, EPS_MIN, DECAY_EPISODES, rng
    )

    # 训练摘要
    n = len(G_hist)
    n5 = n // 5
    print("=== 训练后摘要 ===")
    for label, slc in [("第 1 个 1/5", slice(0, n5)),
                       ("第 2 个 1/5", slice(n5, 2*n5)),
                       ("第 3 个 1/5", slice(2*n5, 3*n5)),
                       ("第 4 个 1/5", slice(3*n5, 4*n5)),
                       ("第 5 个 1/5", slice(4*n5, n))]:
        print(f"  {label}: G_0 mean = {G_hist[slc].mean():>7.2f}, "
              f"max = {G_hist[slc].max():>6.1f}")
    print(f"\n  最后 100 ep: mean = {G_hist[-100:].mean():>7.2f}, "
          f"max = {G_hist[-100:].max():>6.1f}")

    visited      = (N.sum(axis=1) > 0).sum()
    both_visited = ((N[:, 0] > 0) & (N[:, 1] > 0)).sum()
    print(f"\n  最终覆盖: 任一动作访问过 {visited}/{N_BOXES} ({100*visited/N_BOXES:.1f}%)")
    print(f"            两动作都访问过 {both_visited}/{N_BOXES} ({100*both_visited/N_BOXES:.1f}%)")
    print(f"            总更新次数 N.sum() = {N.sum()}")

    # 保存
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    tag = "invN" if ALPHA is None else f"a{ALPHA}"
    plot_path = os.path.join(figs_dir, f"sarsa_{tag}_training.png")
    plot_training(G_hist, eps_hist, cov_hist, plot_path)
    print(f"\n  学习曲线 → {plot_path}")

    Q_path = os.path.join(os.path.dirname(__file__), f"trained_Q_sarsa_{tag}.npy")
    np.save(Q_path, Q)
    N_path = os.path.join(os.path.dirname(__file__), f"trained_N_sarsa_{tag}.npy")
    np.save(N_path, N)
    print(f"  Q & N    → {Q_path}, {N_path}")

    env.close()


if __name__ == "__main__":
    main()
