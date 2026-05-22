"""day3 / 10: Step 6 —— 评估 trained policy.

两件事：
  1. 跑 100 个 eval episode，对比 trained / hand_crafted / random 的 return 分布
  2. 可视化 argmax(Q) 投影到 (θ, θ_dot) 平面，看 RL 自己发现的决策边界
     是否长得像 hand_crafted 的 PD 直线 θ + 0.5·θ_dot = 0
"""
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ====================================================================
# 离散化（必须和训练脚本一致）
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


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


# ====================================================================
# 三个 policy
# ====================================================================
def make_trained_policy(Q):
    def pi(obs):
        return int(np.argmax(Q[discretize(obs)]))
    return pi


def hand_crafted_policy(obs):
    _, _, theta, theta_dot = obs
    return 1 if (1.0 * theta + 0.5 * theta_dot) > 0 else 0


def make_random_policy(rng):
    def pi(obs):
        return int(rng.integers(0, 2))
    return pi


# ====================================================================
# 评估：跑 N 个 episode 拿 return 列表
# ====================================================================
def evaluate(env, policy_fn, n_episodes, seed_base=10000):
    """跑 n_episodes 个 episode，每个用不同 seed。返回 return 数组。"""
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total, length = 0.0, 0
        for t in range(1000):
            a = policy_fn(obs)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            length += 1
            if term or trunc:
                break
        returns[ep] = total
        lengths[ep] = length
    return returns, lengths


# ====================================================================
# 可视化 1：return 分布
# ====================================================================
def plot_return_dist(results, savepath):
    """results: dict[name → returns array]."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, data = list(results.keys()), list(results.values())
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colors = ["#FFC880", "#7EB6FF", "#90EE90"]
    for patch, c in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(c)
    for i, (name, returns) in enumerate(results.items()):
        ax.text(i + 1, returns.mean() + 5, f"μ={returns.mean():.1f}\nσ={returns.std():.1f}",
                ha="center", fontsize=10)
    ax.set_ylabel("episode return (CartPole-v1, max=500)")
    ax.set_title("Return distribution across 100 seeds")
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ====================================================================
# 可视化 2：策略边界投影到 (θ, θ_dot) 切片
# ====================================================================
def plot_policy_slice(Q, savepath):
    """对固定 (i_x=1, i_x_dot=1)，画 θ × θ_dot 网格上 RL 学到的 argmax 和
       hand_crafted 的决策，并行对比。"""
    # 用 bin 中心点做"代表性 obs"，但其实 argmax 只看 box index
    # 所以直接遍历所有 (i_theta, i_theta_dot) 组合即可
    rl_action   = np.zeros((N_THETA_DOT, N_THETA), dtype=int)
    hc_action   = np.zeros((N_THETA_DOT, N_THETA), dtype=int)

    # 取每个 bin 的中心点估"代表性 obs"，给 hand_crafted 用
    theta_centers = _bin_centers(THETA_BINS, lo=-0.21, hi=0.21)
    thdot_centers = _bin_centers(THETA_DOT_BINS, lo=-2.0,  hi=2.0)

    for i_theta in range(N_THETA):
        for i_th_dot in range(N_THETA_DOT):
            box = ((1 * N_X_DOT + 1) * N_THETA + i_theta) * N_THETA_DOT + i_th_dot
            rl_action[i_th_dot, i_theta] = np.argmax(Q[box])
            theta = theta_centers[i_theta]
            thdot = thdot_centers[i_th_dot]
            hc_action[i_th_dot, i_theta] = 1 if 1.0*theta + 0.5*thdot > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- 子图 1：RL 学到的 ---
    ax = axes[0]
    ax.imshow(rl_action, origin="lower", cmap="RdYlBu_r", aspect="auto",
              extent=[-0.5, N_THETA-0.5, -0.5, N_THETA_DOT-0.5])
    ax.set_xticks(range(N_THETA))
    ax.set_xticklabels([f"{c:+.2f}" for c in theta_centers], rotation=45)
    ax.set_yticks(range(N_THETA_DOT))
    ax.set_yticklabels([f"{c:+.2f}" for c in thdot_centers])
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ_dot (rad/s)")
    ax.set_title("RL learned: argmax(Q) on (θ, θ_dot)")
    for i in range(N_THETA_DOT):
        for j in range(N_THETA):
            ax.text(j, i, "→" if rl_action[i,j]==1 else "←",
                    ha="center", va="center", fontsize=14, color="black")

    # --- 子图 2：hand_crafted 的 ---
    ax = axes[1]
    ax.imshow(hc_action, origin="lower", cmap="RdYlBu_r", aspect="auto",
              extent=[-0.5, N_THETA-0.5, -0.5, N_THETA_DOT-0.5])
    ax.set_xticks(range(N_THETA))
    ax.set_xticklabels([f"{c:+.2f}" for c in theta_centers], rotation=45)
    ax.set_yticks(range(N_THETA_DOT))
    ax.set_yticklabels([f"{c:+.2f}" for c in thdot_centers])
    ax.set_xlabel("θ (rad)")
    ax.set_title("hand_crafted: sign(θ + 0.5·θ_dot)")
    for i in range(N_THETA_DOT):
        for j in range(N_THETA):
            ax.text(j, i, "→" if hc_action[i,j]==1 else "←",
                    ha="center", va="center", fontsize=14, color="black")

    # --- 子图 3：差异 ---
    ax = axes[2]
    diff = (rl_action != hc_action).astype(int)
    ax.imshow(diff, origin="lower", cmap="Reds", aspect="auto",
              extent=[-0.5, N_THETA-0.5, -0.5, N_THETA_DOT-0.5], vmin=0, vmax=1)
    ax.set_xticks(range(N_THETA))
    ax.set_xticklabels([f"{c:+.2f}" for c in theta_centers], rotation=45)
    ax.set_yticks(range(N_THETA_DOT))
    ax.set_yticklabels([f"{c:+.2f}" for c in thdot_centers])
    ax.set_xlabel("θ (rad)")
    ax.set_title(f"Disagree (red): {diff.sum()}/{diff.size} cells")
    for i in range(N_THETA_DOT):
        for j in range(N_THETA):
            if diff[i,j]:
                ax.text(j, i, "X", ha="center", va="center", fontsize=14, color="white")

    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return diff


def _bin_centers(bins, lo, hi):
    """给 bin 边界 [b1, ..., bK]，返回 K+1 个 bin 的中心点近似."""
    edges = np.concatenate([[lo], bins, [hi]])
    return 0.5 * (edges[:-1] + edges[1:])


# ====================================================================
# 主流程
# ====================================================================
ALGO_TAG = "sarsa_a0.05"    # 改这个切换不同算法: sarsa_a0.05 / sarsa_a0.1 / sarsa_invN / q_a0.05 ...


def main():
    Q = np.load(os.path.join(os.path.dirname(__file__), f"trained_Q_{ALGO_TAG}.npy"))
    N = np.load(os.path.join(os.path.dirname(__file__), f"trained_N_{ALGO_TAG}.npy"))

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed=42)
    N_EVAL = 100

    # ---- 1. 三 policy 评估 ----
    print(f"=== 评估 [{ALGO_TAG}]：{N_EVAL} 个 seed × 3 个 policy ===")

    pi_trained = make_trained_policy(Q)
    pi_hc      = hand_crafted_policy
    pi_random  = make_random_policy(rng)

    print("\n  跑 trained policy ...")
    r_trained, l_trained = evaluate(env, pi_trained, N_EVAL)
    print("  跑 hand_crafted ...")
    r_hc, l_hc           = evaluate(env, pi_hc,      N_EVAL)
    print("  跑 random ...")
    r_random, l_random   = evaluate(env, pi_random,  N_EVAL)

    print(f"\n{'policy':>15s}  {'mean':>8s}  {'std':>7s}  {'min':>5s}  {'max':>5s}  "
          f"{'撑满 500 比例':>14s}")
    print("  " + "-" * 64)
    for name, r, l in [("trained",      r_trained, l_trained),
                       ("hand_crafted", r_hc,      l_hc),
                       ("random",       r_random,  l_random)]:
        full = (l == 500).mean() * 100
        print(f"{name:>15s}  {r.mean():>8.2f}  {r.std():>7.2f}  "
              f"{r.min():>5.0f}  {r.max():>5.0f}  {full:>13.1f}%")

    # ---- 2. policy 边界可视化 ----
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    dist_path  = os.path.join(figs_dir, f"eval_returns_{ALGO_TAG}.png")
    slice_path = os.path.join(figs_dir, f"policy_slice_{ALGO_TAG}.png")

    plot_return_dist({"trained":      r_trained,
                      "hand_crafted": r_hc,
                      "random":       r_random}, dist_path)
    print(f"\n  return 分布图 → {dist_path}")

    diff = plot_policy_slice(Q, slice_path)
    print(f"  策略切片图   → {slice_path}")
    print(f"  RL 与 hand_crafted 决策一致率: "
          f"{(1 - diff.mean())*100:.1f}% ({diff.size - diff.sum()}/{diff.size} cells)")

    # ---- 3. 哪些 cell 不一致？打印出来分析 ----
    if diff.sum() > 0:
        print(f"\n  不一致的 (i_theta, i_theta_dot) cell：")
        n_visited_check = N[1*N_X_DOT*N_THETA*N_THETA_DOT // N_X
                            : (1+1)*N_X_DOT*N_THETA*N_THETA_DOT // N_X]
        for i_th_dot in range(N_THETA_DOT):
            for i_theta in range(N_THETA):
                if diff[i_th_dot, i_theta]:
                    box = ((1 * N_X_DOT + 1) * N_THETA + i_theta) * N_THETA_DOT + i_th_dot
                    print(f"    i_theta={i_theta}, i_theta_dot={i_th_dot}, "
                          f"N[box]={N[box].tolist()}, Q[box]={Q[box].round(3).tolist()}")

    env.close()


if __name__ == "__main__":
    main()
