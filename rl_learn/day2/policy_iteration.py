"""Policy Iteration —— 解 BOE 的另一条路。

外层循环：
  1) Policy Evaluation: 给定 π_k，闭式解 v_{π_k}
  2) Policy Improvement: π_{k+1}(s) = argmax_a [r(s,a) + γ Σ p(s'|s,a) v_{π_k}(s')]
  直到 π_{k+1} == π_k。

PIT (Policy Improvement Theorem) 保证 v_{π_{k+1}} ≥ v_{π_k}，且有限 MDP 下有限步收敛。
"""
import os
import numpy as np

from grid_world_policy_eval import (
    GRID_H, GRID_W, GAMMA, N_S, N_A, step, closed_form,
)


def policy_iteration(max_outer=100, init="uniform"):
    # 初始策略
    if init == "uniform":
        pi = np.ones((N_S, N_A)) / N_A
    else:  # 任意 deterministic
        pi = np.zeros((N_S, N_A))
        pi[:, 0] = 1.0  # 全部选"上"

    # 环境查表
    next_s = np.zeros((N_S, N_A), dtype=int)
    rew    = np.zeros((N_S, N_A))
    for s in range(N_S):
        for a in range(N_A):
            ns, r = step(s, a)
            next_s[s, a] = ns
            rew[s, a]    = r

    for k in range(max_outer):
        # Step 1: Policy Evaluation —— 闭式解 v_π
        v = closed_form(pi)

        # Step 2: Policy Improvement —— argmax
        Q = rew + GAMMA * v[next_s]
        a_best = Q.argmax(axis=1)
        pi_new = np.zeros((N_S, N_A))
        pi_new[np.arange(N_S), a_best] = 1.0

        # 策略不变 → 收敛
        if np.array_equal(pi, pi_new):
            return v, pi_new, k + 1
        pi = pi_new

    return v, pi, max_outer


def truncated_policy_iteration(inner_iters=5, tol=1e-8, max_outer=1000):
    """中间形式：policy eval 只做有限步迭代（不闭式解到底）。
    inner_iters=1 时退化为 value iteration；inner_iters=∞ 退化为 policy iteration。
    """
    pi = np.ones((N_S, N_A)) / N_A
    v  = np.zeros(N_S)

    next_s = np.zeros((N_S, N_A), dtype=int)
    rew    = np.zeros((N_S, N_A))
    for s in range(N_S):
        for a in range(N_A):
            ns, r = step(s, a)
            next_s[s, a] = ns
            rew[s, a]    = r

    for k in range(max_outer):
        # 用 P_π / r_π 做 inner_iters 步内层迭代
        P = np.zeros((N_S, N_S)); r_pi = np.zeros(N_S)
        for s in range(N_S):
            for a in range(N_A):
                P[s, next_s[s, a]] += pi[s, a]
                r_pi[s] += pi[s, a] * rew[s, a]
        for _ in range(inner_iters):
            v = r_pi + GAMMA * P @ v

        # improvement
        Q = rew + GAMMA * v[next_s]
        pi_new = np.zeros((N_S, N_A))
        pi_new[np.arange(N_S), Q.argmax(axis=1)] = 1.0
        if np.array_equal(pi, pi_new):
            return v, pi_new, k + 1
        pi = pi_new

    return v, pi, max_outer


if __name__ == "__main__":
    v_pi, pi_star_pi, n_outer = policy_iteration()
    print(f"PI 外层迭代次数: {n_outer}")
    print("\nv* (from PI):")
    print(v_pi.reshape(GRID_H, GRID_W).round(2))

    # 与 VI 对比
    from value_iteration import value_iteration
    v_vi, pi_star_vi, n_vi = value_iteration()
    print(f"\nVI 迭代次数: {n_vi}")
    print(f"v(PI) vs v(VI) 最大差: {np.max(np.abs(v_pi - v_vi)):.2e}")
    print(f"两个 π* 是否相同: {np.array_equal(pi_star_pi, pi_star_vi)}")

    # truncated PI 扫描：内层步数 vs 外层迭代数
    print("\nTruncated PI scan (inner iters → outer iters):")
    for K in [1, 3, 5, 10, 50]:
        _, _, n = truncated_policy_iteration(inner_iters=K)
        print(f"  inner={K:>3} → outer={n}")

    # 画图
    from viz import plot_policy, plot_values
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    plot_policy(pi_star_pi, "pi*: from Policy Iteration", os.path.join(figs_dir, "pi_star_pi.png"))
    plot_values(v_pi,       "v*: from Policy Iteration",  os.path.join(figs_dir, "v_star_pi.png"))
    print(f"\n图保存到: {figs_dir}/")
