"""Value Iteration —— 解 Bellman 最优方程 (BOE) 找 π*。

迭代式：
    v_{k+1}(s) = max_a [ r(s,a) + γ Σ_{s'} p(s'|s,a) v_k(s') ]

收敛后：
    π*(s) = argmax_a [ r(s,a) + γ Σ_{s'} p(s'|s,a) v*(s') ]
"""
import os
import numpy as np

from grid_world_policy_eval import (
    GRID_H, GRID_W, GAMMA, N_S, N_A, step, closed_form,
)


def value_iteration(gamma=GAMMA, tol=1e-8, max_iter=10000):
    # 确定性环境：把 (s, a) → (s', r) 缓存成查表，VI 全过程靠它
    next_s = np.zeros((N_S, N_A), dtype=int)
    rew    = np.zeros((N_S, N_A))
    for s in range(N_S):
        for a in range(N_A):
            ns, r = step(s, a)
            next_s[s, a] = ns
            rew[s, a]    = r

    v = np.zeros(N_S)
    for k in range(max_iter):
        # Q[s, a] = r(s,a) + γ v(next_s(s,a))
        Q = rew + gamma * v[next_s]
        v_new = Q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new

    # argmax 提取最优确定性策略
    pi_star = np.zeros((N_S, N_A))
    pi_star[np.arange(N_S), Q.argmax(axis=1)] = 1.0
    return v, pi_star, k + 1


if __name__ == "__main__":
    v_star, pi_star, k = value_iteration()
    print(f"VI 收敛步数: {k}")

    print("\nv* (optimal):")
    print(v_star.reshape(GRID_H, GRID_W).round(2))

    # 自检：用 policy evaluation 跑 π*，应该得到同一个 v*
    v_check = closed_form(pi_star)
    print(f"\nv* vs policy_eval(π*) 最大差: {np.max(np.abs(v_star - v_check)):.2e}")

    from viz import plot_policy, plot_values
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    plot_policy(pi_star, "pi*: optimal", os.path.join(figs_dir, "pi_star.png"))
    plot_values(v_star,  "v*: optimal",  os.path.join(figs_dir, "v_star.png"))
    print(f"\n图保存到: {figs_dir}/")
