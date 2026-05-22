"""
赵世钰 Ch.2 Bellman 方程 —— Policy Evaluation 实践

已知策略 π，求 v_π。两种解法互相验证：
  1) 闭式：v_π = (I - γP_π)^{-1} r_π
  2) 迭代：v_{k+1} = r_π + γ P_π v_k   （Banach contraction，γ-shrink）
"""
import os
import numpy as np

GRID_H, GRID_W = 5, 5
TARGET = (3, 2)
FORBIDDEN = [(1, 1), (2, 1)]
GAMMA = 0.9

# 上, 右, 下, 左, 不动
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
N_S = GRID_H * GRID_W
N_A = len(ACTIONS)


def s2rc(s):
    return (s // GRID_W, s % GRID_W)


def rc2s(r, c):
    return r * GRID_W + c


def step(s, a):
    """确定性转移：返回 (s', r)。撞墙原地停留 + r=-1。"""
    r, c = s2rc(s)
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
        return s, -1
    s_next = rc2s(nr, nc)
    if (nr, nc) == TARGET:
        return s_next, 1
    if (nr, nc) in FORBIDDEN:
        return s_next, -10
    return s_next, 0


def build_P_r(pi):
    """给定 π(a|s)，构建 P_π (N_S×N_S) 和 r_π (N_S)。"""
    P = np.zeros((N_S, N_S))
    r = np.zeros(N_S)
    for s in range(N_S):
        for a in range(N_A):
            s_next, rew = step(s, a)
            P[s, s_next] += pi[s, a]
            r[s] += pi[s, a] * rew
    return P, r


def closed_form(pi):
    P, r = build_P_r(pi)
    return np.linalg.solve(np.eye(N_S) - GAMMA * P, r)


def iterative(pi, tol=1e-8, max_iter=10000):
    P, r = build_P_r(pi)
    v = np.zeros(N_S)
    for k in range(max_iter):
        v_new = r + GAMMA * P @ v
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, k + 1
        v = v_new
    return v, max_iter


# ---------- 两个对比策略 ----------

# 策略 1：均匀随机（每个动作 0.2）
pi_uniform = np.ones((N_S, N_A)) / N_A

# 策略 2：朴素地朝 target 走（只看行列差，会撞 forbidden）
pi_naive = np.zeros((N_S, N_A))
for s in range(N_S):
    rr, cc = s2rc(s)
    if (rr, cc) == TARGET:
        pi_naive[s, 4] = 1.0
    elif rr < TARGET[0]:
        pi_naive[s, 2] = 1.0  # 下
    elif rr > TARGET[0]:
        pi_naive[s, 0] = 1.0  # 上
    elif cc < TARGET[1]:
        pi_naive[s, 1] = 1.0  # 右
    else:
        pi_naive[s, 3] = 1.0  # 左


if __name__ == "__main__":
    # 闭式 vs 迭代，验证两者一致
    v_uni = closed_form(pi_uniform)
    v_it, k = iterative(pi_uniform)
    print(f"闭式 vs 迭代 最大差: {np.max(np.abs(v_uni - v_it)):.2e}, 迭代步数: {k}")

    print("\nv_pi under uniform random:")
    print(v_uni.reshape(GRID_H, GRID_W).round(2))

    v_naive = closed_form(pi_naive)
    print("\nv_pi under naive target-seeking:")
    print(v_naive.reshape(GRID_H, GRID_W).round(2))

    # 画图
    from viz import plot_policy, plot_values
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    plot_policy(pi_uniform, "pi: uniform random",        os.path.join(figs_dir, "pi_uniform.png"))
    plot_values(v_uni,      "v_pi under uniform random", os.path.join(figs_dir, "v_uniform.png"))
    plot_policy(pi_naive,   "pi: naive target-seeking",  os.path.join(figs_dir, "pi_naive.png"))
    plot_values(v_naive,    "v_pi under naive",          os.path.join(figs_dir, "v_naive.png"))
    print(f"\n图保存到: {figs_dir}/")
