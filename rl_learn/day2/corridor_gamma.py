"""1D corridor —— 真正会让 γ 改变 π* 的最简环境。

  s=0 ─── ··· ─── s=5 (+0.3) ─── ··· ─── s=10 (+1.0)

  在 s=5 每步 stay 收 +0.3；在 s=10 每步 stay 收 +1.0。
  动作 = {left, right, stay}。

  trade-off：
    - 早到小奖励：留 s=5，长期价值 0.3/(1-γ)
    - 晚到大奖励：走到 s=10，需 5 步无奖励路程，长期价值 γ^5 · 1/(1-γ)

  在 s=5 比较两者：
    stay 在 s=5：     0.3/(1-γ)
    走去 s=10：       γ^5 · 1/(1-γ)

    临界 γ：γ^5 = 0.3 → γ ≈ 0.786
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

N = 11
ACTIONS = [-1, +1, 0]   # left, right, stay
ACT_NAME = ["←", "→", "·"]
N_A = len(ACTIONS)

REWARD_SMALL = 0.3
REWARD_BIG   = 1.0
S_SMALL = 5
S_BIG   = N - 1


def step(s, a):
    s_next = max(0, min(N - 1, s + ACTIONS[a]))
    if s_next == S_BIG:
        r = REWARD_BIG
    elif s_next == S_SMALL:
        r = REWARD_SMALL
    else:
        r = 0.0
    return s_next, r


def value_iteration(gamma, tol=1e-10, max_iter=50000):
    next_s = np.zeros((N, N_A), dtype=int)
    rew    = np.zeros((N, N_A))
    for s in range(N):
        for a in range(N_A):
            ns, r = step(s, a)
            next_s[s, a] = ns
            rew[s, a]    = r

    v = np.zeros(N)
    for k in range(max_iter):
        Q = rew + gamma * v[next_s]
        v_new = Q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return v, Q.argmax(axis=1), k + 1


COLOR_SMALL = "#FFC880"   # 小金库（s=5）：浅橙
COLOR_BIG   = "#7EB6FF"   # 大金库（s=10）：蓝
COLOR_NORMAL = "white"


def draw_strip(ax, v, pi, gamma, k):
    """一行 corridor: 上半行写 v，下半行画 π 箭头/圆点。"""
    for s in range(N):
        if s == S_SMALL:    fc = COLOR_SMALL
        elif s == S_BIG:    fc = COLOR_BIG
        else:               fc = COLOR_NORMAL
        ax.add_patch(Rectangle((s, 0), 1, 1, facecolor=fc, edgecolor="gray", linewidth=0.6))
        ax.text(s + 0.5, 0.72, f"{v[s]:.2f}", ha="center", va="center", fontsize=9)

        a = pi[s]
        delta = ACTIONS[a]
        cx, cy = s + 0.5, 0.28
        if delta == 0:
            ax.plot(cx, cy, "o", color="black", markersize=7)
        else:
            L = 0.32
            ax.arrow(cx - delta * L / 2, cy, delta * L, 0,
                     head_width=0.08, head_length=0.08,
                     fc="black", ec="black", length_includes_head=True)

    ax.set_xlim(-0.6, N + 0.1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.35, 0.5, f"γ={gamma}", ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(N + 0.05, 0.5, f"VI={k}", ha="left", va="center", fontsize=9, color="gray")


if __name__ == "__main__":
    GAMMAS = [0.3, 0.5, 0.7, 0.78, 0.79, 0.85, 0.99]

    # 终端表格输出（保留原行为）
    print(f"{'γ':>5} | {'v':<70} | π")
    print("-" * 110)
    for g in GAMMAS:
        v, pi, k = value_iteration(g)
        v_str = " ".join(f"{x:>5.2f}" for x in v)
        pi_str = "  ".join(ACT_NAME[a] for a in pi)
        print(f"{g:>5.2f} | {v_str} | {pi_str}")

    # 画图
    n_rows = len(GAMMAS) + 1   # 多一行画顶部"corridor 几何"标注
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.0 * n_rows))

    # 顶部: corridor schematic
    ax0 = axes[0]
    for s in range(N):
        if s == S_SMALL:    fc, label = COLOR_SMALL, "+0.3"
        elif s == S_BIG:    fc, label = COLOR_BIG,   "+1.0"
        else:               fc, label = COLOR_NORMAL, ""
        ax0.add_patch(Rectangle((s, 0), 1, 1, facecolor=fc, edgecolor="gray", linewidth=0.6))
        ax0.text(s + 0.5, 0.7, f"s={s}", ha="center", va="center", fontsize=9, color="gray")
        ax0.text(s + 0.5, 0.3, label, ha="center", va="center", fontsize=10, fontweight="bold")
    ax0.set_xlim(-0.6, N + 0.1)
    ax0.set_ylim(0, 1)
    ax0.set_aspect("equal")
    ax0.set_xticks([]); ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.text(-0.35, 0.5, "corridor", ha="right", va="center", fontsize=11, fontweight="bold")

    # 每行一个 γ
    for i, g in enumerate(GAMMAS):
        v, pi, k = value_iteration(g)
        draw_strip(axes[i + 1], v, pi, g, k)

    plt.tight_layout()
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    out = os.path.join(figs_dir, "corridor_gamma.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图保存到: {out}")
