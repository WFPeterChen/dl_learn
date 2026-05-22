"""Grid world 策略 / 状态值可视化。

颜色：forbidden 橙 / target 蓝 / normal 白。
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from grid_world_policy_eval import (
    GRID_H, GRID_W, TARGET, FORBIDDEN, ACTIONS, N_S, N_A, s2rc,
)

COLOR_FORBIDDEN = "#FFA500"   # orange
COLOR_TARGET    = "#7EB6FF"   # blue
COLOR_NORMAL    = "white"


def _draw_grid_bg(ax):
    for r in range(GRID_H):
        for c in range(GRID_W):
            if (r, c) == TARGET:
                color = COLOR_TARGET
            elif (r, c) in FORBIDDEN:
                color = COLOR_FORBIDDEN
            else:
                color = COLOR_NORMAL
            ax.add_patch(Rectangle((c, r), 1, 1,
                                   facecolor=color, edgecolor="gray", linewidth=0.6))
    ax.set_xlim(0, GRID_W)
    ax.set_ylim(0, GRID_H)
    ax.invert_yaxis()              # 行 0 显示在顶部
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def draw_policy(ax, pi, title=""):
    """在指定 ax 上画策略：箭头长度 ∝ 概率；stay 用圆点。"""
    _draw_grid_bg(ax)
    arrow_max = 0.4
    for s in range(N_S):
        r, c = s2rc(s)
        cx, cy = c + 0.5, r + 0.5
        for a in range(N_A):
            p = pi[s, a]
            if p < 0.01:
                continue
            dr, dc = ACTIONS[a]
            L = arrow_max * p
            if (dr, dc) == (0, 0):
                ax.plot(cx, cy, "o", color="black", markersize=3 + 6 * p)
            else:
                ax.arrow(cx, cy, dc * L, dr * L,
                         head_width=0.07, head_length=0.07,
                         fc="black", ec="black", length_includes_head=True)
    ax.set_title(title)


def draw_values(ax, v, title="", fmt="{:.2f}"):
    _draw_grid_bg(ax)
    for s in range(N_S):
        r, c = s2rc(s)
        ax.text(c + 0.5, r + 0.5, fmt.format(v[s]),
                ha="center", va="center", fontsize=11)
    ax.set_title(title)


def plot_policy(pi, title="", savepath=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_policy(ax, pi, title)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_values(v, title="", savepath=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_values(ax, v, title)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
