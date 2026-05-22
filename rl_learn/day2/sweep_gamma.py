"""扫 γ：观察折扣因子对 v* 和 π* 的几何影响。

预测：
- γ 小（短视）：远处的 +1 通过指数 γ^d 衰减得快，v 地图陡峭
- γ 大（远视）：远处的 +1 几乎不衰减，v 地图变平
- 策略 π* 多数情况下不变（grid world 没有"短期收益 vs 长期收益"的 trade-off）
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from grid_world_policy_eval import GRID_H, GRID_W, TARGET, N_S
from value_iteration import value_iteration
from viz import draw_policy, draw_values

GAMMAS = [0.3, 0.7, 0.9, 0.99]
S_TARGET = TARGET[0] * GRID_W + TARGET[1]
S_01 = 1  # 状态 (0,1)
S_00 = 0  # 状态 (0,0)


if __name__ == "__main__":
    fig, axes = plt.subplots(len(GAMMAS), 2, figsize=(11, 4.5 * len(GAMMAS)))

    print(f"{'γ':>6} {'VI 步数':>8} {'v*[(0,0)]':>12} {'v*[(0,1)]':>12} {'v*[target]':>12}")
    for i, g in enumerate(GAMMAS):
        v_star, pi_star, k = value_iteration(gamma=g)
        print(f"{g:>6.2f} {k:>8d} {v_star[S_00]:>12.4f} {v_star[S_01]:>12.4f} {v_star[S_TARGET]:>12.4f}")
        draw_values(axes[i, 0], v_star, f"v*   (γ={g}, VI iters={k})")
        draw_policy(axes[i, 1], pi_star, f"π*   (γ={g})")

    plt.tight_layout()
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    out = os.path.join(figs_dir, "gamma_sweep.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图保存到: {out}")
