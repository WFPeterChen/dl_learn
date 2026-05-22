"""day3 / 04: 状态离散化 —— Ch.5 MC control 的第一块积木。

任务：定义 discretize(obs) → box index ∈ {0, ..., N_BOXES-1}。
后面 Q-table 就按这个 box index 做行索引。

设计原则：
  - x, x_dot      与平衡核心关系不大 → coarse
  - theta, theta_dot 直接决定杆是否倒 → finer，且在 0 附近切细
  - bin 边界要包住 episode 内可能到达的范围
    (theta 自然在 ±0.21 范围内被截断；x 在 ±2.4 内)

最早做这件事的人：Michie & Chambers 1968 BOXES system。
"""
import numpy as np


# ----------------------------------------------------------------------
# 离散化的 bin 边界
# ----------------------------------------------------------------------
# np.digitize 约定：bins = [b1, b2, ..., bK] 切出 K+1 个 bin
#   返回值是 0..K，含义：
#     0 if x < b1
#     i if b_{i-1} <= x < b_i
#     K if b_K <= x
#
# 所以 N 个边界 → N+1 个 bin。
X_BINS         = np.array([-0.8, 0.8])                          # 3 bins
X_DOT_BINS     = np.array([-0.5, 0.5])                          # 3 bins
THETA_BINS     = np.array([-0.105, -0.017, 0.0, 0.017, 0.105])  # 6 bins
THETA_DOT_BINS = np.array([-0.87, -0.5, 0.0, 0.5, 0.87])        # 6 bins

N_X         = len(X_BINS) + 1
N_X_DOT     = len(X_DOT_BINS) + 1
N_THETA     = len(THETA_BINS) + 1
N_THETA_DOT = len(THETA_DOT_BINS) + 1
N_BOXES     = N_X * N_X_DOT * N_THETA * N_THETA_DOT


# ----------------------------------------------------------------------
# 核心函数：4D 连续 → 1D 整数
# ----------------------------------------------------------------------
def discretize(obs):
    """连续 obs ∈ ℝ^4  →  box index ∈ [0, N_BOXES)."""
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x, X_BINS))
    i_x_dot     = int(np.digitize(x_dot, X_DOT_BINS))
    i_theta     = int(np.digitize(theta, THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))

    # 4D index → 1D index：标准 row-major 拍平公式
    # box = ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot
    #
    # 这相当于把 4D 索引看成 N_X_DOT * N_THETA * N_THETA_DOT 进制的数：
    #   box = i_x * (N_X_DOT*N_THETA*N_THETA_DOT)
    #       + i_x_dot * (N_THETA*N_THETA_DOT)
    #       + i_theta * (N_THETA_DOT)
    #       + i_theta_dot
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


def box_to_indices(box):
    """反向：1D box index → (i_x, i_x_dot, i_theta, i_theta_dot)。debug 用。"""
    i_theta_dot = box %  N_THETA_DOT;   box //= N_THETA_DOT
    i_theta     = box %  N_THETA;       box //= N_THETA
    i_x_dot     = box %  N_X_DOT;       box //= N_X_DOT
    i_x         = box
    return i_x, i_x_dot, i_theta, i_theta_dot


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== 离散化设计 ===")
    print(f"  x      : {N_X} bins, 边界 {X_BINS.tolist()}")
    print(f"  x_dot  : {N_X_DOT} bins, 边界 {X_DOT_BINS.tolist()}")
    print(f"  theta  : {N_THETA} bins, 边界 {THETA_BINS.tolist()}")
    print(f"  th_dot : {N_THETA_DOT} bins, 边界 {THETA_DOT_BINS.tolist()}")
    print(f"  → 总 box 数 = {N_X}×{N_X_DOT}×{N_THETA}×{N_THETA_DOT} = {N_BOXES}")

    # 几个例子测试
    examples = [
        ("初始状态附近",            [ 0.027, -0.006,  0.036,  0.020]),
        ("完全静止 s=0",            [ 0.0,    0.0,    0.0,    0.0  ]),
        ("杆向右倒中",              [ 0.0,    0.5,    0.10,   1.0  ]),
        ("杆刚要倒翻 (≈terminal)",  [ 0.0,    0.0,    0.20,   2.0  ]),
        ("车飘到右边沿",            [ 2.3,    0.0,    0.0,    0.0  ]),
        ("镜像：杆向左倒",          [ 0.0,   -0.5,   -0.10,  -1.0  ]),
        ("微小角度差异",            [ 0.0,    0.0,    0.005,  0.0  ]),
    ]

    print(f"\n=== 离散化测试 ===")
    print(f"{'描述':<25} {'连续 obs':<42} {'box':<5} {'4D 拆分'}")
    print("-" * 100)
    for desc, obs in examples:
        b = discretize(np.array(obs))
        idxs = box_to_indices(b)
        obs_str = "[" + ", ".join(f"{v:+.3f}" for v in obs) + "]"
        print(f"{desc:<25} {obs_str:<42} {b:<5} {idxs}")

    # 一个关键测试：同一个 box 内，多个连续状态都映到同一 ID
    print(f"\n=== 信息丢失测试 ===")
    print("以下 3 个不同 obs 都映到同一个 box（这就是离散化的代价）：")
    similar_obs = [
        [0.0, 0.0, 0.005, 0.1],
        [0.3, 0.2, 0.010, 0.3],
        [-0.5, -0.1, 0.012, 0.4],
    ]
    for obs in similar_obs:
        b = discretize(np.array(obs))
        print(f"  obs = {obs}  →  box = {b}")
