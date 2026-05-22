"""day3 / 05: Step 2 —— Q-table + ε-greedy 策略.

(包含 Step 1 的 discretize 直接 inline，每个 step 文件保持自包含)

数据结构：
    Q[box, action]：(N_BOXES, N_ACTIONS) numpy 数组
    init = 0   零初始化最简单、无偏；
               注意：Q 全 0 + 纯 greedy → np.argmax 返回 0 → 永远向左
               所以零初始化 + ε > 0 是必需的搭配

    替代：optimistic init（init 为大正数）能不靠 ε 也产生探索

策略：ε-greedy
    π(a|s) = (1-ε)·1[a=argmax]  +  ε/|A|

    实现：以 ε 概率随机均匀，否则取 argmax
"""
import numpy as np
import gymnasium as gym


# ====================================================================
# 来自 Step 1（discretize）—— 复制过来保持文件自包含
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
# Step 2 新增：Q-table + ε-greedy
# ====================================================================
N_ACTIONS = 2  # CartPole-v1: {0=left, 1=right}


def init_Q():
    """全 0 初始化。也可以 np.full((N_BOXES, N_ACTIONS), 100.0) 做 optimistic init."""
    return np.zeros((N_BOXES, N_ACTIONS))


def epsilon_greedy(Q, box, eps, rng):
    """ε-greedy 动作选择。

    参数：
        Q   : (N_BOXES, N_ACTIONS) Q-table
        box : 当前状态的离散 box index
        eps : 探索概率 ∈ [0, 1]
        rng : np.random.Generator（外部传入便于复现）

    流程：
        1. 抽一个 [0, 1) 的均匀随机数 u
        2. 若 u < eps：均匀随机选一个 action（探索）
        3. 否则：argmax_a Q[box, a]（利用）
    """
    if rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(Q[box]))


# ====================================================================
# 自检
# ====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # ---- 测试 1：数据结构 -------------------------------------------
    Q = init_Q()
    print("=== Q-table 数据结构 ===")
    print(f"  shape   : {Q.shape}            (= N_BOXES × N_ACTIONS)")
    print(f"  dtype   : {Q.dtype}")
    print(f"  初值    : Q[0] = {Q[0]}     (零初始化)")
    print(f"  内存    : {Q.nbytes} bytes  (远小于真机器人需要)")

    # ---- 测试 2：ε-greedy 的概率分布 -------------------------------
    print("\n=== ε-greedy 频率验证 ===")
    print("假设 box 100 上 Q = [1.0, 5.0]，所以 argmax = action 1")
    Q[100, 0] = 1.0
    Q[100, 1] = 5.0

    print(f"\n  {'ε':>5} | {'P(action=1) 实测':>18} | {'理论值':>10}")
    print("  " + "-" * 45)
    for eps in [0.0, 0.05, 0.1, 0.5, 1.0]:
        # 大量采样估计经验频率
        N = 100_000
        actions = np.array([epsilon_greedy(Q, box=100, eps=eps, rng=rng)
                            for _ in range(N)])
        p_emp = actions.mean()                          # action=1 的频率
        p_theory = (1 - eps) + eps * 0.5                # 公式：argmax 的总概率
        print(f"  {eps:>5.2f} | {p_emp:>18.4f} | {p_theory:>10.4f}")

    # ---- 测试 3：未训练的 Q + ε-greedy 跑一个 episode --------------
    print("\n=== 未训练 Q (全 0) + ε-greedy 跑 episode ===")
    print("预测：因为 Q=0 时 argmax 永远是 0（左推），")
    print("      纯 greedy 会变成 always_left；ε 越大越随机")

    env = gym.make("CartPole-v1")
    for eps in [0.0, 0.1, 0.5, 1.0]:
        Q = init_Q()                                    # 每次 reset Q
        lengths = []
        for ep in range(20):                            # 跑 20 个 episode 看分布
            obs, _ = env.reset(seed=ep)
            for t in range(1000):
                box = discretize(obs)
                a = epsilon_greedy(Q, box, eps=eps, rng=rng)
                obs, r, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            lengths.append(t + 1)
        L = np.array(lengths)
        print(f"  ε={eps:>4.2f}: 平均撑住 {L.mean():>5.1f} 步 (std {L.std():>4.1f})")
    env.close()
