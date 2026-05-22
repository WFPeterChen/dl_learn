"""day3 / 06: Step 3 —— episode 生成 + 反向回报计算.

(包含 Step 1+2 的代码 inline，每个 step 文件保持自包含)

任务：
  1. 用当前 Q + ε-greedy 跑 1 个 episode
  2. 记录 trajectory: [(box_t, a_t, r_{t+1})] for t = 0..T-1
  3. 反向算 G_t = r_{t+1} + γ·G_{t+1}（终止时 G_T = 0）
  4. 返回 trajectory + Gs 数组，供 Step 4 更新 Q

下标约定（赵世钰 / Sutton 通用）：
    s_t  →  a_t  →  r_{t+1}, s_{t+1}
    所以 reward 列表的第 t 个元素是 r_{t+1}
"""
import numpy as np
import gymnasium as gym


# ====================================================================
# 来自 Step 1+2（保持自包含）
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


def discretize(obs):
    x, x_dot, theta, theta_dot = obs
    i_x         = int(np.digitize(x,         X_BINS))
    i_x_dot     = int(np.digitize(x_dot,     X_DOT_BINS))
    i_theta     = int(np.digitize(theta,     THETA_BINS))
    i_theta_dot = int(np.digitize(theta_dot, THETA_DOT_BINS))
    return ((i_x * N_X_DOT + i_x_dot) * N_THETA + i_theta) * N_THETA_DOT + i_theta_dot


def init_Q():
    return np.zeros((N_BOXES, N_ACTIONS))


def epsilon_greedy(Q, box, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(Q[box]))


# ====================================================================
# Step 3 新增：跑 episode + 反向算 G_t
# ====================================================================
def run_episode(env, Q, eps, gamma, rng, seed=None):
    """跑一个 episode，返回 trajectory 和 Gs。

    Returns:
        trajectory: list[(box, action, reward)]，长度 T
                    元素 t 表示：在 box=box_t 选 action=a_t，得到 reward=r_{t+1}
        Gs: np.array shape (T,)，Gs[t] = G_t = 步 t 起的折扣回报
    """
    obs, _ = env.reset(seed=seed)
    boxes, actions, rewards = [], [], []

    # ---- 正向：交互、记录 ----
    for t in range(1000):
        box = discretize(obs)
        a   = epsilon_greedy(Q, box, eps, rng)
        obs, r, term, trunc, _ = env.step(a)
        boxes.append(box)
        actions.append(a)
        rewards.append(r)
        if term or trunc:
            break

    # ---- 反向：DP 算 G_t ----
    # G_t = r_{t+1} + γ·G_{t+1}，G_T = 0
    T = len(rewards)
    Gs = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):       # t = T-1, T-2, ..., 0
        G = rewards[t] + gamma * G
        Gs[t] = G

    trajectory = list(zip(boxes, actions, rewards))
    return trajectory, Gs


# ====================================================================
# 自检
# ====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    env = gym.make("CartPole-v1")
    GAMMA = 0.99

    # ---- 测试 1：跑一个 episode，看完整 trajectory + G_t ----
    print("=== 测试 1：未训练 Q + ε=1.0（纯随机）跑 1 episode ===\n")
    Q = init_Q()
    trajectory, Gs = run_episode(env, Q, eps=1.0, gamma=GAMMA, rng=rng, seed=0)
    T = len(trajectory)

    print(f"  episode 长度 T   = {T}")
    print(f"  γ                = {GAMMA}")
    print(f"  G_0 (整段回报)   = {Gs[0]:.4f}")
    print(f"  理论 G_0         = (1-γ^T)/(1-γ) = {(1 - GAMMA**T) / (1 - GAMMA):.4f}")
    print(f"  (理论假设每步 r=1，CartPole 撑住时正好如此)")

    print(f"\n  前 5 步：")
    print(f"    {'t':>3} {'box':>5} {'a':>3} {'r':>5} {'G_t':>10}")
    for t in range(min(5, T)):
        box, a, r = trajectory[t]
        print(f"    {t:>3} {box:>5} {a:>3} {r:>5.1f} {Gs[t]:>10.4f}")
    if T > 10:
        print(f"    ...")
        for t in range(T - 5, T):
            box, a, r = trajectory[t]
            print(f"    {t:>3} {box:>5} {a:>3} {r:>5.1f} {Gs[t]:>10.4f}")

    # ---- 测试 2：验证递归关系 G_t = r_{t+1} + γ·G_{t+1} ----
    print(f"\n=== 测试 2：验证 Bellman 采样递归 ===")
    print("  随机抽几个 t，检查 G_t = r_{t+1} + γ·G_{t+1}：")
    sample_ts = rng.choice(T - 1, size=min(4, T - 1), replace=False)
    for t in sorted(sample_ts):
        _, _, r_next = trajectory[t]
        G_t          = Gs[t]
        G_t_next     = Gs[t + 1]
        recursive    = r_next + GAMMA * G_t_next
        diff         = abs(G_t - recursive)
        print(f"    t={t:>3}: G_t={G_t:.4f}, r+γ·G_(t+1)={recursive:.4f}, "
              f"差={diff:.2e}")

    last_box, last_a, last_r = trajectory[-1]
    print(f"\n  最后一步：G_(T-1) = {Gs[-1]:.4f} = r_T = {last_r:.4f}  "
          f"(无 G_T，递归终止条件)")

    # ---- 测试 3：几个 seed 的 G_0 分布 ----
    print(f"\n=== 测试 3：5 个 seed 的 G_0 分布（每次重新 init Q） ===")
    for ep in range(5):
        traj, Gs = run_episode(env, init_Q(), eps=1.0, gamma=GAMMA,
                               rng=rng, seed=ep)
        print(f"  seed={ep}: T={len(traj):>3}, G_0={Gs[0]:>7.3f}")

    env.close()
