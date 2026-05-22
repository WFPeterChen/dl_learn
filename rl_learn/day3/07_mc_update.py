"""day3 / 07: Step 4 —— MC 更新（增量式样本平均）.

(包含 Step 1+2+3 的代码 inline)

核心：
    每访问一次 (box, a)，把对应的 G_t 累入运行平均

增量公式：
    N[s, a] ← N[s, a] + 1
    Q[s, a] ← Q[s, a] + (G - Q[s, a]) / N[s, a]
                       └────────┬────────┘
              "误差驱动"，是所有 RL 更新规则的祖型

每访问 (every-visit)：trajectory 里 (s,a) 出现几次就更新几次
"""
import numpy as np
import gymnasium as gym


# ====================================================================
# 来自 Step 1+2+3（保持自包含）
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


def init_QN():
    """同时返回 Q 和 N（两个并行的 (N_BOXES, N_ACTIONS) 数组）."""
    Q = np.zeros((N_BOXES, N_ACTIONS))
    N = np.zeros((N_BOXES, N_ACTIONS), dtype=int)
    return Q, N


def epsilon_greedy(Q, box, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(Q[box]))


def run_episode(env, Q, eps, gamma, rng, seed=None):
    obs, _ = env.reset(seed=seed)
    boxes, actions, rewards = [], [], []
    for t in range(1000):
        box = discretize(obs)
        a   = epsilon_greedy(Q, box, eps, rng)
        obs, r, term, trunc, _ = env.step(a)
        boxes.append(box); actions.append(a); rewards.append(r)
        if term or trunc:
            break
    T = len(rewards)
    Gs = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        Gs[t] = G
    return list(zip(boxes, actions, rewards)), Gs


# ====================================================================
# Step 4 新增：MC 更新
# ====================================================================
def mc_update_every_visit(Q, N, trajectory, Gs):
    """每访问 MC：trajectory 中每出现一次 (box, a)，就更新一次 Q[box, a]."""
    for t, (box, a, _r) in enumerate(trajectory):
        N[box, a] += 1
        Q[box, a] += (Gs[t] - Q[box, a]) / N[box, a]


# ====================================================================
# 自检
# ====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    env = gym.make("CartPole-v1")
    GAMMA = 0.99

    # 跑 500 个 episode，固定 ε=0.5，跟踪 Q 的演化
    N_EPISODES = 500
    EPS = 0.5

    Q, N = init_QN()

    # 跟踪几个特定 (box, a) 的 Q 值演化
    # 165 是初始状态附近 (理论上每 episode t=0 都路过)
    track_pairs = [(165, 0), (165, 1), (171, 0), (171, 1)]
    Q_histories = {pair: [] for pair in track_pairs}
    G_history   = []

    for ep in range(N_EPISODES):
        traj, Gs = run_episode(env, Q, eps=EPS, gamma=GAMMA, rng=rng, seed=ep)
        mc_update_every_visit(Q, N, traj, Gs)
        G_history.append(Gs[0])
        for pair in track_pairs:
            Q_histories[pair].append(Q[pair])

    # ---- 报告 1：episode return 的演化 -------------------------
    print(f"=== 训练 {N_EPISODES} episode (ε={EPS}, γ={GAMMA}) ===\n")
    print(f"  G_0 (前  50 ep 平均) = {np.mean(G_history[ :50]):>7.3f}")
    print(f"  G_0 (中 200 ep 平均) = {np.mean(G_history[200:300]):>7.3f}")
    print(f"  G_0 (后  50 ep 平均) = {np.mean(G_history[-50:]):>7.3f}")
    print(f"  对照: random policy ≈ 22 步 → G_0 ≈ 19.85")
    print(f"        hand_crafted 500 步 → G_0 = 99.34")

    # ---- 报告 2：覆盖率 ---------------------------------------
    print(f"\n=== Box 覆盖率 ===")
    visited      = (N.sum(axis=1) > 0).sum()
    both_visited = ((N[:, 0] > 0) & (N[:, 1] > 0)).sum()
    print(f"  访问过任一 action 的 box: {visited:>3}/{N_BOXES} ({100*visited/N_BOXES:>5.1f}%)")
    print(f"  两个 action 都访问过:     {both_visited:>3}/{N_BOXES} ({100*both_visited/N_BOXES:>5.1f}%)")
    print(f"  总更新次数 = N.sum() = {N.sum()}")

    # ---- 报告 3：访问最多的 5 个 box ---------------------------
    print(f"\n=== 访问最多的 5 个 box ===")
    box_visits = N.sum(axis=1)
    top5 = np.argsort(-box_visits)[:5]
    print(f"  {'box':>4}  {'N[·,0]':>6}  {'N[·,1]':>6}  {'Q[·,0]':>8}  {'Q[·,1]':>8}  argmax  diff")
    for box in top5:
        amax  = np.argmax(Q[box])
        diff  = Q[box, 1] - Q[box, 0]
        print(f"  {box:>4}  {N[box,0]:>6}  {N[box,1]:>6}  {Q[box,0]:>8.3f}  "
              f"{Q[box,1]:>8.3f}  {amax:>6}  {diff:>+5.2f}")

    # ---- 报告 4：跟踪 (box, a) 的 Q 收敛轨迹 -----------------
    print(f"\n=== Q[box, a] 收敛轨迹（LLN 工作）===")
    print(f"  采样点：episode = 50, 100, 200, 300, 400, 499\n")
    print(f"  {'(box, a)':>10}  {'N[box,a]':>10}  "
          f"{'ep=50':>8} {'ep=100':>8} {'ep=200':>8} {'ep=300':>8} {'ep=400':>8} {'ep=499':>8}")
    for pair in track_pairs:
        h = Q_histories[pair]
        n_final = N[pair]
        snaps = [h[49], h[99], h[199], h[299], h[399], h[-1]]
        snaps_str = "  ".join(f"{q:>6.2f}" for q in snaps)
        print(f"  {str(pair):>10}  {n_final:>10}  {snaps_str}")

    env.close()
