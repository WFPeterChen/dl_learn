"""day6 / DQN on InvertedPendulum-v5 (MuJoCo).

从 tabular Q (day5) 跨越到函数近似 — NN 替代 Q-table.
跑在 dl-box (RTX 4070Ti Super).

⭐ 3 处 TODO 是 DQN 的核心 (你填):
   (1) QNet            — 4 层 NN: 4 → 512 → 128 → 3
   (2) select_action   — ε-greedy 探索
   (3) dqn_update      — 一次 SGD update (这是 DQN 的心脏)
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque


# ====================================================================
# 超参数 — 一处统一管理, 后面调参方便
# ====================================================================
ENV_ID            = "InvertedPendulum-v5"
N_ACTIONS         = 3                              # 离散化 force ∈ {-3, 0, +3}
ACTION_VALUES     = [-3.0, 0.0, 3.0]               # 对应 action index 0, 1, 2

STATE_DIM         = 4
HIDDEN_1          = 512
HIDDEN_2          = 128

GAMMA             = 0.99
LR                = 1e-3                           # Adam learning rate
BATCH_SIZE        = 64
REPLAY_CAPACITY   = 100000
WARMUP_STEPS      = 1000                           # buffer 至少这么多才开训
TARGET_SYNC_FREQ  = 500                            # 每 N 个 train step 同步 target net
TOTAL_STEPS       = 100000

EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY_STEPS   = 50000                          # ε 线性衰减步数

EVAL_EVERY        = 5000
EVAL_EPISODES     = 10

DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED              = 42


# ====================================================================
# (A) Action discretize wrapper — 我写好, 看懂即可
# ====================================================================
class DiscreteActionWrapper(gym.ActionWrapper):
    """把 InvertedPendulum-v5 的连续 action (∈[-3,+3]) 包成 3 个离散 action.

    DQN 需要离散动作 — Q-net 输出 Q[s, a] 对每个离散 a.
    Agent 输出 int 0/1/2, 这个 wrapper 转成连续 force 喂给 env.
    """
    def __init__(self, env, action_values):
        super().__init__(env)
        self._action_values = np.asarray(action_values, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(action_values))

    def action(self, a):
        return np.array([self._action_values[a]], dtype=np.float32)


# ====================================================================
# (1) Q-Network ⭐ TODO 你填
# ====================================================================
class QNet(nn.Module):
    """MLP: STATE_DIM (4) → HIDDEN_1 (512) → HIDDEN_2 (128) → N_ACTIONS (3).

    输出: Q(s, a) for 所有 a, shape (batch_size, N_ACTIONS).

    ───────────────────────────────────────────────────────────────
    ⭐ TODO 1: 在 __init__ 和 forward 里填代码
    ───────────────────────────────────────────────────────────────

    提示:
      • 用 nn.Sequential 把几层叠起来最简单
      • 隐藏层后加 nn.ReLU(); 输出层 *不要* 加任何 activation
        (因为 Q 值可正可负, ReLU 会把负值砍成 0, 错的)
      • forward 直接 self.net(s) 就行

    代码框架 (你照着填):

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, HIDDEN_1),
                # TODO: ReLU
                # TODO: Linear HIDDEN_1 -> HIDDEN_2
                # TODO: ReLU
                # TODO: Linear HIDDEN_2 -> N_ACTIONS
            )

        def forward(self, s):
            # s shape (batch, 4), 返回 (batch, 3)
            return self.net(s)

    为什么这样设计:
      • day5 tabular: Q 是 |S|×|A| 个独立数值, 每个 (s, a) 单独存
      • DQN: Q 是参数化函数 Q(s, a; θ), θ 共享 → 相似 s 的 Q 值会泛化
      • 隐藏层维度大 (512, 128) 给 NN 足够"表达力"学复杂 Q 函数
      • 4 层 (输入+2 隐+输出) 是 deep RL 常用最小配置
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(STATE_DIM, HIDDEN_1),
                nn.ReLU(),
                nn.Linear(HIDDEN_1, HIDDEN_2),
                nn.ReLU(),
                nn.Linear(HIDDEN_2, N_ACTIONS)
        )
        # ---------------------------------------------------------

    def forward(self, s):
        # s shape (batch, 4), 返回 (batch, N_ACTIONS=3)
        return self.net(s)


# ====================================================================
# (B) Replay buffer — 我写好
# ====================================================================
class ReplayBuffer:
    """循环 buffer 存 (s, a, r, s', done).

    DQN 核心 trick #1: 打破时序相关性 + 接近 uniform sampling.
    对应你 day5 spark #008 二次升级的洞察 (uniform behavior 让 Q-learning 收敛).
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        """随机抽 batch_size 个 transitions, 返回 5 个 GPU tensor."""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.tensor(np.array(s),      dtype=torch.float32, device=DEVICE),
            torch.tensor(a,                dtype=torch.long,    device=DEVICE),
            torch.tensor(r,                dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(s_next), dtype=torch.float32, device=DEVICE),
            torch.tensor(done,             dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ====================================================================
# (2) ε-greedy action selection ⭐ TODO 你填
# ====================================================================
def select_action(q_net, state, epsilon):
    """ε 概率随机, (1-ε) 概率 argmax Q(s, ·).

    输入:
      q_net    main Q network (要训练的那个, 不是 target net)
      state    np.array, shape (STATE_DIM,)  ← 单个状态, 不是 batch
      epsilon  float, ∈ [0, 1]

    返回:
      int, action index ∈ {0, 1, ..., N_ACTIONS-1}

    ───────────────────────────────────────────────────────────────
    ⭐ TODO 2
    ───────────────────────────────────────────────────────────────

    分两步:

    步骤 ①: 用 random.random() 判断要不要探索
      if random.random() < epsilon:
          return random.randint(0, N_ACTIONS - 1)    # 完全随机选一个 action

    步骤 ②: 否则 greedy — 用 q_net forward, 然后 argmax
      • state 是 np.array (4,), q_net 需要 (batch, 4) 输入 → 加 batch 维:
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=DEVICE).unsqueeze(0)
      • q_net forward, 但 *不需要梯度* (这里只是选 action, 不训练):
            with torch.no_grad():
                q = q_net(state_t)        # shape (1, N_ACTIONS)
      • argmax 拿最大那个的 index, 转成 python int:
            return int(q.argmax(dim=1).item())

    为什么要 torch.no_grad():
      默认 PyTorch 会建 computation graph (计算图) 准备求导, 占内存且慢.
      这里 forward 只是用来"挑动作", 不需要梯度 → no_grad 节省资源.
    """
    # ① 探索: ε 概率随机选 action
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)

    # ② 利用: (1-ε) 概率 argmax Q(s, ·)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q = q_net(state_t)                          # (1, N_ACTIONS)
        return int(q.argmax(dim=1).item())


# ====================================================================
# (3) DQN update step ⭐⭐ TODO 你填 — 这是 DQN 的心脏
# ====================================================================
def dqn_update(main_net, target_net, optimizer, batch):
    """一次 SGD update. 输入 batch, 返回 loss (标量).

    数学:
      target = r + γ · max_{a'} Q_target(s', a') · (1 - done)
                                                    ↑
                                          terminal 时不 bootstrap
      loss   = MSE(Q_main(s, a), target)
      θ ← θ - α · ∇_θ loss

    ───────────────────────────────────────────────────────────────
    ⭐ TODO 3 (最核心)
    ───────────────────────────────────────────────────────────────

    分 4 步:

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    步骤 ① 解包 batch
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        s, a, r, s_next, done = batch
        # shapes:
        # s         (B, 4)
        # a         (B,)        action index (long)
        # r         (B,)
        # s_next    (B, 4)
        # done      (B,)        1.0 if terminal, 0.0 otherwise

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    步骤 ② 计算 TD target (用 target_net, 不要梯度)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with torch.no_grad():
            q_next = target_net(s_next)            # (B, N_ACTIONS)
            q_next_max = q_next.max(dim=1)[0]      # (B,)  ← 这就是 max_{a'} Q_target(s', a')
            target = r + GAMMA * q_next_max * (1.0 - done)
                                                    # (1 - done) 让 terminal 时只用 r
                                                    # 这正是 day5 学到的 "terminal anchor"

        为什么 with torch.no_grad():
          target 是 "ground truth", 我们不希望 backward 时梯度流过 target.
          (流过 target 会导致训练不稳定 — 这是 DQN 论文明确避免的)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    步骤 ③ 计算 main net 在 (s, a) 上的 Q (有梯度)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        q_all = main_net(s)                                  # (B, N_ACTIONS)
        q_sa  = q_all.gather(1, a.unsqueeze(1)).squeeze(1)   # (B,)

        gather 解释 (DQN 特有的 trick, 第一次看会困惑):
          • q_all 是 (B, 3), 含每个状态对所有动作的 Q
          • a 是 (B,) 的 action index, 我们要 "每行取对应 action 那一列"
          • a.unsqueeze(1) → (B, 1)
          • q_all.gather(1, a.unsqueeze(1)) → (B, 1)
            意思: 在 dim=1 (列) 上, 按 a.unsqueeze(1) 的索引提取元素
          • .squeeze(1) → (B,)
          • 这等价于 q_sa[i] = q_all[i, a[i]]

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    步骤 ④ Loss + backward + optimizer step
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        loss = nn.functional.mse_loss(q_sa, target)
        optimizer.zero_grad()    # 清掉上次的梯度 (PyTorch 默认累积)
        loss.backward()           # 自动求导
        optimizer.step()          # θ ← θ - α · ∇L
        return loss.item()        # python float, 供 logging
    """
    # ① 解包 batch
    s, a, r, s_next, done = batch
    # shapes: s (B,4), a (B,) long, r (B,), s_next (B,4), done (B,)

    # ② 算 TD target (用 target_net, 不要梯度)
    with torch.no_grad():
        q_next = target_net(s_next)                 # (B, N_ACTIONS)
        q_next_max = q_next.max(dim=1)[0]           # (B,)  max_{a'} Q_target(s', a')
        target = r + GAMMA * q_next_max * (1.0 - done)
        # (1 - done): terminal 时只用 r, 不 bootstrap (你 day5 学到的 terminal anchor)

    # ③ 算 main_net 在 (s, a) 上的 Q (有梯度)
    q_all = main_net(s)                                  # (B, N_ACTIONS)
    q_sa  = q_all.gather(1, a.unsqueeze(1)).squeeze(1)   # (B,)
    # gather 的含义: q_sa[i] = q_all[i, a[i]] — 每行取对应 action 那一列

    # ④ MSE loss + backward + SGD step
    loss = nn.functional.mse_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ====================================================================
# (C) Evaluate — 我写好
# ====================================================================
def evaluate(env, q_net, n_episodes=10):
    """Greedy eval: ε=0, 跑 n_episodes 个 episode, 返回 (mean, std)."""
    q_net.eval()                    # NN 进入 eval 模式 (本例无 dropout/BN, 形式上)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10000 + ep)
        total = 0.0
        max_steps = env.spec.max_episode_steps or 1000
        for t in range(max_steps + 1):
            with torch.no_grad():
                s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a = int(q_net(s_t).argmax(dim=1).item())
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        returns.append(total)
    q_net.train()                   # 训练模式
    return float(np.mean(returns)), float(np.std(returns))


# ====================================================================
# (D) Main train loop — 我写好整体框架, 调用了你填的 select_action / dqn_update
# ====================================================================
def main():
    # 复现性
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ---- env ----
    env      = DiscreteActionWrapper(gym.make(ENV_ID), ACTION_VALUES)
    eval_env = DiscreteActionWrapper(gym.make(ENV_ID), ACTION_VALUES)

    # ---- nets + optim + buffer ----
    main_net   = QNet().to(DEVICE)
    target_net = QNet().to(DEVICE)
    target_net.load_state_dict(main_net.state_dict())   # 初始 sync
    target_net.eval()                                    # target net 不训练

    optimizer = optim.Adam(main_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(REPLAY_CAPACITY)

    n_params = sum(p.numel() for p in main_net.parameters())
    print(f"=== DQN on {ENV_ID} ===")
    print(f"  device: {DEVICE}, n_params: {n_params}")
    print(f"  γ={GAMMA}  lr={LR}  batch={BATCH_SIZE}  target_sync={TARGET_SYNC_FREQ}")
    print()

    # ---- training loop ----
    obs, _ = env.reset(seed=SEED)
    history = {"step": [], "eval_mean": [], "eval_std": [], "epsilon": [], "loss": []}
    last_loss = 0.0

    for total_step in range(1, TOTAL_STEPS + 1):
        # ε schedule (线性衰减)
        epsilon = max(EPS_END,
                      EPS_START - (EPS_START - EPS_END) * total_step / EPS_DECAY_STEPS)

        # 1) Env step
        a = select_action(main_net, obs, epsilon)
        obs_next, r, term, trunc, _ = env.step(a)
        done = float(term or trunc)
        buffer.add(obs, a, r, obs_next, done)
        obs = obs_next if not (term or trunc) else env.reset()[0]

        # 2) SGD update (buffer 够大才训)
        if len(buffer) >= WARMUP_STEPS:
            batch = buffer.sample(BATCH_SIZE)
            last_loss = dqn_update(main_net, target_net, optimizer, batch)

        # 3) Target net sync (周期性 copy main → target)
        if total_step % TARGET_SYNC_FREQ == 0:
            target_net.load_state_dict(main_net.state_dict())

        # 4) Evaluate
        if total_step % EVAL_EVERY == 0:
            mean_ret, std_ret = evaluate(eval_env, main_net, EVAL_EPISODES)
            history["step"].append(total_step)
            history["eval_mean"].append(mean_ret)
            history["eval_std"].append(std_ret)
            history["epsilon"].append(epsilon)
            history["loss"].append(last_loss)
            print(f"step {total_step:>6d} | ε={epsilon:.3f} | loss={last_loss:.4f} | "
                  f"eval = {mean_ret:>7.2f} ± {std_ret:>6.2f} | buf={len(buffer)}")

    # ---- 保存 ----
    base = os.path.dirname(__file__) or "."
    pt_path = os.path.join(base, "trained_dqn.pt")
    torch.save(main_net.state_dict(), pt_path)
    print(f"\n✓ model → {pt_path}")

    # ---- 出图 ----
    figs_dir = os.path.join(base, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].errorbar(history["step"], history["eval_mean"], yerr=history["eval_std"],
                     marker='o', lw=2, capsize=3, color='purple')
    axes[0].axhline(1000, color='green', ls='--', label='max (1000 steps)')
    axes[0].set_xlabel("train step"); axes[0].set_ylabel("eval return")
    axes[0].set_title("DQN learning curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["step"], history["epsilon"], color='orange', lw=2)
    axes[1].set_xlabel("train step"); axes[1].set_ylabel("ε")
    axes[1].set_title("ε schedule"); axes[1].grid(alpha=0.3)

    axes[2].plot(history["step"], history["loss"], color='navy', lw=2)
    axes[2].set_xlabel("train step"); axes[2].set_ylabel("loss")
    axes[2].set_title("TD loss (last sample at eval time)"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figs_dir, "dqn_training.png")
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ figs  → {fig_path}")

    env.close(); eval_env.close()


if __name__ == "__main__":
    main()
