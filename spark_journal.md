# Spark Journal — 现象、猜想、和待解之谜

> 独立研究者的 idea 积累地。
> 原则：**看到反直觉现象不要 scroll 过去，记下来。**
>
> 每条 entry 的生命周期：
> 💡 刚发现（观察到现象 / 冒出问题）
> 🔬 研究中（读了论文 / 做了实验）
> 📝 有初步答案（可以写 blog / workshop）
> ✅ 已写成论文或已在文献里找到定论
> 🗑️ 死因：证伪 / 已被解决 / 不值得继续

---

## 条目模板

```
### [编号] [标题] 🟡 状态

- **日期**：YYYY-MM-DD
- **触发场景**：在做什么时想到这个
- **现象 / 观察**：具体看到什么
- **猜想**：初步的 why
- **可能相关的工作**：论文、博客、他人想法
- **下一步可能做的最小实验**：1 张 GPU 能跑的
- **笔记**：后续更新
```

---

## 活跃条目

### #001 低温度采样的死循环机制 💡 → 🔬

- **日期**：2026-04-21（2026-04-21 升级思路）
- **触发场景**：讨论训好的斗破 GPT 生成行为
- **现象**：T → 0（argmax 解码）时，生成内容会陷入周期性重复或固定 token 循环
- **猜想**：
  - 死循环 = 确定性动力系统的 attractor；Pigeonhole 论证：block_size 有限 + 确定 transition → 必循环
  - 三股推力：常用字的"能量井" / induction head 延续重复 pattern / context window 周期性
  - 高 T 或 top-k/p = 注入噪声让系统逃出吸引子
- **关键升级（由用户提出）**：这不是比喻，**LLM decoding = 一个定义在离散 token space 上的高维动力系统**，可以**严格地**套用非线性动力系统 + 混沌理论的所有工具：
  - T 是 **bifurcation parameter**；T 从 0 到 ∞ 扫描是一张 bifurcation diagram
  - 固定循环 = **fixed point / limit cycle attractor**；prompt 就是 initial condition，决定落入哪个 basin of attraction
  - 中等 T 下连贯但不重复 = **edge of chaos** 临界态
  - token 扰动对后续生成的放大 = **Lyapunov exponent**
  - 采样噪声逃出循环 = **noise-induced escape from attractor**
- **潜在论文角度**："A Dynamical Systems View of LLM Decoding"
  - 首次系统地把 LLM sampling 形式化为动力系统
  - 用 attractor/basin/bifurcation 工具分析 repetition、diversity、coherence trade-off
  - 在小模型上给出可视化（bifurcation 图、吸引域图）
  - 实用价值：给 sampling 超参一个几何解释，指导 repetition penalty / temperature schedule 设计
- **可能相关的工作**：
  - Holtzman et al. 2020 "The Curious Case of Neural Text Degeneration"（Nucleus Sampling）
  - Olsson et al. 2022 (Anthropic) Induction head
  - Bertschinger & Natschläger 2004 "Edge of chaos in RNN"
  - Poole et al. 2016 "Exponential expressivity in deep networks"
  - 需要进一步查：LLM + dynamical system 的 prior art（这是关键，决定 novelty）
- **下一步最小实验**：
  - [x] **文献查（2026-04-21 完成）**：领域不是空地，但具体切入点还有空间
  - [ ] 小说 GPT 上扫 T ∈ [0.01, 2.0] × 30 个值，固定 prompt 生成 1000 步，统计：平均循环长度、唯一 n-gram 比例、生成熵
  - [ ] 画出 T 为横轴的 bifurcation 图（Feigenbaum 式，这是最有说服力的可视化）
  - [ ] 估 Lyapunov 指数：两个 prompt 差 1 个 token，追踪后续 divergence 速度
  - [ ] attention 熵 vs 循环发生的相关性（见 #002）
- **理论参考书**：Strogatz *Nonlinear Dynamics and Chaos*（用户之前读过一部分）
- **Novelty 检查（2026-04-21）**：
  - **领域不空，但具体切入点还开**。2025 已有两篇关键相关工作：
    - Wang et al. 2025 (ACL, arXiv:2502.15208) "Unveiling Attractor Cycles in LLMs" —— attractor 视角，但在 **paraphrase 段落级**
    - Li et al. 2025 (arXiv:2503.13530) "Cognitive Activation and Chaotic Dynamics in LLMs: A Quasi-Lyapunov Analysis" —— Lyapunov 指数，但在 **layer-wise activation** 空间
    - Ugail & Howard 2026 (arXiv:2601.11622) "Dynamical Systems Analysis Reveals Functional Regimes in LLMs"
    - "Mapping the Edge of Chaos" (arXiv:2501.04286) —— edge-of-chaos 视角但用于 **training** 不是 decoding
    - "Geometric Dynamics of Agentic Loops in LLMs" (arXiv:2512.10350)
  - **Agent 判断的 novel 切入点**：
    - token-level decoding 的 bifurcation diagram（temperature 作为控制参数）
    - token 轨迹的 Lyapunov exponent（不是 layer-wise 的）
    - 这个精确组合没找到现有工作
  - **定位**："不是开新话题，是加入一个 2025 年刚冒头的对话"。必须引用并延伸 Wang 2025 / Li 2025。
- **状态更新**：可做，但**时间敏感**。领域刚起步，2026 年内有 workshop 空间；2027 可能就被填满。
- **笔记**：从 #001 工程问题升级为潜在研究方向。文献检查完成，具体切入点还有 novelty。

---

### #002 低 temperature 下 attention 熵的变化 💡

- **日期**：2026-04-21
- **触发场景**：#001 的子问题
- **现象 / 猜想**：推测循环发生前，模型的 attention 会从"散开"变成"集中"，可以用熵作为预警信号
- **可能相关的工作**：Anthropic 的 attention pattern 分析
- **下一步最小实验**：生成时每步记录 attention 熵曲线，看循环临界点
- **笔记**：如果真的有预警，可做 workshop 论文

---

### #003 数据规模 vs 模型深度的边际收益 💡

- **日期**：2026-04-21
- **触发场景**：Day 14，4 本书 vs 1 本书
- **现象**：从 1 本（660 万字）加到 4 本（2000 万字），val loss 从 2.97 → 2.58，效果明显；猜想模型深度加倍收益会更少
- **猜想**：在数据不饱和阶段，加数据 > 加深度；Chinchilla 已有理论，但在同风格语料上是否有偏差未知
- **下一步最小实验**：固定架构，扫 dataset size（1 本 / 2 本 / 4 本），再固定数据扫 d_model / num_layers，画边际收益曲线
- **笔记**：（待填）

---

### #004 小模型学不到逻辑的"容量墙" 💡

- **日期**：2026-04-21
- **触发场景**：和 3b1b 视频呼应，GPT-2 vs GPT-3 的"逻辑涌现"
- **现象 / 问题**：7M 参数模型无论训多久，也学不出多步推理；必须等规模上去。为什么？是不是容量以下根本不可能表达推理？
- **猜想**：推理 = 可组合的结构，小模型的 representation space 不够区分"相似但不同"的含义；有可能用 CoT（把推理展开成序列）在小模型上绕过？
- **可能相关的工作**：
  - Emergent Abilities (Wei et al. 2022)
  - Phi-2 / Phi-3 用 curated data 让小模型出现推理
  - CoT promptability scaling
- **下一步最小实验**：用 synthetic 逻辑数据集（比如括号匹配 / 简单算术）训练不同规模小模型，找容量-任务复杂度的拐点

---

### #005 Model Collapse：数据多样性的研究方向 💡

- **日期**：2026-04-21
- **触发场景**：闲聊 LLM 风格趋同
- **现象 / 问题**：互联网上 AI 生成内容比例上升，下一代模型训在上一代输出上，会不会"熵减崩塌"
- **可能相关的工作**：
  - Shumailov et al. 2023 "The Curse of Recursion" (Model Collapse 原论文)
  - 最近的 data provenance 研究
- **下一步最小实验**：用小模型做自循环训练模拟，看几代后分布崩塌
- **笔记**：有一定热度的方向，门槛不高

---

### #006 非 backprop + 网状神经网络 + RL 做机器人控制 💡

- **日期**：2026-04-21
- **触发场景**：用户在公司环境的 Claude 里讨论过的 idea，今天重提
- **现象 / 问题**：现行机器人 RL 默认栈是"层状 NN + backprop + 梯度 RL (PPO/SAC)"，但这套栈有几个痛点：样本效率差、在线适应难、灾难性遗忘、生物不合理。**如果换成"网状/mesh NN + 非 backprop 学习 + RL"会怎样？**
- **深层动机（2026-04-21 澄清）**：不是问题驱动，是**生物合理性直觉驱动**。两条主线：
  - **架构仿生**：生物用两套拓扑解决两类问题。小脑（稀疏局部、granule cell 每个只收 ~4 个输入）做实时运控；大脑皮层（稠密长程、锥体细胞数千突触）做抽象思考。现在 ML 主流用同构 Transformer 解决一切，这不对
  - **学习规则仿生**：backprop 是**过渡解**。与物理世界交互的 agent 必须用**在线、局部、异步**的学习规则——因为真实世界不会等你攒 batch + 反向传播。现在 sim-to-real 本质是"仿真里 backprop 训爆、真机冻结部署"，这和生物完全相反
  - Backprop 的生物不可能性（Crick 1989 起已知）：weight transport problem / update locking / 全局同步 / 要求精确误差信号 / BPTT 要求反向时间缓存
- **核心 trade-off**：
  - **算力劣势**：网状 NN 在 GPU 上跑不过 dense/conv（GPU 为 GEMM 优化）。典型差距 10-100×
  - **潜在优势**：局部学习、在线适应、硬件友好（神经形态芯片）、与物理世界交互时不必停机训练
  - 这个 trade-off 是否根本（fundamental）还是当前硬件伪影（artifact），本身就是研究问题
  - **关键洞察**：就算 GPU 永远赢过 neuromorphic，**在线学习的需求也不会消失**——这是深层动机不依赖硬件发展的原因
- **相关成熟方向**（不是一片空白）：
  - **架构仿生线**：
    - CMAC / Albus 1975 —— 直接照抄小脑的控制器，70-80 年代机器人控制用过
    - Marr-Albus 理论 —— granule cell 做 sparse expansion coding + Purkinje 线性读出 + climbing fiber 监督，50 年前已有
    - Liquid Neural Networks / CfC（Hasani MIT, 2020-2023）—— 19 神经元控制无人机
    - Reservoir Computing / Echo State Network（Jaeger 2001）—— 随机稀疏高维投影 + 线性读出
    - HTM / Numenta —— 直接模仿皮层 6 层柱状结构
  - **非 backprop 学习规则线**（按离 backprop 距离排序）：
    - Feedback Alignment（Lillicrap 2016）—— 反向用随机固定权重也能学
    - Direct Feedback Alignment —— 误差直接从输出投射每层，解决 locking
    - Forward-Forward（Hinton 2022）—— 完全无反向通路
    - Equilibrium Propagation（Scellier & Bengio 2017）—— 能量模型
    - **Predictive Coding**（Rao & Ballard 1999，最近复兴）—— 每层只用局部预测误差更新，**目前最有希望的生物合理 + 可工程化方向**
    - STDP（神经形态主流）—— 纯局部 spike-timing 规则
    - Hebbian + 三因子规则（RL 最接近生物）
  - **进化/搜索线**：NEAT（Stanley 2002）、OpenAI Evolution Strategies 2017
- **当前领域状态**：已有几百 PhD 在做；性能冠军仍是 backprop + PPO/SAC；非 backprop 路线打不过 backprop 的"杀手应用"问题未解
- **潜在独立研究切入点**：
  - Liquid NN 在新机器人任务上的系统评估
  - Forward-Forward + RL 的理论扩展
  - NEAT × Transformer 策略结构的结合
  - **Mesh 稀疏度 vs 样本效率的 scaling law**（可能是最小可做的实验方向）
- **用户计划的小 demo（待具体化）**：
  - 对比 dense MLP vs 稀疏/mesh 网络在同一 RL 任务（如 CartPole / MountainCar）上的样本效率
  - 其他 demo 待聊
- **概念精确化（2026-04-30）—— RL 里的"backprop"其实是两层结构**：
  - RL 的 Bellman update 本身就是 **temporal backprop**（沿时间反向传 value），独立于 NN，tabular RL 也有
  - NN 的 backprop 是 **spatial backprop**（沿层反向传 gradient via chain rule）
  - **两者同源**，都是 dynamic programming 的应用：Werbos 1974 博士论文把最优控制的 adjoint method 移植到 NN 上、发明了 backprop；Bellman 1957 的 DP 已经包含"沿时间反传"的递归结构；Sutton 1988 TD 论文明确称 TD 为 "backpropagation through time"
  - **同构对应**（不是类比，是同一数学现象的两个实例）：
    - RL 的"长 horizon credit assignment 难题" = NN 的 "vanishing gradient 难题"
    - 折扣因子 γ^T → 0   ⟺   深网 Jacobian 连乘 → 0
    - eligibility trace / λ-return（RL 解药）   ⟺   ResNet skip / LSTM gate（NN 解药）
  - Deep RL 的更新**嵌套两层 backprop**：外层对 θ 求导（spatial），内层 Bellman 递归（temporal）；两层各自的发散倾向叠加 ≈ Sutton 的 deadly triad 的根源
- **对本条研究方向的精确化（2026-04-30）**：
  - 生物事实：大脑做 temporal backprop（多巴胺 = TD-error，Schultz 1997 实验已证），但做不了 spatial backprop（weight transport problem, Crick 1989）
  - 因此**生物可信的 RL = 保留 temporal backprop + 消灭 spatial backprop**
  - 这给 #006 一个清晰的研究边界：
    - temporal backprop 不能碰（去掉就不是 RL 了）
    - 要替换的是 spatial backprop：用 Predictive Coding / EqProp / Forward-Forward 让权重更新只用局部信号
    - 外部接口保持不变（agent ↔ env + TD-error 驱动），只是 θ 更新方式换
  - **甜区**：Scellier 2017 EqProp 已在 supervised 上 match backprop；它和 RL 的 marriage 几乎没人系统做过 → 这是 #006 的最尖切入点
- **长期研究方向定位（2026-05-01，PhD spec）—— Cross-Embodiment Fast Adaptation**：
  - **触发**：用户提出"骑自行车 → 摩托车 / 不同共享单车 / '魂穿'到另一个人身上"的类比。人类做这件事极快、几乎零示教、全程在线适应；当前 deep RL（含 RT-X / π0 / OpenVLA 等 foundation models）远做不到（差 2-3 个数量级，意味着**架构问题**而非工程问题）
  - **本质识别**：人类的快速 cross-body 适应**不是 fine-tune**（一锅端调权重），是**分层重映射**——抽象层冻结（balance / countersteer / 任务目标），接口层快速重新校准（actuator mapping / 物理常数 / 重心）。LLM fine-tune 范式假设的"全网络精修"在这里就是错的
  - **生物对应**：人类至少用 3 个异构子系统并行做这件事
    - 小脑 (cerebellum)：实时 motor control + forward model；学习规则**局部、监督式**（climbing fiber 误差信号）；时间尺度秒-分钟（极快）
    - 基底节 (basal ganglia)：高层 RL credit assignment；学习规则 TD-error（多巴胺信号）
    - 顶叶皮层 (parietal cortex)：body schema —— 不变的"我的身体"抽象；能容纳工具延伸（Maravita & Iriki 2004 猴子工具实验）
    - **快/慢学习分离 + 学习规则异构** 是人类换车快的根本原因
  - **核心 claim（可证伪）**：
    > 端到端 backprop 训练**架构上**做不到这种快速适应——不是数据问题、不是 hyperparameter 问题，是单网络强制 skill + embodiment 缠绕的必然结果
  - **#006 产品 spec（量化目标）**：
    - 适应时间 < 几分钟（人类换车级别）
    - 采样 < 100 步
    - 无人类示教
    - 全程局部、在线学习规则
    - 高层 skill 表征几乎不动
    - 架构异构分层：skill 慢 / body schema 中 / actuator interface 快
  - **最小验证实验（CartPole 换车）**：
    - env_A：默认物理常数（masspole=0.1, length=0.5, force_mag=10）；A 上训出最优 policy
    - env_B：masspole / length / force_mag 各改 ±50%
    - 对比三种迁移：① 直接 transfer（退化 baseline）② 端到端 backprop fine-tune（mainstream 基线）③ 分层 + 上层冻结 + 下层局部适应（#006 路线）
    - 横轴 fine-tune steps、纵轴 return；如果 ③ 显著快于 ②，就是 workshop 论文级别的结果
    - 资源：1 张 GPU 的零头算力即可；CartPole 简单到所有理论可算清楚，又复杂到能展示真实差距
- **算力现实约束（2026-05-01）**：
  - 真正开展研究时算力有限（RTX 3070 + 3080，预备 4070Ti Super，必要时租 4090×4）
  - 必须选**最小可证伪切片**做深，不要一口气做整套异构系统
  - 候选切入顺序（由浅入深、按算力梯度）：
    1. CartPole 换车实验（cross-embodiment 最小验证）—— 现在就能跑
    2. 分层架构扩展到 MountainCar / Pendulum
    3. 单一局部学习规则（PC 或 EqProp）替换 spatial backprop，先在 supervised 复现
    4. 把 ③ 的局部学习规则集成到 ②，形成完整 prototype
    5. Mesh 稀疏性 vs 适应速度的 scaling law（需要更多算力）
    6. Liquid NN / Neuromorphic 硬件落地（需要协作 / 真机）
  - **当前状态**：💡 长期研究方向（PhD-scale），不是短期 spark；等用户读完 Ch.7-10 + 选准最小切片再启动 ①
- **短期 workshop paper 计划（2026-05-04，目标 9-11 月投稿）**：
  - **触发**：用户需要快速发一篇论文做 PhD 申请；#006 全图太大不适合短期
  - **选定 niche**："Local Learning Rules for Online Adaptation under Dynamics Shift: A Tabular Case Study"
  - **niche 形成逻辑**（基于 2026-05-04 文献调研）：
    - 主流 cross-embodiment 已卷死（Multi-Loco / URMA / GENBOT-1K 2025 大数据 + 大模型）
    - RMA 范式被认为"基本搞定" → 但"在线持续学习"那一档没人做
    - Continual RL 抗遗忘 (LEGION / LSTOL 2025) 和 Test-Time Adaptation (TARL / WorMI 2025) 是热区，但**全用 NN + backprop tricks**
    - 生物可信学习 + RL 几乎空地：唯一近邻 = Saoud 2022 EqProp Actor-Critic / Active Predictive Coding 2024 / NeuralASM 2025（全是 supervised-flavored，没人做 cross-dynamics adaptation）
    - **3 个开放区交集 = 极少有人 fish in this pond**
  - **核心 claim（一行）**：在动力学发生变化时（cross-embodiment 的 toy 版），**局部学习规则**（Hebbian/PC-like）比 **backprop 风格的微调**显著更快地恢复到接近最优 policy；优势可用动力系统理论（contraction）解释
  - **实验设计**（CartPole 上即可起步，1 张 3070 够用）：
    - env_A：默认物理（masspole=0.1, length=0.5, force_mag=10）训出 baseline policy
    - env_B：改物理常数（例如 masspole=0.3, length=0.7, force_mag=8）
    - 对比 4 种 adaptation 策略：
      1. Frozen baseline（直接 deploy 退化基准）
      2. Continued MC（vanilla 微调）
      3. Local rule（Hebbian-like 局部更新，类似 SARSA-λ 但 trace 用衰减 indicator）
      4. RMA-style adapter（小 NN 估"环境 z"做 conditioning）
    - 衡量：recovery time / sample complexity / 鲁棒性曲线
    - 预期排序：3 ≥ 2 > 4 > 1（待验证）
  - **理论组件**：用 Borkar 两时间尺度 SA 解释为啥局部更新更快收敛 → 这把 #007 的工具用起来了
  - **6 个月时间线**：
    - Month 1（现在 5 月）：继续 RL 学习 + 精读 Saoud 2022 / RMA 论文 + 读 Borkar SA 第 6 章
    - Month 2（6 月）：跑 CartPole cross-dynamics 实验 4 种策略对比
    - Month 3（7 月）：实验完善 + 加理论 sketch (Borkar 接口)
    - Month 4（8 月）：写 paper draft
    - Month 5（9 月）：迭代 + ablation
    - Month 6（9-11 月）：投稿
  - **目标投稿场所**：
    - NeurIPS 2026 workshops（截稿通常 9-10 月）：Self-Supervised Learning / Continual Learning / Tiny Papers / "Beyond Backprop"
    - ICLR 2027 Tiny Papers：截稿 ~11 月，4 页 OK
    - CoRL 2026 Workshops：若做出 sim quadruped 扩展
  - **与 #006 全图的连接**：
    - 这篇 workshop paper = #006 的 Level-0 实证 + 理论种子
    - 不会"跑题"——是用最小化设置先验证核心 claim（局部规则 > backprop 在 online adaptation 上）
    - PhD 申请时用这篇 + #006 全图作为 thesis proposal，强逻辑链
- **关键约束（2026-05-04 用户明确）—— 影响 venue 选择**：
  - 独立研究者：无附属机构、无 co-authors（独立单作者）
  - **不能到场参会**：在职上班，无法（哪怕虚拟）注册和 presentation
  - 愿付合理范围版面费（< $4000）换发表
  - 要含金量（PhD 申请审阅人能认可）
  - **结论：会议路线全部排除**（NeurIPS/ICLR/RLC 都要至少 1 作者注册参会）；只走期刊路线
- **Venue 决策（2026-05-04 / 修订 v3 — 针对 HKUST(GZ) PhD 申请）**：
  - **目标 PhD 院校：香港科技大学（广州）HKUST(GZ)**
  - **HKUST(GZ) 评价规则**：参照 HKUST 港校学术规定，国际化标准；但评审委员会有内地背景教师，CCF / 中科院评级是 implicit 加分项
  - **关键风险发现（2026-05-04 调研）**：
    - **Frontiers 系列在 PRD 圈子风险升级**：澳门城市大学 2025-2026 学年将 Hindawi/MDPI/Frontiers 全部 1301 本期刊打包黑名单；合肥大学等内地高校跟进；多本 Frontiers 子刊（in Surgery / in Energy Research）已被中科院预警
    - **TMLR 国内体系尴尬**：不被 SCI 收录、无中科院分区、无 IF；国内"申请-考核制"博士申请**可能不算 SCI 论文**；HKUST(GZ) 偏港校大概率认，但不是 100% 保险
    - **2025 年是中科院最后一版分区表**（2026 起停止更新）→ 2025 版成为长期参照
  - **修订后的优先级**：
    - **第一战场** ⭐⭐⭐：**IEEE TNNLS** (Transactions on Neural Networks and Learning Systems)
      - APC ~$2,895 (Gold OA)
      - **CCF-B 类 + 中科院 1区 TOP + IF ~10.4**（国际 + 国内双重认可）
      - Bio-plausible RL 完美 fit（CIS 计算智能社区核心刊）
      - HKUST(GZ) 任何评审委员会都买账 → PhD 申请最强加分
      - 难点：审稿周期长（6-18 月到接收）；需要早投
    - **平行动作**：
      - **TMLR**（$0，免费、快、ML 圈强）→ supplement 而非 main publication
      - **arXiv preprint**（$0，立即可见，PhD 申请材料直接可用）
    - **拼实力的冲刺**：
      - **Nature Communications**（$6,290，IF 14.7，中科院 1区 TOP，难度高）
      - **Patterns (Cell Press)**（$5,790，IF 8.0，三因子综述发表地，narrative 完美）
    - **第二轮备选**（若主战场拒/拖）：
      1. Neurocomputing (Elsevier)：$2,930，CCF-C 中科院 2区，IF 6.0
      2. Neural Networks (Elsevier)：$3,000+，CCF-B 中科院 2区，IF 6.0，慢
      3. IEEE Access：$2,160，3-6 周快审，需包装成"工程"，国内评价混杂
    - **完全跳过（针对 HKUST(GZ) 申请）**：
      - **Frontiers 系列**（风险升级，PRD 圈子可能跟随澳门城市大学的全黑名单做法）
      - MDPI 系（黑名单常客）
      - PLOS One（topic 偏离）
      - Nature Machine Intelligence（APC $12,850 太贵且占位太满）
      - 强制到场的纯会议路线（除非找到代讲人）
  - **会议代讲路线（辅助）**：
    - NeurIPS / ICLR / ICML workshops 通常允许 substitute presenter
    - 找代讲：academic 朋友挂二作 / 付费服务 ~$300-500 / X 公开招募
    - 注册费 ~$300-700
    - 优先级：辅助，不是主战场（投稿截稿 8-9 月压力大）
- **HKUST(GZ) 候选导师池（2026-05-04 调研）**：
  - **第一梯队（同时联系，多轨并行）**：
    - **Qiang Nie (聂强)** — ROAS Thrust / RIL Lab (Robot Intelligence and Learning)
      - http://ril-lab.top/en/  邮箱 qiangnie@hkust-gz.edu.cn
      - 2024年5月加入HKUST(GZ)创建RIL Lab，新PI扩张期
      - 研究方向：人形机器人、生物医疗机器人、Motion Prediction、HRI Behavior Understanding
      - **lab 名字"Robot Intelligence and Learning"和你的 niche 字面重合**
      - 18 万/年博士奖学金（高），同时招 PhD/博后/RA/实习
      - **强匹配理由**：humanoid 是 #006 目标硬件；motion prediction 是 cross-dynamics 兄弟问题
    - **Yu Huan (虞欢)** — INTR + ROAS Thrust / AMS Lab (Mobility and Autonomous Systems)
      - amslab.org
      - 研究方向：PDE/ODE Control、Dynamical System Modeling、Physics-informed Learning、RL
      - 招 Fall 2026 + Spring 2026（slots 较多）
      - **和 #007 spark 完美对应**：动力系统 + RL + 控制理论
      - 应用偏 transportation/UAV 但理论框架和 #006/#007 同源
      - **如果论文强调"动力系统视角"那一面，她可能比 Liang 更深 fit**
    - **Junwei Liang (梁俊卫)** — AI Thrust（前面已分析）
      - 1-2 slots（竞争最激烈的）
      - CV→robotics 转向，VLA + locomotion + cross-task generalization
  - **第二梯队（备胎，5月底统一发邮件）**：
    - **Tianxiang Zhao (赵天翔)** — AI Thrust，2026年2月新加入，**新课题组最扩张期**
    - **Hao Liu (刘浩)** — AI Thrust，liuh@ust.hk，2026/2027 Fall 招生
    - **Chen Yize (陈绎泽)** — AI Thrust，control + RL（power systems application）
    - **物联网学域所有老师** — IoT Thrust，**滚动录取，对申请者背景无硬性要求**，对独立研究者最友好
  - **三轨并行联系策略**：
    - 不要把鸡蛋放 Liang 一个篮子（slots 少 + 竞争激烈）
    - 三封邮件 personalize（每封提到该老师具体论文 + 你方向和他的连接）
    - 三位都回 → 3 倍机会；独苗失败风险显著降低
- **PhD 申请实操策略（2026-05-04 / 修订 v4 — 锁定 HKUST(GZ) 三梯队）**：
  - **目标导师**：三位平行联系（Nie / Yu Huan / Liang）；备胎 5 位
  - **导师契合度分析**：
    - Liang 课题组**正从 CV 转向 robotics / embodied AI**（2023+ 加速）
    - 近期重点：Quadruped/humanoid locomotion (Omni-Perception CoRL 2025 Oral, End-to-End Humanoid 2025), cross-domain/cross-task generalization, VLA models
    - **强重合**：硬件平台（quadruped / humanoid）+ 兄弟问题（cross-domain → cross-dynamics）
    - **强互补**：他工程导向 + 大模型；你理论 + bio-plausible local learning
    - 课题组**正处扩张期**，需要新方向学生 → 独立研究者 + 特定方向 = 利好
  - **修订后的 venue 主战场**（基于 Liang 的 publish ecosystem）：
    - **首选** ⭐⭐⭐⭐⭐：**IEEE RA-L (Robotics and Automation Letters)**
      - 在他的 publish 风格内（课题组 2026 有 RA-L 论文 "Stairway to Success"）
      - APC ~$1,950（hybrid，可选不付费走非 OA）
      - **会议展示是可选**（ICRA/IROS optional）→ 完美匹配独立研究者无法到场
      - **6 个月承诺出版**（接收即出版）
      - CCF-B + 中科院 2区 TOP + IF 5.3（Q1）
      - 8 页双栏限制 → 精炼论文
    - **平行**：arXiv preprint（$0，Liang 自己也大量用）
    - **TMLR**（$0）作为 supplement，海外加分但国内打折
  - **论文 narrative 重新包装**（针对 RA-L + Liang）：
    - 原 ML framing："Local Learning Rules for Online Adaptation under Dynamics Shift"
    - **新 robotics framing**："Online Robot Locomotion Adaptation via Local Learning Rules: A Quadruped Case Study under Morphological Perturbation"
    - 实验加 Pendulum / Hopper / Half-Cheetah（RA-L reviewer 习惯看到）
    - 引用 Liang 的 Omni-Perception（明确说自己的工作是其在 unseen dynamics 下的扩展）
  - **备胎**：
    - IEEE TNNLS：$2,895，CCF-B + 中科院 1区 TOP，含金量略高但和 Liang 风格略远；RA-L 拒了改投 TNNLS（reviewer 评论可复用）
    - TPAMI：他的列表里有，aspirational 长跑，1+ 年审稿
  - **会议 workshop（找代讲）**：
    - CoRL / ICRA / IROS / ICCV / CVPR workshops（他大量发这些）
    - 代讲方案：academic 朋友、付费服务、X 招募
    - 加分项不是主战场
  - **跳过**：Frontiers（PRD 风险）/ Neurocomputing（不在他风格）/ Patterns（topic 偏 ML） / Nature Comm（topic 偏） / MDPI / PLOS One
  - **关键时间线**（关键修订）：
    - **2026-05 底前**：发第一封邮件给 Liang（"独立研究者，正准备 RA-L paper"）+ 附 1 页 research statement
    - 2026-05 余下 + 2026-06：精读 Liang 课题组近 5 篇核心论文（Omni-Perception / Stairway to Success / End-to-End Humanoid / Cross-task VLA / Domain discrepancy）—— 既是 RA-L paper 的 related work，也是给他展示"我读过你工作"的凭证
    - 2026-06-08：跑实验 Layer 1+2，开始 RA-L draft
    - 2026-09：draft 完成；arXiv 上线；投 RA-L
    - 2026-09 同时：第二封邮件给 Liang，附 arXiv 链接
    - 2026-10-12：审稿期间做 ablation；开始 HKUST(GZ) PhD 正式申请
    - 2027-01-03：RA-L 接收（6 个月承诺）；PhD 申请材料含接收信
    - 2027-04-09：PhD 录取季
  - **早联系导师的核心理由**：
    - 5 月邮件 → 9 月 paper 投出 → 12 月正式申请 → 1-2 月录取季
    - 形成 8 个月的"印象积累"链
    - vs "12 月才认识 + 未发表 paper" 的对照组 → 完全不一样的录取概率
- **用户决定（2026-05-04，对前述 Q1/Q2/Q3 的回答）**：
  - **Q1（Layer 3 ODRL benchmark）**：慢慢看情况，先做 Layer 1+2，时间够再加 Layer 3
  - **Q2（局部学习规则选哪个）**：等 RL Ch.6-10 学完 + 了解三因子/PC/EqProp 后再决定
  - **Q3（硬件）**：有啥用啥；NVIDIA Newton（GPU-accelerated physics, 看起来不错）也可考虑；该租 4090 就租
- **✅ 主线任务锁定（2026-05-06）—— RA-L + TMLR 双轨投稿**：
  - 等用户补齐 RL 基础知识（Ch.6-10）后，主线任务 = 这篇 paper
  - **双轨策略**：同一篇 paper 拆两个版本，抢两个时间窗口
    - **轨道 1（快轨）— TMLR**：Layer 1 only 版（4 页 short paper，CartPole + theory sketch），7 月底投，16 周 ≈ 11 月接收 → **PhD 申请时有正式接收信**
    - **轨道 2（长轨）— RA-L**：Layer 1+2 完整版（8 页 full paper，CartPole + Pendulum/Hopper），9 月投，6 个月承诺 ≈ 2027-03 接收 → **PhD 面试时有第二篇**
    - arXiv 平行（$0，7 月底 Layer 1 版同步上线）
  - **为什么是这两个 venue**：
    - RA-L：CCF-B + 中科院 2区 TOP + 6 个月承诺出版 + 会议展示可选 + $1,950 APC + Liang 课题组生态内有
    - TMLR：$0 APC + 快（16 周）+ ML 圈认可 + 编委强（Charlin/Kamath/Murray/Shah）
    - 两者互补：TMLR 抢速度（11 月接收），RA-L 抢含金量（CCF-B + 机器人专属）
  - **与三位导师的匹配**：
    - Liang：RA-L 在他的生态里 → 投 RA-L 他买账
    - Nie：TNNLS 更 fit 但他也认 IEEE 体系 → RA-L OK
    - Yu Huan：两个都可以 → 不挑
- **完整时间线（2026-05-06 锁定 / v3——7 月底完稿 + AI 工具加速）**：
  - 2026-05-06 ~ 05-15：精读关键论文（Saoud 2022 / RMA / Patterns 2025 / Liang 5 篇 / Nie 代表作 / Yu Huan 代表作）
  - 2026-05 底前：给 Liang / Nie / Yu Huan 发第一封套磁信
  - 2026-06：Layer 1 实验（CartPole 4 种 adaptation 策略）+ Ch.7-8
  - 2026-07 中：Layer 1 实验完成 + paper draft（Layer 1 only 版）
  - 2026-07-31：**Paper Layer 1 版完成 → arXiv 上线 → 投 TMLR**
  - 2026-08：Layer 2 实验（Pendulum / Hopper）+ 扩写 full 版
  - 2026-09 中：**Paper full 版完成 → 投 RA-L**
  - 2026-09 底：给三位导师发第二封邮件（附 arXiv + TMLR/RA-L 投稿状态）
  - 2026-10-11：审稿期间做 ablation；若 TMLR 有 revise → 快速修回
  - **2026-11：TMLR 预期接收**（PhD 申请有正式接收信）
  - **2026-12 / 2027-01 PhD 申请窗口**：
    - 材料组合：arXiv + "Accepted at TMLR" + "Under review at RA-L" + GitHub + Research statement
    - **两个正式发表证据（一个接收 + 一个在审），比单一 arXiv 强两个档次**
  - 2027-01-03：PhD 面试轮；若有 RA-L 接收 → 更新 CV
  - 2027-03：RA-L 预期接收
  - 2027-04：PhD 录取季
  - **AI 工具加速假设**：代码生成（Claude）+ 论文骨架（Claude）+ 文献搜索（Claude）+ 云端 GPU 并行实验 → 传统 6 个月压缩到 3 个月可行
- **核心竞争优势（2026-05-06 补充）—— Bambu Lab 两年机电工程师经历**：
  - 本质：不是简历条目，是**研究 agenda 的动机来源**
  - 三项直接加分：
    1. 物理直觉——碰过真电机/传感器/振动，知道建模局限在皮肤上
    2. 机器人学基础层已掌握——运动学/动力学/PID/传感器融合/实时系统
    3. 独立项目交付能力——工业界练出来的"从 PPT 到稳定量产"闭环能力
  - 唯一短板：无正式研究经历（无发表论文）→ 这篇 RA-L/TMLR paper **恰好补这个**
  - 申请材料写法原则：写成"研究动机的起源"而非简历条目
    - "仿真里完美的参数到真机上因公差/温漂失效 → cross-dynamics adaptation 不是算法好奇心，是实际必要"
    - "工业界每次硬件迭代都重标定/重训 → 在线持续学习是唯一可扩展的做法"
  - 套磁信里各加一句（三位导师不同变体），research statement 开篇用这个做动机锚点
- **笔记**：这个方向大而散，关键是选准"一小段"做深。不要想做一切。下次聊进一步 narrow down。

### #007 RL 算法谱系的动力系统视角：从 contraction 到 chaos 💡

- **日期**：2026-04-27
- **触发场景**：在 `rl_learn/day2/` grid world + corridor 实验里扫 γ；用户在 corridor 中观察到 π* 在 γ=0.786 处切换，提出"是不是可以用非线性动力学和混沌解释"
- **现象 / 观察**：
  - corridor (1D 走廊：s=5 给 +0.3, s=10 给 +1.0)：γ < 0.786 → π*(s=5)=stay；γ > 0.786 → π*(s=5)=right。临界值正好对应 γ⁵=0.3
  - 这是**真正的一阶分岔**（参数空间中两个 deterministic 稳态在临界 γ 处切换；策略空间离散使其形如相变）
  - 但 corridor 这里**不是混沌**：单 agent + 已知模型 + finite MDP 的 Bellman 算子是 sup-norm γ-contraction，Banach 不动点定理直接禁止混沌
  - **真正的动力系统谱系**：不同 RL setting 跨度极大
    - 单 agent BOE：γ-contraction，唯一全局吸引子（最温顺的极端）
    - Multi-agent / game dynamics：replicator dynamics 在 zero-sum 博弈上**已被严格证明可混沌**（Sato et al. 2002）
    - Policy gradient on non-convex landscape：saddle、limit cycle
    - TD / Q-learning 的随机逼近：用 ODE 视角分析稳定性（Borkar）
- **核心猜想**：
  - 非线性动力学 / 分岔理论是**非 backprop 学习规则稳定性分析的天然语言**，不只是比喻
  - 具体到 #006 的研究方向：
    - EqProp 两阶段 = 能量函数 fixed-point convergence 问题
    - Feedback Alignment 的"对齐相变"（W 和 B 何时学到 alignment）= 分岔
    - Predictive coding 的层间误差传播 = 耦合 ODE 稳定性
    - Mesh NN + 异步更新 = 分布式动力学，可能 limit cycle 而非收敛
  - 这些用 contraction / Lyapunov / bifurcation / phase transition 语言能给统一的分析框架，比传统 SGD convergence 工具更合适
- **与 #001 / #006 的关系**：
  - **#001（LLM decoding 动力系统）和 #007 共享同一条思想主线**：把神经/学习系统形式化为动力系统，用 attractor/bifurcation/Lyapunov 分析。#001 关注 inference-time（decoding），#007 关注 learning-time（policy / weight evolution）
  - **#007 是 #006 的理论支撑**。要做 mesh + 非 backprop + RL，传统优化收敛工具不够；动力系统是必备工具箱
- **可能相关的工作**：
  - **教科书 / 综述**：
    - Borkar, *Stochastic Approximation: A Dynamical Systems Viewpoint*（最直接的接口）
    - Bertsekas, *DP and Optimal Control*（动力系统化处理 DP）
    - Strogatz, *Nonlinear Dynamics and Chaos*（用户已读部分）
  - **核心论文**：
    - Tsitsiklis 1994 "Asynchronous Stochastic Approximation and Q-learning" —— TD/Q convergence via ODE
    - Sato, Akiyama, Farmer 2002 "Chaos in learning a simple two-person game" —— RL 真混沌的经典
    - Mertikopoulos & Sandholm 系列：game dynamics 的 cycles / non-convergence
    - Akian & Gaubert：tropical / max-plus 代数视角的 BOE，分岔的代数结构
    - Scellier & Bengio 2017 EqProp（已在 spark/mesh_rl/literature/ 里）
- **下一步可能做的最小实验**：
  - **Level 0**：corridor 上画真正的 bifurcation diagram（γ × order parameter，如 v(s=5)−v(s=10) 或"选 right 的状态数"）。这是把 #007 的洞察可视化的最小步骤，1 张 GPU 的零头算力。
  - **Level 0.5（2026-05-02 新增，CartPole 换车 × #007 视角）**：在 (masspole, length) 或 (force_mag, masspole) 二维参数平面上扫描，画 hand_crafted PD policy 的"稳定区"边界（撑满 500 步的参数子集）。这是 #006 cross-embodiment 实验的 #007 几何对偶——把"R_A → R_B 切换"可视化为参数路径穿越稳定区的轨迹
  - **Level 1**：2 agent rock-paper-scissors，独立 Q-learning，画 (q1, q2) 相轨迹看是否出现 limit cycle / chaotic orbit。复现 Sato 2002。
  - **Level 2**：在 mesh 网络上跑 Hebbian / predictive coding 学习规则，监控权重相空间轨迹，看 fixed point / 周期 / 混沌的边界
  - **Level 3**：把 #006 的 Level 1 baseline 改成"contraction 可证明收敛"的局部学习规则，对比 backprop 的样本效率
- **概念深化与 #006 整合（2026-05-01/02）**：
  - **Temporal backprop 是 #007 的本质对象**（来自 2026-04-30 讨论）：
    - 用户精确化指出"RL 自带反向传播"——对应的是 Bellman update 的 **temporal backprop**（沿时间反传 value），独立于 NN 存在
    - Bellman update = V/Q 空间上的离散时间动力系统；"动力系统看 RL"不是借喻，是**结构同构**（DP 的迭代结构 ≡ 动力系统轨迹）
    - 启发：研究切入应聚焦在 mainstream contraction 工具失效的角落（multi-agent / 非 backprop 学习规则 / 参数空间分岔），而不是 Borkar-Tsitsiklis 已经覆盖的 single-agent tabular convergence
  - **Cross-embodiment fast adaptation = 参数空间延拓**（来自 2026-05-01 讨论）：
    - #006 的 R_A → R_B 切换 = MDP 参数空间中的 homotopy / parameter continuation
    - 同型机器人物理参数邻近 → 策略空间最优 attractor 邻近
    - 给"快速适应"一个**几何意义**：不需从头爬 V 函数，沿 parameter path 移动到新 attractor 即可
    - 候选工具：homotopy methods（数值优化）、bifurcation continuation（动力系统）、persistence of normally hyperbolic invariant manifolds (NHIM)
  - **Multi-timescale 理论 = #006 异构架构的形式分析工具**：
    - 用户提的"小脑（快）+ 基底节（中）+ 皮层（慢）"三层架构 = 经典 slow-fast dynamical system
    - 数学工具：singular perturbation, GSPT (geometric singular perturbation theory), **Borkar 两时间尺度 stochastic approximation**
    - Borkar 定理：双学习率下，快变量"看到"慢变量像常数、慢变量"看到"快变量像 stationary 平均 → 整个系统等价于一个低维动力系统
    - 这给"为啥异构分层架构能快速适应"一条**严格证明路径**：快层在慢层冻结视角下满足 contraction，不需慢层一起更新；这正是 #006 产品 spec 的形式化版本
  - **#007 战略定位调整**：从"独立的看 RL 的角度"升级为 **#006 的理论分析支撑**
    - #006 = 造出来（architecture + 实验）
    - #007 = 证出来（contraction / bifurcation / multi-timescale 分析）
    - 双轨结构（系统建造 + 形式分析）是好 PhD 论文的标准格局
  - **理论起手式更新（2026-05-02）**：Borkar SA 第 6 章（Two-Timescale SA）从"单时间尺度 contraction"扩展到多层架构 —— 这是连接 #006 异构架构到 #007 形式分析的关键钥匙
- **笔记**：
  - 时间敏感性：和 #001 一样，"动力系统看 RL"已经不是空地（Borkar 1990s、Mertikopoulos 2010s 都做了），但**把它专门用在非 backprop / mesh / 在线学习规则的稳定性分析**这个特定切入点还相对开。和 #006 绑在一起做更扎实
  - 起手式：先读 Borkar SA 前 4 章 + Mertikopoulos 一篇 game dynamics 综述 + 把 corridor 的分岔图画出来，再决定切入点

---

### #008 Tabular control 下 MC ≫ Sarsa：离散化 + bootstrap 的偏差放大 💡

- **日期**：2026-05-15
- **状态定位**：非 #006 论文主线，作为 baseline 印证 / 反教科书直觉的实证。可能未来 spawn 成独立 workshop note
- **触发场景**：day5 Sarsa 实验，对比同环境（CartPole-v1，3×3×6×6=324 boxes 离散化）下不同算法的 greedy evaluation 表现
- **现象 / 观察**：
  - 同样的 ε-greedy 探索 / 同样的离散化 / 同样的 γ=0.99：
    - **MC + α=1/N + 5000 ep**：greedy eval mean = 500.00，撑满 500 步 = **100%**
    - **Sarsa + α=0.05 + 80000 ep**：greedy eval mean = 271.51，撑满 500 步 = **1%**
  - MC 用 1/16 的训练量学到完美 policy；Sarsa 跑 80k ep 始终停在次优 attractor
  - 跟教科书 "TD ≥ MC" 直觉（Sutton & Barto 第 6 章 random walk MRP）相反
- **另外观察到的相关现象**：
  - **Training G_0 严重低估 Sarsa 真实性能**：training 时 G_0 mean = 84，greedy eval = 271，差 3.2×。原因是 ε-greedy behavior policy 的 5% 随机动作让 episode 时常提前终止
  - **1/N 步长 + Sarsa = 灾难**：α=1/N 在 Sarsa 上 final = 63（比 α=0.05 的 89 还低 26 分）。机制：1/N 假设 stationary 目标，但 Sarsa 的 bootstrap target 是 non-stationary，早期"垃圾 target"被永久锚定
  - **常数 α 在 Sarsa 上有稳态噪声球**：87-88 是 α=0.05 + ε=0.05 的稳态中心；80k ep 后仍在这附近 oscillation
- **核心猜想（机制）**：
  - **MC**：target = G_t 是真实采样 return，离散化粒度的不准确性只表现为 sample variance（noise）；1/N 平均能精确收敛到 ε-greedy policy 下的 v_π
  - **Sarsa**：target = r + γ Q(s', a') 依赖当前 Q 的离散估计；**离散化误差是 systematic bias 而非 noise**，bootstrap 沿轨迹反向传播会**放大 bias**
  - 结果：Sarsa 收敛到 "Bellman 算子在粗离散化空间的 fixed point"，**这个不动点 ≠ 真实 v_π**
  - 一句话总结：**MC 把离散化误差当噪声平均掉；Sarsa 把它当真值放大**
- **与 #006 论文主线的关系**：
  - **不是主线 claim**：#006 paper 的核心是"local learning rules vs backprop in online cross-dynamics adaptation"，不是 "MC vs Sarsa in tabular"
  - **可作 baseline 印证**：写 paper 时可在 related work / baseline 章节引一句"naive bootstrap-based local update 在粗离散化下劣于 MC-style return"，作为"为什么要研究更好的 local rule"的动机锚点
  - **可能 spawn 工作**：如果验证这个现象在 Pendulum / MountainCar / Hopper 上也成立，可写一个独立的 workshop note："When MC Outperforms TD: A Discretization Effect"
- **与 #007 spark 的关系**：
  - **直接对应 #007 工具**：bootstrap 的 sup-norm contraction 在粗离散化下不再严格成立（因为 Bellman 算子作用在"近似空间"上），可能产生伪不动点
  - **动力系统诠释**：Sarsa 的稳态噪声球（87-88 中心，long-tail oscillation）= 离散动力系统的 limit cycle / strange attractor 候选
  - **可能要做的最小实验**：在 corridor / grid world 上画 "Sarsa Q 收敛点 vs MC Q 收敛点" 的差异图，看离散化粒度 → bias gap 的 scaling 关系
- **可能相关的工作**：
  - Sutton & Barto 第 6/8 章关于 "deadly triad"（函数近似 + bootstrap + off-policy 三件套发散）—— 粗离散化是函数近似的退化情况
  - Tsitsiklis & Van Roy 1997 "An Analysis of Temporal-Difference Learning with Function Approximation" —— TD 在函数近似下可能发散的经典证明
  - 待查：是否有论文专门研究"离散化粒度 → bootstrap bias"的 scaling
- **下一步可能做的最小实验**：
  - **Level 0**（30 分钟）：把 day3 的 discretization 改细（比如 6×6×12×12 = 5184 boxes），重跑 MC 和 Sarsa，看 Sarsa 是否随粒度变细而追上 MC
  - **Level 1**（1-2 天）：写一个最小可验证的环境（finite MDP 或 corridor），扫描"离散化粒度" × "γ"，画 Sarsa-Q 和 MC-Q 的差异 heatmap
  - **Level 2**（1 周）：扩展到 Pendulum / MountainCar，看现象是否 robust
- **笔记**：
  - 不要为这个分支偏离 #006 主线。记下来，等论文主体跑完有余力时再 spawn
  - 但写 #006 paper 的 motivation section 时，**这个数据点是可引用的实证**（"in our exploratory tabular experiments, ..."）
  - 时间敏感性中等：离散化 + bootstrap 的话题不新，但具体在 control + tabular 上的清晰数据点不常见

### 📌 重大升级（2026-05-17）—— Q-learning 加入 + 完整 ordering

- **触发**：day5 train_q.py 完成 Q-learning 实验，对比 ε-greedy vs uniform behavior
- **完整 greedy evaluation 数据（100 seed × 5 算法）**：
  ```
  算法                            greedy eval mean    撑满 500
  ─────────────────────────────────────────────────────────
  MC          α=1/N  (5k ep)        500.00          100%
  Q-learning  uniform α=0.05 (80k)  500.00          100%   ← 和 MC 并列!
  Sarsa       ε-greedy α=0.05 (80k) 271.51            1%
  Q-learning  ε-greedy α=0.05 (80k) 137.03            0%   ← 最差!
  ```
- **完整 ordering**：**MC ≈ Q-learning(uniform) ≫ Sarsa > Q-learning(ε-greedy)**
- **机制升级（关键新理解）**：
  - 原诊断 "Q-learning 失败 = maximization bias 主导" **只对了一半**
  - 真实机制：max bias 单独不致命，**ε-greedy behavior 才是放大器**
  - Self-reinforcing feedback (自我强化反馈) 循环：
    1. max(noisy Q) 高估某个 (s, a*)
    2. ε-greedy 偏向 argmax → 反复采到 a*
    3. 反复访问被高估的 (s, a*)
    4. bootstrap 反复传播这个高估
    5. 回到 ①，循环放大
  - uniform behavior 打破耦合：max 仍然偏高估，但 behavior 不偏向 argmax，高估被均匀采样的其他 (s, a') "稀释" → 收敛到 Q*
- **理论对应**：Watkins 1989 原 Q-learning 收敛证明用 uniform behavior **不是巧合**，是 max bias 不被自我强化的理论必要
- **DQN 重新解读**：replay buffer (经验回放) 的真正价值是 **"近似 uniform sampling over visited (s, a)"**，不只是 sample efficiency。target network 缓解的也是这个 self-reinforcing feedback
- **paper 价值升级**：
  - 这个完整 ordering + 机制解释是个**可发表的小 note** 候选
  - 对 #006 paper：local learning rules 设计要特别注意 "behavior-target 解耦"，避免 self-reinforcing feedback
  - 对 #007 spark：max + ε-greedy 的耦合是个 nonlinear feedback (非线性反馈)，可能用动力系统语言描述（fixed point / limit cycle in Q-space）
- **下一步候选实验**：
  - **Double Q-learning**：解耦"选 a"和"估值"，看是否 ε-greedy + Double Q 也能恢复 MC 水平 → 验证 self-reinforcing feedback 机制
  - **Generative model + 高斯状态采样**（用户 2026-05-17 提出）：见下方"延伸思考"，可能比 uniform behavior 更高效
- **延伸思考（2026-05-17 用户提出，2026-05-19 实施）**：
  - 想法：env 支持 `env.unwrapped.state = (...)` 任意重置 → 以最优状态 (0,0,0,0) 为中心高斯采样状态，每个状态只走一步收集 (s, a, r, s')，做 Q-learning update
  - 这是 **generative model setting** 或 **simulator access** 经典框架（Kearns & Singh 2002, Azar et al. 2013）
  - 实施后发现关键: naive 1-step Q-learning + generative 不 work, 需要 FQI (Fitted Q-Iteration) 算法形式才能享受 generative 的优势。详见下面 "三次升级"。

### 📌 三次升级（2026-05-19）—— FQI 实施 + 完整闭环

- **触发**：day5 train_q_generative.py 和 train_q_fqi.py 实施 generative model 实验
- **三阶段实验数据**：
  - **① Naive 1-step Q-learning + generative + Gaussian** (50k transitions, σ_θ=0.05)：
    - greedy eval mean = 140.1，学习曲线剧烈震荡（73-469），从未达 500
    - 跟 ε-greedy Q-learning 的 137 同水平，**完全没有 trajectory 优势**
  - **② FQI 初版** (10k transitions, σ_θ=0.05, 100 iters)：
    - 最终 greedy eval = 61.3，100 轮 ‖ΔQ‖∞ 仍 0.37（≈ γ^100 = 0.366）
    - 失败诊断三个问题：
      - 覆盖率 461/648 = 71%（29% (s,a) 永远 Q=0）
      - **terminal transitions: 0/10000**  ← 最关键，截断 [-0.18, 0.18] 剥离了 anchor signal
      - γ-contraction 严格限制收敛速率，100 轮远不够
  - **③ FQI 修复版** (50k transitions, σ_θ=0.10, 截断放宽到 [-0.22, 0.22])：
    - 覆盖率 621/648 = 95.8%，terminal 1639/50000 = 3.3%
    - **iter 5 就达到完美 greedy eval = 500.0**，100 轮全程稳定
    - 最早达到 500 的 iteration: **5** ⭐
- **关键洞察（最深的发现）—— policy 收敛 ≪ Q 收敛**：
  - iter 5 时 ‖ΔQ‖∞ = 0.96（Q 几乎没变），但 greedy eval 已 = 500.0
  - iter 100 时 ‖ΔQ‖∞ = 0.37（Q 还在动），greedy eval 仍稳定 = 500.0
  - 物理：**argmax(Q) 只看相对关系不看绝对值**，Q 距离真实 Q* 还远但相对大小已正确
  - 工程教训：监控 ‖ΔQ‖∞ < ε 太保守浪费迭代；用 "argmax(Q) 在 K 轮内不变" 作为早停判据
  - DQN loss 还在涨但 policy 已稳定的现象，根源就是这个 argmax 的 robustness
- **完整 ordering（最终版）**：
  ```
  算法                                  Env interactions      Greedy eval
  ────────────────────────────────────────────────────────────────────
  MC                  α=1/N  / 5k ep    ~500k                500.0  ✓
  Q-FQI               50k transitions   50k                  500.0  ✓ ⭐
  Q-learning uniform  α=0.05 / 80k ep   ~1.8M                500.0  ✓
  ────────────────────────────────────────────────────────────────────
  Sarsa ε-greedy      α=0.05 / 80k ep   ~21M update          271.5
  Q-learning ε-greedy α=0.05 / 80k ep   ~20M update          137.0
  Q-naive-generative  1-step / 50k      50k                  140.1
  ────────────────────────────────────────────────────────────────────
  
  Q-FQI 用 1/10 的环境交互达到 MC 同样性能 → generative model 的工程价值印证
  ```
- **完整 mental map（最终）**：
  - Trajectory-following 在 tabular 下有**两个隐藏的 free lunch**：
    - ① Implicit backward propagation（trajectory 内 Q 沿链自然反传）
    - ② Natural inclusion of terminal anchor（episode 自然以 terminal 结尾，提供 ground truth signal）
  - 单纯把 trajectory 换成 generative + 1-step Q-learning，**两个 free lunch 都丢失** → naive 失败
  - 用 FQI（batch Bellman iteration）+ Gaussian sampling 跨越 terminal 边界，**把 free lunch 显式还原** → 反而更高效（1/10 的环境交互）
  - **结论：算法形式决定能否享受 generative model 优势**。Kearns & Singh / Azar 等理论 bound 用的就是 model-based 或 batch FQI 算法，不是 1-step stochastic Q-learning
- **对 #006 paper 的强化**：
  - sim-to-real 中：sim 端有 generative model → **应该用 FQI 或类似 batch 算法**训 prior Q*
  - real 端只有 trajectory → 用 online local rule 适应残差
  - 这不只是 architecture choice，是 sample efficiency 的 **10× 优化**
  - 这条 spark 完整闭环了 #006 paper baseline 设计的逻辑链
- **对 #007 spark 的延伸**：
  - "policy 收敛 ≪ Q 收敛" 是个动力系统现象：**argmax 操作在 Q 的某个商空间稳定**，Q 本身在 γ-contraction 下慢慢收敛
  - 可用 quotient dynamics（商动力系统）视角形式化：策略空间是 Q-space 的离散投影，投影下 contraction 在投影前先稳定
  - 这是个值得 Borkar SA 风格的形式分析切入点
- **新可发表结果候选**：这条完整 spark thread（naive Q-learn 失败 → 诊断 → FQI 修复 + Gaussian sampling + terminal anchor → 1/10 sample efficiency）可作为 #006 paper 的 Section 3 "Why algorithm form matters even at tabular"，或者独立 workshop note "Trajectory's Two Free Lunches: Why Generative Model Setting Needs FQI"

### 📌 四次升级（2026-05-20）—— Task 难度升级 + Iterated FQI

- **触发**：把 task 难度升级（reward shape + 细化离散化 + 20s episode），测试 q-fqi 极限
- **任务设计**：
  - Reward shape：r = 0.5 r_position + 0.5 r_static（vs 默认每步 +1）
    - r_position = 1 - (x/2.4)²，鼓励居中
    - r_static = 1 - 0.5(x_dot/3)² - 0.5(θ_dot/2)²，鼓励静止
    - terminal: r = 0
  - 离散化：3025 boxes（5 × 5 × 11 × 11，重点细化 θ / θ_dot 中心）vs 之前 324
  - Max episode = 1000 步 = 20s @ 50Hz（vs 之前 500 步 = 10s）
  - 目标"钉钉子"：杆静止 + 车居中 + 撑 20s 满分（1000 / 1000）
- **实验 ① — Q-FQI nail (单 Gaussian, 200k transitions)**:
  - 最终 return = 441.44 / 1000，ep_len = 518.6
  - 撑满 1000 步：0%
  - **task 升级后单 Gaussian FQI 失败**，远不如默认 task 的完美
- **失败诊断（三层耦合）**：
  - **Distribution mismatch (covariate shift)** ⭐ 最深问题：训练分布 = Gaussian(μ=0) 集中在原点；部署 trajectory 分布 = 杆稍微离开原点的状态访问频率更高 → policy 在原点完美、远点不知道怎么办
  - **覆盖率 77.6%**：细化离散化后 6050 (s,a) pairs，200k 不够（22% 永远 Q=0）
  - **Reward shaping 让 value function 复杂化**：从 V≈99 plateau 变成 V(s) 复杂函数 → 需要更精细 representation + 更多数据
- **实验 ② — Iterated FQI（Gaussian + trajectory 混合 buffer）**:
  ```
  initial: 100k Gaussian (σ 增大 [1.0, 1.0, 0.15, 1.0]) → 99.0% coverage
  outer 0: FQI 100 iter → return 692, ep_len 746
  outer 1: + 100k trajectory (Q_0 ε-greedy) → FQI 100 iter → return 832, ep_len 909
  outer 2: + 100k trajectory (Q_1) → FQI 100 iter → return 940, ep_len 1000 ⭐ 撑满 20s, std 仅 13.8!
  outer 3: + 100k trajectory (Q_2) → FQI 100 iter → return 747, ep_len 947 ← 退化!
  ```
- **关键发现 ① — Iterated FQI 大幅缓解 distribution mismatch**：
  - outer 2 撑满 1000 步 100%，return 940/1000 = 94%
  - 比 nail 单 Gaussian (441 / 518) 提升 113% return / 93% ep_len
  - 机制：trajectory data 让训练分布逐步对齐 deployment 分布，policy 在远状态也学到决策
- **关键发现 ② — Sweet spot 现象（反直觉）**：
  - outer 2 完美 → outer 3 退化（return 940 → 747，std 14 → 113）
  - **不是 "iter 越多越好"**，存在 optimal stopping point
  - 三个可能机制：
    - **Data imbalance**：buffer 占比 Gaussian:traj 从 50:50 → 25:75，Gaussian 的 broad coverage 优势被稀释
    - **Self-reinforcing trajectory data**：trajectory 用当前 Q 收集，偏向当前 policy 偏好 → 微小偏差被 trajectory 强化（类似 #008 二次升级揭示的 max + ε-greedy feedback）
    - **Deadly triad 退化版**：bootstrap + 离散化(函数近似) + off-policy data 三件套，触发轻度 instability (不稳定)
- **跟现代 offline RL 的对应**：
  - 这次 sweet spot 现象就是 offline RL 文献的 "distribution shift" 实证（Levine et al. 2020 综述）
  - CQL (Conservative Q-Learning, Kumar 2020) / IQL (Implicit Q-Learning, Kostrikov 2022) 等方法都是为了缓解这个问题
  - 你的 iterated FQI 用 "保留 Gaussian baseline" 的方式 ad-hoc 缓解，正式方法是 conservative regularization (保守正则化)
- **对 #006 paper 的强化升级**：
  - 之前 paper 主线：sim 端 generative + FQI 训 prior, real 端 online local rule 适应
  - 这次发现：sim 端不能只用 generative，需要 **iterated FQI (Gaussian + closed-loop trajectory)** 才能解决 distribution mismatch
  - 但 iterated 不能太深（sweet spot），需要 early stopping criterion
  - 这是个 paper 实施细节，但很重要
- **对 #007 spark 的延伸**：
  - Sweet spot 现象 = 在 policy improvement (策略改进) 的动力系统中出现 bifurcation：
    - outer < 2: improvement region (改进区)
    - outer = 2: optimal region (最优区)
    - outer > 2: degradation region (退化区)
  - 可以用 Borkar 两时间尺度 SA 形式化：trajectory data update 是慢变量，Q update 是快变量，慢变量的 drift 决定 stability
- **新工程教训**：
  - Reward shaping 不是 free lunch，看似简单的奖励改变让 task 难度大幅上升
  - Distribution mismatch 是 offline RL 核心难点，连 tabular setting 都不能幸免
  - Iterated FQI 是 Sweet spot 操作，需要 careful tuning，不像默认 task 的 FQI 那样"一锤子买卖"
- **完整最终 ordering（task 升级版，20s 钉钉子 task）**：
  ```
  算法                                  ep_len    return    撑满 1000
  ──────────────────────────────────────────────────────────────────
  Q-FQI nail (单 Gaussian, 200k)         518      441       0%
  Q-FQI iter outer 0 (Gaussian only)     746      692       
  Q-FQI iter outer 1 (+ 100k traj)       909      832       
  Q-FQI iter outer 2 (+ 200k traj) ⭐    1000     940       100%
  Q-FQI iter outer 3 (+ 300k traj)       947      747       (退化)
  ──────────────────────────────────────────────────────────────────
  ```
- **关键 caveat（适用范围）**：这一系列结果在 tabular CartPole 上 demonstrate，离散化粒度 / reward shape / σ 选择都是 toy env 特性。真实复杂环境（高维连续 / sparse reward / 不可重置 env）下需要换工具（DQN / DDPG / SAC + 真实物理仿真器如 Newton）。day5 的价值是 "build mental map (心智图)"，不是 "直接 deploy 算法"

---

### #009 Fine action discretization in DQN ≠ smooth control 💡

- **日期**：2026-05-24
- **触发**：day10-16 系列实验, 想在 MuJoCo InvertedPendulum 上让杆"钉钉子"(居中 + 小幅 + 不抖)
- **核心 finding（反直觉）**：DQN 把 action 从 3 个 (±3, 0) 加细到 31 个 (-3..+3 每 0.2) **几乎没改善 smoothness**, 仍然 bang-bang 行为
- **完整实验链路**（day10 → day16, 全在 dl-box GPU）：
  ```
  day10  31 action + default reward (+1/step):       撑 1000 ✓  bang-bang
  day11  + reward shape (10θ² penalty):              撑 1000 ✓  仍 bang-bang
  day12  + θ_dot/x/x_dot 全维度 reward + 800k step:  撑住 ✓     贴右壁 (local opt)
  day13  + x² 系数 0.05 → 0.3 + 1.5M step:           撑 1000 ✓  贴左壁 (mean|x|=0.86)
  day14  Double DQN + du penalty + resume:           失败       Q stale + du 让 POMDP
  day15 v2  from scratch + control-metric checkpoint: 早期 mean|x|=0.07 ⭐, 后期 collapse
  day15 v3  lower lr + early stop:                   稳定但没找到强 policy
  day15 sync5000  target_sync 500 → 5000:            居中 ✓ survive ✓ |u|=2.03 jitter ⭐ 当前最佳
  day16  narrow action [-1.5,+1.5] step 0.1:        survive 崩 (力不够压扰动)
  ```
- **机制分析**：
  - **Max operation 自我强化（#008 二次升级机制的 NN 版本）**: argmax Q(s, ·) 在 31 个 a 上做; Q 估计有 noise (NN approximation error); noise 量级 (~0.5-2) 大于邻近 fine action 的 Q 差距 (e.g. Q(s, 0.2) vs Q(s, 0.4) 差距 < 0.1); → argmax 在邻近 fine action 之间反复"翻转", 表现为 effective bang-bang
  - **离散化粒度 marginal benefit ≈ 0**: 在 NN approximation error 下, 31 action vs 3 action 几乎等价
  - **DQN max op 是本质限制**, 不是 hyperparameter 问题
- **工程发现（值得记下）**：
  - **Target sync interval 500 → 5000 是关键** (day15 实验): 让 bootstrap target 更稳定, 找到 centered policy
  - **改 reward 后 resume 旧 Q 不可行** (day14 失败): Q 估值在新 reward 下 stale, fine-tune 不收敛
  - **du penalty 必须 prev_u in state**: 否则 reward 依赖 hidden info → POMDP, NN 学不到
  - **Reward shaping 的 local optimum**: x 系数小 (0.05) → policy 学到"贴壁"是 stable state (cart 撞壁后不再漂移, 杆稳定微倾)
- **理论 takeaway**：要真正 smooth control 必须**离开 DQN max op 范式**, 用 policy gradient / actor-critic (continuous action 原生支持)
- **与 #008 的关系**：
  - #008 二次升级机制 (max + ε-greedy 自我强化) 在 tabular setting 揭示
  - #009 是同机制在 deep RL + NN approximation 下的实证: noise 来源不同 (一个是 sample, 一个是 NN approx), 但机制同构
  - → max op 是 DQN 范式的根本限制, 跨 tabular / deep 都成立
- **对 #006 paper 的强支撑**：
  - 给 paper 一个**反 DQN 的强 motivation**: "31-action discretization 在 InvertedPendulum 上仍 bang-bang, 离散化粒度的 marginal benefit 接近 0"
  - 直接论证 paper 选 continuous action 算法 (SAC/DDPG/policy gradient) 而非 fine-discretization DQN 的必要性
  - 可作 paper 的 Section 2 "Why naive solutions fail" 的实证
- **当前最佳 policy**: `day15/trained_ddqn_center_sync5000_best.script.pt`
  - mean|x| = 0.057 ✓ (居中)
  - survive = 1000 ✓
  - mean|u| = 2.03 (jitter, 待 SAC 解决)
- **下一步**: 用户学完赵世钰 Ch.9 (Policy Gradient) 后实施 SAC / DDPG, 解决 bang-bang 根本问题
- **paper-worthy 完整 story**：
  - 第一幕 (#008): tabular setting, MC ≫ Sarsa, 揭示 max + ε-greedy 自我强化
  - 第二幕 (#008 二次/三次升级): Q-learning(uniform) ≈ MC, behavior 解耦修复
  - 第三幕 (#008 三次升级 FQI): 1/10 sample efficiency, generative model + iterative refinement
  - 第四幕 (#009): deep RL + 31-action 仍 bang-bang, 揭示 max op 是范式根本限制
  - 第五幕 (待写): continuous action algorithm (SAC) 真正解决问题

---

## Meta 规则

- 新条目默认状态 💡
- Claude 和我共同维护这个文档
- 每周（或者每次想起来）一起回顾一次：哪些可以升级状态、哪些该归档
- 有明显进展的条目 → 从这里 spawn 到正式的 `spark/<topic>/` 实验文件夹
