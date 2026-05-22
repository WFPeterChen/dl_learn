# Mesh NN + 非 Backprop + RL 机器人控制 — 文献库

> 方向：spark journal #006
> 目标：读完这 4 篇，在脑海里对"为什么非 backprop 路线值得做、它已经走到哪里、瓶颈在哪"有清晰的 map 图。
>
> **使用方式**：
> 1. 按推荐顺序读，一篇一个 md 文件。
> 2. 读的时候在对应文件的"阅读笔记"部分填你的理解。
> 3. 读完一篇过来找 Claude 对齐理解、打磨疑点、补文献。
> 4. 四篇都读完后，一起汇总成 `00_map.md` —— 四条线如何交织成我们方向的理论地基。

---

## 推荐阅读顺序（由浅入深 + 历史脉络）

| # | 文件 | 为什么在这个位置 |
|---|---|---|
| 1 | [`01_rao_ballard_1999_predictive_coding.md`](01_rao_ballard_1999_predictive_coding.md) | **神经科学锚点**。奠定"大脑为什么可能本来就不用 backprop"的底层直觉 |
| 2 | [`02_lillicrap_2016_feedback_alignment.md`](02_lillicrap_2016_feedback_alignment.md) | **第一次打破 backprop 神话**。最震撼的结果：随机固定权重也能学 |
| 3 | [`03_scellier_2017_equilibrium_propagation.md`](03_scellier_2017_equilibrium_propagation.md) | **理论深度最高**。把"完全局部学习"做到数学最清楚的一篇 |
| 4 | [`04_hasani_2022_liquid_nn.md`](04_hasani_2022_liquid_nn.md) | **最工程、最贴近机器人**。19 个神经元控制无人机的那条线 |

---

## 四篇合起来回答的问题

读完这 4 篇，你应该能用自己的话回答：

1. **生物合理性的锚**：Rao & Ballard —— 皮层的计算原则长什么样？（predictive coding / 每层只用局部预测误差）
2. **工程可行性的锚**：Lillicrap —— backprop 里"精确反向权重"这个要求，能放宽到什么程度？
3. **数学完备性的锚**：Scellier —— 完全局部的学习规则，怎么在理论上等价于（或逼近）backprop？
4. **现实应用的锚**：Hasani —— 稀疏 / 连续时间 / mesh-ish 架构在真实控制任务里能跑到什么程度？

这四个锚点构成 #006 方向的**四个立柱**：神经科学 + 算法理论 + 数学证明 + 工程验证。缺一条都会让这个方向立不住。

---

## 还没读但必须提的延伸（读完 4 篇再看）

| 主题 | 关键工作 |
|---|---|
| Forward-Forward | Hinton 2022 (NeurIPS 基调演讲前后的 paper) |
| Direct Feedback Alignment | Nøkland 2016 |
| CMAC / 小脑模型 | Albus 1975 |
| Reservoir Computing | Jaeger 2001 "Echo State Networks" |
| 三因子 Hebbian + RL | Frémaux & Gerstner 2016 |
| 神经形态 + STDP + RL | Bellec et al. 2020 (e-prop) |

---

## 读完之后

- 汇总到 `00_map.md`（Claude 帮你搭）
- 升级 spark journal #006 状态 💡 → 🔬
- 开始设计 Level 1 实验（dense MLP vs mesh NN 样本效率对比）
