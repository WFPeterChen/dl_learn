# #3 Scellier & Bengio 2017 — Equilibrium Propagation

## 元信息

- **完整标题**：Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation
- **作者**：Benjamin Scellier, Yoshua Bengio
- **年份 / 期刊**：2017, *Frontiers in Computational Neuroscience* 11:24
- **arxiv preprint**：arXiv:1602.05179
- **推荐阅读顺序**：第 3 篇（#006 文献库）
- **在 #006 里的位置**：**理论深度最高**。把"完全局部的学习规则"做到数学上最干净的一篇。

---

## 读前：为什么读这篇？

Feedback Alignment 是"把 backprop 里不合生物的部分替换掉"的**工程解**。EqProp 是**理论解**：它从完全不同的出发点（能量基模型 + 动力学不动点）推出一套学习规则，这套规则**每个权重的更新只依赖它自己两端神经元的状态**（完全局部），但在数学上**可以证明它和 backprop 计算的梯度任意接近**。

关键在两阶段训练（two-phase learning）：

1. **Free phase**：网络在输入 $x$ 下自由演化到能量最低点 $s^*_{\text{free}}$，读出预测。
2. **Nudged phase**：在输出上"轻推"一个正比于 loss 的小力 $\beta$，让系统演化到新的不动点 $s^*_{\text{nudged}}$。
3. **更新权重**：$\Delta W \propto \frac{1}{\beta} (\text{local statistics in nudged} - \text{local statistics in free})$。

这个规则**完全局部** —— 每个突触只需要看它两端神经元在两个不动点下的活动差。**完全没有反向通路、没有 weight transport、没有 update locking、没有 BPTT 缓存**。

而且数学上可以证明：当 $\beta \to 0$，EqProp 的更新方向**严格等于** backprop 给出的梯度方向。这是过去 30 年里"生物合理学习规则"方向最干净的一个理论结果。

---

## 读前：重点关注

这篇是四篇里**数学最重**的。建议策略：第一遍读个大意，第二遍对着公式慢推。重点：

1. **能量函数 $E(s, \theta)$ 长什么样**：对于 Hopfield-like 网络，$E$ 是二次型的。记住它的结构。
2. **Free phase 和 Nudged phase 的区别**：nudged phase 多了一个 $\beta \cdot \ell$ 项（$\ell$ 是 loss）。
3. **为什么不动点的"差分"会给出梯度信号**：这是全篇的数学核心。关键词：**隐函数定理**。
4. **更新规则的局部性**：自己动手验证 $\Delta W_{ij}$ 只依赖 $s_i$ 和 $s_j$（两端神经元活动），不依赖任何全局误差信号。
5. **与 contrastive Hebbian learning 的关系**：EqProp 的数学身世其实很老（Movellan 1991 / Xie & Seung 2003）。Scellier 的贡献是把它和现代 backprop 用严格数学桥接起来。

---

## 阅读笔记

### 1. 问题 — 作者想解决什么？

> TODO 你填

### 2. 前人做法 — Hopfield / contrastive Hebbian / Boltzmann machine 这条线已经有什么工作？

> TODO 你填（重点：为什么前人工作一直没能和 backprop 对齐）

### 3. 本文贡献 — 核心定理是什么？一句话描述

> TODO 你填

### 4. 关键公式（这一节要花最多时间）

> 请逐个写清楚：
>
> - 能量函数 $E(u, \theta)$ 的形式（$u$ 是神经元状态向量）
> - Free phase 不动点方程 $\partial E / \partial u = 0$
> - Nudged phase 的总能量 $E + \beta \ell$ 与其不动点
> - **两阶段梯度估计**（中心公式）：
>
>   $$\lim_{\beta \to 0} \frac{1}{\beta} \left( \frac{\partial E}{\partial \theta}(s^*_{\text{nudged}}) - \frac{\partial E}{\partial \theta}(s^*_{\text{free}}) \right) = \frac{\partial \ell}{\partial \theta}$$
>
>   —— 自己推一遍，用隐函数定理。这个推导是全篇灵魂。
> - 局部更新规则：$\Delta W_{ij}$ 的表达式

### 5. 实验设计 — 在什么任务上验证？和 backprop 差多少？

> TODO 你填

### 6. 疑点 / 卡点

> 可能的卡点：
> - 隐函数定理那一步怎么跨过去？
> - 为什么 $\beta \to 0$ 极限下还能在有限 $\beta$ 下学到东西？
> - Nudged phase 的"轻推"在物理上怎么实现？

---

## 读后：跟 #006 的连接

- **这篇给 #006 提供了什么**：理论最干净的"完全局部学习规则"范式。如果未来我们需要数学背书，这是最硬的一块。
- **EqProp 的直接后继**：
  - Laborieux et al. 2021 — Scaling EqProp to deeper nets（ImageNet-scale）
  - Ernoult et al. 2019 — EqProp + 空间局部误差（生物更合理）
  - Kendall et al. 2020 — Analog physical implementation of EqProp on memristive hardware
- **为什么 EqProp 对机器人控制有吸引力**：它的两阶段动力学天然适合"机器人与环境交互"—— 每个 interaction step 都可以看作一次 nudged phase。这是 #006 方向里值得深挖的角度。

---

## 讨论区（Claude 补充）

- 隐函数定理那一步的手把手推导
- EqProp 和 contrastive divergence (Hinton 2002) 的数学关系
- 最新 (2024-2025) EqProp 在机器人控制任务上的工作
- 如果我们要做 Level 1 基线实验，EqProp 和 FA 里哪个更适合当"非 backprop" 组？
