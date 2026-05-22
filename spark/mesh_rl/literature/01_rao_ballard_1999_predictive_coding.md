# #1 Rao & Ballard 1999 — Predictive Coding in the Visual Cortex

## 元信息

- **完整标题**：Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects
- **作者**：Rajesh P. N. Rao, Dana H. Ballard
- **年份 / 期刊**：1999, *Nature Neuroscience* 2(1): 79–87
- **链接**：DOI `10.1038/4580`（Nature Neuroscience 网站，非 OA；可通过 Google Scholar 找到 PDF）
- **推荐阅读顺序**：第 1 篇（#006 文献库）
- **在 #006 里的位置**：**神经科学锚点**。为"大脑可能本来就不用 backprop"提供底层直觉。

---

## 读前：为什么读这篇？

backprop 是深度学习的支柱，但它的生物不可能性早在 1989 年就被 Crick 提出（weight transport / update locking / 全局同步 / 要求精确误差信号）。那问题来了：**如果大脑不用 backprop，它用什么？**

Predictive Coding 是过去 30 年神经科学界给出的**最有说服力的一个答案**。它的核心主张：皮层的每一层都在**预测下一层的活动**，只把**预测误差**（而不是原始信号）向上传递。学习规则因此天然是**局部的** —— 每个突触只需要看到"本层活动 + 本层预测误差"就能更新，不需要反向信号从很远的地方回传。

这篇 1999 年的 paper 是现代 predictive coding 路线的起点。它的直接贡献是用 predictive coding 解释视觉皮层里一类反常现象（extra-classical receptive field effects），但真正重要的是它给出了一套**具体的、可计算的**皮层模型。

---

## 读前：重点关注

读的时候请盯着下面几件事，它们是后续一切"predictive coding 能否替代 backprop"讨论的源头：

1. **模型架构长什么样**：分层 + 每层有 representation units 和 error units。这是所有现代 PC 模型的骨架。
2. **预测从哪来**：top-down（从高层向低层预测低层活动）。这和 CNN 的 bottom-up 特征抽取**正好相反**。
3. **误差信号怎么算**：简单的减法：$\epsilon = r - \hat{r}$，其中 $\hat{r}$ 是上一层投下来的预测。
4. **学习规则**：Hebbian 式的局部规则。盯住它 —— 你要在脑子里清清楚楚地看到"只用本层信号就能更新权重"。
5. **extra-classical receptive field effects**：这是论文的"实验验证"。不用精读，但知道它是在解释什么现象。
6. **能量函数视角**：PC 可以写成"最小化分层预测误差的联合能量函数"。这个视角后面会和 Scellier 的 EqProp 接上。

---

## 阅读笔记

（读完把你的理解写在下面。不追求完整，能讲清核心就行。卡住的地方标 `⚠️` 后面我们一起讨论。）

### 1. 问题 — 作者想解决什么？

> TODO 你填

### 2. 前人做法 — 当时已有什么工作？为什么不够？

> TODO 你填

### 3. 本文贡献 — 核心主张是什么？用你自己的话说一遍

> TODO 你填

### 4. 关键公式（亲手推一遍，别只看）

> 请在笔记里显式写出：
> - 分层模型的预测公式（$\hat{r}$ 怎么来）
> - 误差信号的定义（$\epsilon$）
> - 能量 / loss 的表达式
> - 权重更新规则（注意：它应该只依赖本层局部信号）

### 5. 实验设计 — 他们怎么验证理论？

> extra-classical receptive field effects 是什么？PC 模型怎么解释它？

### 6. 疑点 / 卡点（标 ⚠️ 的地方后面讨论）

> TODO 你填

---

## 读后：跟 #006 的连接

（读完后填。或者等跟 Claude 聊完再补。）

- **这篇给 #006 提供了什么**：
- **它的局限是什么**（1999 年的模型离"能在 MNIST 上打过 backprop"还有多远？）：
- **现代 PC 路线走到哪了**（Millidge / Bogacz / Friston 这些名字要熟）：

---

## 讨论区（Claude 补充）

读完来这里找 Claude，下面留给我们一起写：

- notation 对齐（Rao-Ballard 的符号 vs 现代 PC 论文的符号）
- 与 Friston 的自由能原理（Free Energy Principle）的关系
- PC 和 Helmholtz machine / variational autoencoder 的关系（不是巧合）
- PC 路线目前在 ML benchmark 上的真实性能（不要被"生物合理"情怀迷惑）
