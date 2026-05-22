# #2 Lillicrap et al. 2016 — Feedback Alignment

## 元信息

- **完整标题**：Random synaptic feedback weights support error backpropagation for deep learning
- **作者**：Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, Colin J. Akerman
- **年份 / 期刊**：2016, *Nature Communications* 7, 13276
- **arxiv preprint**：arXiv:1411.0247（2014 年已上传，2016 年正式发表）
- **推荐阅读顺序**：第 2 篇（#006 文献库）
- **在 #006 里的位置**：**第一次打破 backprop 神话**。最震撼的结果：反向传播根本**不需要精确的反向权重**。

---

## 读前：为什么读这篇？

backprop 里藏着一个生物上说不通的细节：反向传播误差时，用的是**前向权重矩阵的精确转置** $W^T$。这意味着每个反向连接必须和对应的前向连接**精确匹配**。生物神经元怎么做到这个？—— 做不到。这叫 **weight transport problem**。

1989 年 Crick 就指出这个问题。25 年里大家都觉得：backprop 就是对生物不友好，认了吧。

然后 Lillicrap 做了一件让整个社区震惊的事：**把反向传播的 $W^T$ 换成一个完全随机的、固定不变的矩阵 $B$**。按任何先验，这不该能学。但实验结果是：**能学**，而且在 MNIST 这种任务上几乎和真 backprop 一样好。

这篇 paper 的意义远超它的技术贡献本身：它首次从**工程上**证明"精确反向权重不是 backprop 的必需品"。这一下打开了"如何找到比 backprop 更松的学习规则"的整个研究路线 —— 后面的 DFA、FF、EqProp 全是这条线的延伸。

---

## 读前：重点关注

读的时候盯住这几点，其他都可以放过：

1. **核心算法（一眼要看穿）**：训练时，前向用 $W$，反向用一个**随机初始化后冻结**的 $B$ 代替 $W^T$。$B$ 永远不更新。就这么简单。
2. **为什么居然能学**：paper 里给了一个优美的几何直觉 —— **前向权重 $W$ 会在训练中"主动对齐"反向权重 $B$**（"feedback alignment"就是这么来的）。请一定搞懂这个对齐为什么会发生。
3. **关键定理 / 数学条件**：大致说明 $W^T B$ 和某个单位矩阵"足够相近"时，误差信号方向是正确的。别抠细节，抓大意。
4. **实验范围**：从 toy problem 到 MNIST，注意作者**没有**在 ImageNet 上展示。这条路线的局限后续会明朗。
5. **局限讨论**：作者自己指出 FA 在 deep conv net 上扩展性较差。这个局限催生了后续的 DFA (Nøkland 2016) 和 target propagation 等方向。

---

## 阅读笔记

### 1. 问题 — 作者想解决什么？

> TODO 你填

### 2. 前人做法 — 传统 backprop 的"生物不合理点"有哪些？这篇打的是哪一个？

> TODO 你填（重点：weight transport problem）

### 3. 本文贡献 — Feedback Alignment 的核心算法用一句话描述

> TODO 你填

### 4. 关键公式（亲手推一遍）

> 请写出：
> - 标准 backprop 的反向信号公式
> - FA 的反向信号公式（差别就在哪里）
> - "feedback alignment" 现象的数学描述：$W^T$ 和 $B$ 为什么会随训练靠近？
> - 这和标准梯度下降的 loss 降低速度差多少？

### 5. 实验设计

> 任务选的是什么？怎么和 backprop 对比？表现差距在哪？

### 6. 疑点 / 卡点

> TODO 你填。
> 思考题：如果 FA 这么有效，为什么过去 10 年大家还在用 backprop？paper 之后出现了什么新证据让 FA 的光环褪色？

---

## 读后：跟 #006 的连接

- **这篇给 #006 提供了什么**：证明"精确反向权重不是必需的"，为所有"放松 backprop"的尝试提供了第一块理论证据。
- **FA 的直接后继**：
  - Nøkland 2016 — Direct Feedback Alignment（误差直接从输出层投射到每层，解决 update locking）
  - Frenkel et al. 2021 — DRTP（direct random target propagation）
  - Lansdell et al. 2020 — 把 FA 和 RL 结合
- **在 #006 方向里的角色**：如果我们的"mesh NN + 非 backprop + RL" 路线要落地，FA / DFA 是**最保守、最工程化**的起点。不会像 EqProp 那么理论重，也不会像纯 Hebbian 那样性能差太多。

---

## 讨论区（Claude 补充）

- FA 的 scaling 瓶颈：为什么在大模型上失效？
- FA × Transformer 有没有人做过？结果如何？
- FA 和 target propagation 的关系
- 如果我们想在 CartPole 上做 "dense MLP + backprop" vs "mesh + FA" 的对照，实验该怎么设计？
