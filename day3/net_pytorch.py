import torch
import torch.nn as nn
import numpy as np

# ============================================================
# Day 3: 用 PyTorch 重写 Day 2 的手写数字分类
#
# 对比你手写的代码：
#   你写的                    PyTorch 对应
#   ─────                    ──────────
#   class Layer               nn.Module（父类）
#   class Linear(Layer)       nn.Linear
#   class ReLU(Layer)         nn.ReLU()
#   class Softmax(Layer)      （CrossEntropyLoss 内置了）
#   loss_fn.backward()        loss.backward()（自动求梯度！）
#   W -= lr * grad_W          optimizer.step()
# ============================================================


# --- 任务: 补全这个网络 ---
# 继承 nn.Module，就像你的 SimpleNet 继承时一样
# nn.Module 就是 PyTorch 版的 "Layer" 父类

class DigitNet(nn.Module):

    def __init__(self):
        super().__init__()  # 调用父类的 __init__，PyTorch 要求必须写这行
        # 任务: 创建两个层
        self.fc1 = nn.Linear(64, 32)     #← 对应你写的 Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)     #← 输出 10 个类
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# 训练
# ============================================================
if __name__ == "__main__":
    from sklearn.datasets import load_digits

    # 数据
    digits = load_digits()
    X = torch.tensor(digits.data / 16.0, dtype=torch.float32)  # 转成 PyTorch 张量
    y = torch.tensor(digits.target, dtype=torch.long)          # 整数标签，不需要 one-hot

    # 创建网络、损失函数、优化器
    torch.manual_seed(42)
    net = DigitNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0)

    # 训练循环
    for epoch in range(500):
        # 1. 前向传播
        output = net(x=X)

        # 2. 算损失
        loss = loss_fn(output, y)

        # 3. 反向传播（PyTorch 自动帮你算所有梯度！）
        optimizer.zero_grad()  # 先清零上一轮的梯度
        loss.backward()        # 自动计算梯度

        # 4. 更新权重
        optimizer.step()

        if epoch % 100 == 0:
            predictions = output.argmax(dim=1)
            accuracy = (predictions == y).float().mean().item() * 100
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | 准确率: {accuracy:.1f}%")

    predictions = output.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item() * 100
    print(f"\n训练完成！最终准确率: {accuracy:.1f}%")
