import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ============================================================
# Day 4: 训练集/测试集划分 + 小批量训练
# ============================================================


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # 加载数据
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target

    # 关键：划分训练集和测试集 —— 80% 训练，20% 测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 转成 tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 创建网络
    torch.manual_seed(42)
    net = DigitNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0)

    # 训练
    for epoch in range(500):
        # 前向传播 + 反向传播（只用训练集）
        output = net(X_train)
        loss = loss_fn(output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            # 训练集准确率
            train_pred = output.argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item() * 100

            # 测试集准确率（网络没见过这些数据）
            with torch.no_grad():  # 评估时不需要算梯度，省内存
                test_output = net(X_test)
                test_pred = test_output.argmax(dim=1)
                test_acc = (test_pred == y_test).float().mean().item() * 100

            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | 训练: {train_acc:.1f}% | 测试: {test_acc:.1f}%")

    print("\n训练完成！")
