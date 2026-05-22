import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


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
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    device = torch.device('cuda')

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    torch.manual_seed(42)
    net = DigitNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 学习率变小了，下面解释

    batch_size = 32
    num_epochs = 50
    n_train = len(X_train)

    for epoch in range(num_epochs):
        # --- 任务 1: 生成打乱的索引 ---
        # 提示：torch.randperm(n_train) 返回 [0, n_train) 的随机排列
        perm = torch.randperm(n_train)

        epoch_loss = 0.0

        # --- 任务 2: 按 batch 遍历训练集 ---
        # 提示：range(起点, 终点, 步长)
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]

            # --- 任务 3: 用索引取出这一批数据 ---
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            # --- 任务 4: 前向、算loss、反向、更新 ---
            output = net(xb)
            loss = loss_fn(output, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 5 == 0:
            avg_loss = epoch_loss / (n_train // batch_size)
            with torch.no_grad():
                test_output = net(X_test)
                test_acc = (test_output.argmax(dim=1) == y_test).float().mean().item() * 100
            print(f"Epoch {epoch:3d} | 平均Loss: {avg_loss:.4f} | 测试准确率: {test_acc:.1f}%")

    print("\n训练完成！")
