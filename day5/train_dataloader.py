import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ============================================================
# Day 5: Dataset + DataLoader + 模型保存/加载
#
# 对比 Day 4 你手写的 mini-batch：
#   Day 4 你写的                    Day 5 PyTorch 帮你做
#   ─────────                      ──────────────────
#   torch.randperm(n)              DataLoader(shuffle=True)
#   for start in range(0,n,bs)     for xb, yb in dataloader
#   X_train[batch_idx]             自动取 batch
# ============================================================


# --- 任务 1: 完成 DigitsDataset 类 ---
# 继承 Dataset，实现 __len__ 和 __getitem__
# 提示：
#   __len__ 返回 self.X 的长度
#   __getitem__ 返回第 idx 个样本和标签：(self.X[idx], self.y[idx])

class DigitsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 包装成 Dataset
    train_dataset = DigitsDataset(X_train, y_train)
    test_dataset = DigitsDataset(X_test, y_test)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"第 0 个样本: X shape={train_dataset[0][0].shape}, y={train_dataset[0][1]}")

    # --- 任务 2: 创建 DataLoader ---
    # 提示：DataLoader(dataset, batch_size=32, shuffle=True/False)
    #   训练集要打乱（shuffle=True），测试集不需要打乱
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda')
    torch.manual_seed(42)
    net = DigitNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练
    num_epochs = 50
    for epoch in range(num_epochs):
        net.train()  # 切换到训练模式（后面 Day 8 学 Dropout 时会用到）
        epoch_loss = 0.0
        n_batches = 0

        # 看！不用手动 randperm + 切片了，直接 for 循环
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            output = net(xb)
            loss = loss_fn(output, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # 评估
        if epoch % 5 == 0:
            net.eval()  # 切换到评估模式
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = net(xb).argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += len(yb)

            avg_loss = epoch_loss / n_batches
            test_acc = correct / total * 100
            print(f"Epoch {epoch:3d} | 平均Loss: {avg_loss:.4f} | 测试准确率: {test_acc:.1f}%")

    # --- 任务 3: 保存模型 ---
    # 提示：torch.save(net.state_dict(), '文件路径')
    torch.save(net.state_dict(), '/home/peter/dl_learn/day5/digit_model.pth')
    print("\n模型已保存！")

    new_net = DigitNet().to(device)
    new_net.load_state_dict(torch.load('/home/peter/dl_learn/day5/digit_model.pth'))
    new_net.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = new_net(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)

    test_acc = correct / total * 100
    print(f"加载模型后测试准确率: {test_acc:.1f}%")