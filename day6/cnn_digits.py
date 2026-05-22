import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ============================================================
# Day 6: CNN 卷积神经网络
#
# 对比之前的全连接网络：
#   DigitNet (Day 3-5)              CNN (Day 6)
#   ──────────────                  ────────
#   输入: 64 维向量（拍平的）        输入: 1×8×8 图片（保留空间结构）
#   nn.Linear(64, 32)              nn.Conv2d(1, 16, 3) 卷积提特征
#   nn.Linear(32, 10)              nn.MaxPool2d(2) 缩小尺寸
#                                  最后再接全连接做分类
# ============================================================


class DigitsDataset(Dataset):
    def __init__(self, X, y):
        # 关键变化：reshape 成 (N, 1, 8, 8)
        # 1 = 通道数（灰度图），8×8 = 图片尺寸
        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 8, 8)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 任务 1: 补全卷积层 ---
        # 第一层卷积：输入 1 通道，输出 16 通道，卷积核 3×3
        # 提示：nn.Conv2d(输入通道, 输出通道, 卷积核大小)
        self.conv1 = nn.Conv2d(1, 16, 3)

        # 第二层卷积：输入 16 通道，输出 32 通道，卷积核 3×3
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.pool = nn.MaxPool2d(2)  # 2×2 池化，尺寸减半

        # 全连接层：把特征图拍平后接分类
        # 8×8 → conv1(3×3) → 6×6 → pool(2×2) → 3×3
        # 3×3 → conv2(3×3) → 1×1
        # 最终：32 个通道 × 1 × 1 = 32 维
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # x 的 shape: (batch, 1, 8, 8)

        # --- 任务 2: 补全前向传播 ---
        # 流程：conv1 → relu → pool → conv2 → relu → 拍平 → fc
        # 提示：
        #   torch.relu(self.conv1(x))   卷积+激活
        #   self.pool(...)              池化
        #   x.view(x.size(0), -1)      拍平（-1 表示自动算剩下的维度）
        x = torch.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = torch.relu(self.conv2(x)) 
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 

        return x


if __name__ == "__main__":
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = DigitsDataset(X_train, y_train)
    test_dataset = DigitsDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda')
    torch.manual_seed(42)
    net = DigitCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 换成 Adam 优化器，CNN 常用

    # 打印网络结构和参数量
    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {total_params}\n")

    num_epochs = 30
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            output = net(xb)
            loss = loss_fn(output, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0:
            net.eval()
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

    print("\n训练完成！")
