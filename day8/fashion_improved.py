import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

# ============================================================
# Day 8: 优化器 + Dropout + 学习率调度
#
# 对比 Day 7：
#   Day 7                          Day 8
#   ─────                          ─────
#   无 Dropout                     加 Dropout 防过拟合
#   固定学习率                      StepLR 学习率衰减
#   91.9%                          目标 > 92%
# ============================================================


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # --- 任务 1: 加 Dropout ---
        # 提示：nn.Dropout(概率)
        # 卷积层之后用 0.25，全连接层之间用 0.5
        self.dropout1 = nn.Dropout(0.25)  # 卷积后用，丢 25%
        self.dropout2 = nn.Dropout(0.5)  # 全连接之间用，丢 50%

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # --- 任务 2: 在合适的位置插入 dropout ---
        # 流程：conv1 → relu → pool → dropout1 → conv2 → relu → pool → dropout1
        #       → 拍平 → fc1 → relu → dropout2 → fc2
        x = torch.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x)) 
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    train_data = FashionMNIST(root='/home/peter/dl_learn/data', train=True,
                              download=False, transform=transforms.ToTensor())
    test_data = FashionMNIST(root='/home/peter/dl_learn/data', train=False,
                             download=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device('cuda')
    torch.manual_seed(42)
    net = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # --- 任务 3: 创建学习率调度器 ---
    # 提示：torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # 每 10 个 epoch 学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {total_params}\n")

    num_epochs = 30
    best_acc = 0.0
    patience = 5
    no_improve = 0

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

        # --- 任务 4: 每个 epoch 结束后调用 scheduler ---
        # 提示：scheduler.step()
        scheduler.step()

        # 评估
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
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | 测试: {test_acc:.1f}% | LR: {lr_now:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
            torch.save(net.state_dict(), '/home/peter/dl_learn/day8/best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停！最佳: {best_acc:.1f}%")
                break

    print(f"\n训练完成！最佳测试准确率: {best_acc:.1f}%")
