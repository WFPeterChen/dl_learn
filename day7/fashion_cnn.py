import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

# ============================================================
# Day 7 综合练习：Fashion-MNIST 图像分类
#
# 这次你要自己完成大部分代码！
# 数据集：60000 张 28×28 灰度图，10 类衣服鞋子
#
# 回顾你这一周学的工具：
#   Day 3: nn.Module, nn.Linear, torch.relu
#   Day 4: train/test split, mini-batch
#   Day 5: Dataset, DataLoader, 模型保存
#   Day 6: nn.Conv2d, nn.MaxPool2d, CNN 结构
# ============================================================


# --- 任务 1: 设计 CNN 网络 ---
# 图片是 1×28×28（比 Day 6 的 1×8×8 大很多）
# 建议结构：
#   conv1: 1→16, 卷积核3, padding=1   → 输出 16×28×28（padding=1 保持尺寸不变）
#   pool                               → 16×14×14
#   conv2: 16→32, 卷积核3, padding=1   → 32×14×14
#   pool                               → 32×7×7
#   拍平                               → 32×7×7 = 1568 维
#   fc1: 1568→128
#   fc2: 128→10
#
# 新知识 padding=1：在图片边缘补一圈零，让卷积后尺寸不缩小
# nn.Conv2d(1, 16, 3, padding=1)  ← 加个参数就行

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)  
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = torch.relu(self.conv2(x)) 
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # --- 任务 2: 加载数据 ---
    # 提示：FashionMNIST 自带 Dataset，不用自己写 DigitsDataset 了
    # transform=transforms.ToTensor() 会自动把图片转成 tensor 并归一化到 [0,1]
    train_data = FashionMNIST(root='/home/peter/dl_learn/data', train=True,
                              download=False, transform=transforms.ToTensor())
    test_data = FashionMNIST(root='/home/peter/dl_learn/data', train=False,
                             download=False, transform=transforms.ToTensor())

    # 创建 DataLoader（和 Day 5 一样）
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # --- 任务 3: 创建网络、loss、optimizer，搬到 GPU ---
    device = torch.device('cuda')
    torch.manual_seed(42)
    net = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) 

    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {total_params}\n")
    # --- 任务 4: 训练循环 ---
    # 和 Day 5/6 的结构一模一样：
    # for epoch → for xb,yb in train_loader → 前向→loss→反向→更新
    # 每个 epoch 结束后在测试集上评估准确率

    num_epochs = 30
    best_acc = 0.0
    patience = 5       # 连续 5 个 epoch 没提升就停
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

        if True:  # 每个 epoch 都打印
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
            if test_acc > best_acc:
                best_acc = test_acc
                no_improve = 0
                torch.save(net.state_dict(), '/home/peter/dl_learn/day7/best_model.pth')  # 存最好的模型
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"连续 {patience} 轮没提升，早停！最佳: {best_acc:.1f}%")
                    break  # 跳出 for 循环

    # --- 任务 5: 保存模型 ---
    torch.save(net.state_dict(), '/home/peter/dl_learn/day7/digit_model.pth')
    print("\n模型已保存！")
    print("训练完成！")
