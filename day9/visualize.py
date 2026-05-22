import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Day 9: 训练可视化
#
# 新工具：
#   matplotlib — Python 的画图库，类似 MATLAB 的 plot
#   plt.plot(x, y) — 画折线图
#   plt.imshow(matrix) — 画热力图（用于混淆矩阵）
# ============================================================

CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# --- 任务 1: 画 loss 曲线 ---
# 参数：train_losses 和 test_losses 都是 list，每个元素是一个 epoch 的平均 loss
# 提示：
#   plt.figure() — 创建新图
#   plt.plot(train_losses, label='训练') — 画折线，label 是图例文字
#   plt.plot(test_losses, label='测试')
#   plt.xlabel('Epoch') — x 轴标签
#   plt.ylabel('Loss') — y 轴标签
#   plt.legend() — 显示图例
#   plt.savefig('路径.png') — 保存成图片
#   plt.close() — 关闭图，释放内存

def plot_loss(train_losses, test_losses, save_path):
    plt.figure() #— 创建新图
    plt.plot(train_losses, label='Train') #— 画折线，label 是图例文字
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch') #— x 轴标签
    plt.ylabel('Loss') #— y 轴标签
    plt.legend() #— 显示图例
    plt.savefig(save_path) #— 保存成图片
    plt.close() #— 关闭图，释放内存

# --- 任务 2: 画混淆矩阵 ---
# 参数：all_labels 是真实标签 list，all_preds 是预测标签 list
# 提示：
#   1. 先算混淆矩阵：创建 10×10 的零矩阵
#      for label, pred in zip(all_labels, all_preds):
#          matrix[label][pred] += 1
#   2. 画图：
#      plt.figure(figsize=(10, 8)) — 创建大一点的图
#      plt.imshow(matrix, cmap='Blues') — 用蓝色热力图
#      plt.colorbar() — 显示颜色条
#      plt.xticks(range(10), CLASS_NAMES, rotation=45) — x 轴标类别名
#      plt.yticks(range(10), CLASS_NAMES)
#      plt.xlabel('预测')
#      plt.ylabel('实际')
#      plt.tight_layout() — 自动调整布局防止文字被裁
#      plt.savefig(save_path)
#      plt.close()

def plot_confusion_matrix(all_labels, all_preds, save_path):
    # 第一步：算矩阵（for 循环只管这一步）
    matrix = np.zeros((10, 10), dtype=int)
    for label, pred in zip(all_labels, all_preds):
        matrix[label][pred] += 1

    # 第二步：画图（只画一次）
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(10), CLASS_NAMES, rotation=45)
    plt.yticks(range(10), CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 记录每个 epoch 的 loss（用来画图）
    train_loss_history = []
    test_loss_history = []

    num_epochs = 50
    for epoch in range(num_epochs):
        # 训练
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

        scheduler.step()
        avg_train_loss = epoch_loss / n_batches
        train_loss_history.append(avg_train_loss)

        # 评估（同时记录测试 loss）
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        n_test_batches = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                output = net(xb)
                test_loss += loss_fn(output, yb).item()
                pred = output.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
                n_test_batches += 1

        avg_test_loss = test_loss / n_test_batches
        test_loss_history.append(avg_test_loss)
        test_acc = correct / total * 100
        print(f"Epoch {epoch:3d} | 训练Loss: {avg_train_loss:.4f} | 测试Loss: {avg_test_loss:.4f} | 测试: {test_acc:.1f}%")

    # 画 loss 曲线
    plot_loss(train_loss_history, test_loss_history,
              '/home/peter/dl_learn/day9/loss_curve.png')
    print("Loss 曲线已保存到 day9/loss_curve.png")

    # 收集所有测试集的预测结果（用来画混淆矩阵）
    all_labels = []
    all_preds = []
    net.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb).argmax(dim=1)
            all_labels.extend(yb.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    # 画混淆矩阵
    plot_confusion_matrix(all_labels, all_preds,
                          '/home/peter/dl_learn/day9/confusion_matrix.png')
    print("混淆矩阵已保存到 day9/confusion_matrix.png")
