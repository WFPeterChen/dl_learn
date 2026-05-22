import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import torchvision.models as models

# ============================================================
# Day 10: 迁移学习 — 用预训练 ResNet18 做 Fashion-MNIST 分类
#
# ResNet18 是在 ImageNet（128万张图，1000类）上训练好的
# 我们拿来，换掉最后一层，只训练最后一层
#
# 注意：ResNet 输入要求 3 通道 224×224，Fashion-MNIST 是 1 通道 28×28
# 所以需要 transforms 来调整
# ============================================================

# 数据预处理：把 1×28×28 灰度图变成 3×224×224
# Resize: 28→224
# Grayscale→RGB: 1通道→3通道（复制3份）
# Normalize: ResNet 要求的标准化参数
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # 灰度图复制成3通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 的均值
                         std=[0.229, 0.224, 0.225]),    # ImageNet 的标准差
])

if __name__ == "__main__":
    train_data = FashionMNIST(root='/home/peter/dl_learn/data', train=True,
                              download=False, transform=transform)
    test_data = FashionMNIST(root='/home/peter/dl_learn/data', train=False,
                             download=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device('cuda')

    # --- 任务 1: 加载预训练 ResNet18 ---
    # 提示：models.resnet18(weights='IMAGENET1K_V1')
    # weights='IMAGENET1K_V1' 表示用 ImageNet 预训练权重
    net = models.resnet18(weights='IMAGENET1K_V1')

    # --- 任务 2: 冻结所有层（不训练） ---
    # 提示：遍历所有参数，设 requires_grad = False
    # for param in net.parameters():
    #     param.requires_grad = False
    for param in net.parameters():
        param.requires_grad = False

    # 解冻最后两个残差块（layer4）+ 全连接层
    for param in net.layer4.parameters():
        param.requires_grad = True

    # 替换最后一层
    net.fc = nn.Linear(512, 10)
    net = net.to(device)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"总参数: {total:,} | 可训练: {trainable:,} | 冻结: {total - trainable:,}")

    loss_fn = nn.CrossEntropyLoss()
    # 把所有需要训练的参数传给优化器，用小学习率
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)

    # 只需要很少的 epoch！
    num_epochs = 15
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

        net.eval()
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total_samples += len(yb)

        avg_loss = epoch_loss / n_batches
        test_acc = correct / total_samples * 100
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | 测试: {test_acc:.1f}%")

    torch.save(net.state_dict(), '/home/peter/dl_learn/day10/resnet_fashion.pth')
    print(f"\n训练完成！最终测试准确率: {test_acc:.1f}%")
