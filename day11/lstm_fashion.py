import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

# ============================================================
# Day 11: RNN / LSTM 入门
#
# 把 Fashion-MNIST (28×28) 看作序列：
#   - 每张图 = 28 个时间步
#   - 每一步输入一行 28 维向量
#   - LSTM 从上到下"读"完整张图，最后的隐藏状态用来分类
#
# 对比：
#   CNN（Day 7-10）         LSTM（Day 11）
#   ─────                  ─────
#   同时看整张图           一行一行顺序读
#   擅长空间特征           擅长时序/顺序特征
#   用 Conv2d              用 LSTM（循环）
# ============================================================


class FashionLSTM(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()

        # --- 任务 1: 创建 LSTM 层 ---
        # 提示：nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #   input_size  = 每个时间步输入向量的维度（这里是 28，一行像素）
        #   hidden_size = 隐藏状态的维度（你决定，越大越强但越慢）
        #   num_layers  = 堆几层 LSTM（多层能学更复杂特征）
        #   batch_first = True 意味着输入 shape 是 (batch, seq_len, input_size)
        #                 不加这个参数，默认是 (seq_len, batch, input_size) —— 容易搞错

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 


        # --- 任务 2: 全连接层做最终分类 ---
        # 提示：LSTM 最后一步的隐藏状态维度是 hidden_size
        #       要映射到 num_classes = 10
        self.fc = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        # 输入 x 的 shape: (batch, 1, 28, 28)  —— 和 CNN 一样
        # 但 LSTM 不要 channel 维度，我们把它 squeeze 掉，变成 (batch, 28, 28)
        # 这样就是 "batch 张图，每张 28 行，每行 28 个像素"
        x = x.squeeze(1)  # (batch, 28, 28)

        # --- 任务 3: 喂给 LSTM ---
        # 提示：out, (h_n, c_n) = self.lstm(x)
        #   out   shape: (batch, seq_len=28, hidden_size) —— 每一步的输出
        #   h_n   shape: (num_layers, batch, hidden_size) —— 最后时刻所有层的隐藏状态
        #   c_n   shape: (num_layers, batch, hidden_size) —— 最后时刻所有层的 cell 状态
        #
        # 分类任务通常用"最后一步"的输出，即 out[:, -1, :]
        # 它的 shape 是 (batch, hidden_size)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]     # TODO: 取出最后一步 out[:, -1, :]

        # --- 任务 4: 过全连接层得到 logits ---
        logits = self.fc(last_step)  # TODO: 填这里
        return logits


if __name__ == "__main__":
    train_data = FashionMNIST(root='/home/peter/dl_learn/data', train=True,
                              download=False, transform=transforms.ToTensor())
    test_data = FashionMNIST(root='/home/peter/dl_learn/data', train=False,
                             download=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device('cuda')
    torch.manual_seed(42)
    net = FashionLSTM().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {total_params:,}\n")

    num_epochs = 15
    best_acc = 0.0
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
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)

        avg_loss = epoch_loss / n_batches
        test_acc = correct / total * 100
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | 测试: {test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), '/home/peter/dl_learn/day11/best_lstm.pth')

    print(f"\n训练完成！最佳测试准确率: {best_acc:.1f}%")
