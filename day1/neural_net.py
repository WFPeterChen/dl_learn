import numpy as np

# ============================================================
# Day 1: Python OOP + 手写神经网络
# 目标：理解类(class)的写法，同时复习前向传播
# ============================================================

# --- 知识点 1: 类的基本结构 ---
# class 类名:
#     def __init__(self, 参数):   # 构造函数，创建对象时自动调用
#         self.属性 = 值           # self 代表"这个对象自己"
#
#     def 方法名(self, 参数):      # 方法就是绑定在类上的函数
#         ...


class Linear:
    """
    全连接层（线性层）
    对应公式: output = input @ W + b
    """

    def __init__(self, in_features, out_features):
        # 用小随机数初始化权重，避免对称性问题
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        # x: shape (batch_size, in_features)
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        self.grad_W =  self.x.T @ grad_output
        self.grad_b =  np.sum(grad_output, axis=0)
        return grad_output @ self.W.T


class ReLU:
    """
    激活函数层
    对应公式: output = max(0, input)
    """

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0)

class Sigmoid:

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1- self.output)
    
class MSELoss:                      
               
    def forward(self, prediction, target):
        return np.mean((prediction - target) **2)
    def backward(self, prediction, target):
        return 2 * (prediction - target) / prediction.size

class SimpleNet:
    """
    两层神经网络：输入层 -> 隐藏层 -> 输出层
    这里演示如何把多个层组合成一个网络
    """

    def __init__(self, input_size, hidden_size, output_size):
        # 创建各层（子对象）
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.fc1.forward(x)   # 线性变换
        x = self.relu.forward(x)  # 非线性激活
        x = self.fc2.forward(x)   # 输出层
        x = self.sigmoid.forward(x)
        return x

    def backward(self, grad_output):
        grad = self.sigmoid.backward(grad_output)
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)

# ============================================================
# 主程序：创建网络，跑一次前向传播
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    # 创建网络和数据
    net = SimpleNet(input_size=3, hidden_size=4, output_size=2)
    loss_fn = MSELoss()
    x = np.random.randn(5, 3)
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])

    lr = 0.5  # 学习率：每次更新的步长

    # 训练 200 轮
    for epoch in range(200):
        # 1. 前向传播
        output = net.forward(x)

        # 2. 算损失
        loss = loss_fn.forward(output, target)

        # 3. 反向传播
        grad = loss_fn.backward(output, target)
        net.backward(grad)

        # 4. 更新权重（梯度下降）
        net.fc1.W -= lr * net.fc1.grad_W
        net.fc1.b -= lr * net.fc1.grad_b
        net.fc2.W -= lr * net.fc2.grad_W
        net.fc2.b -= lr * net.fc2.grad_b

        # 每 50 轮打印一次
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    print(f"\n训练完成！最终 Loss: {loss:.4f}")
    print(f"\n网络输出:\n{output}")
    print(f"\n目标:\n{target}")
