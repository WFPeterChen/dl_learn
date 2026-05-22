import numpy as np

# ============================================================
# Day 2: 继承(inheritance) + 真实数据训练
# 目标：理解继承，重构 Day 1 的代码，然后用真实数据训练
# ============================================================

# --- 知识点: 继承 ---
# Day 1 里 Linear、ReLU、Sigmoid 都有 forward 和 backward 方法
# 但它们之间没有"关系"，Python 不知道它们是同一类东西
#
# 继承：定义一个"父类"当模板，其他类从它"继承"
# C 语言类比：
#   想象你有很多 struct 都需要一个 name 字段和 print 函数
#   C 里你只能复制粘贴，Python 里可以用继承自动获得


class Layer:
    """
    所有层的父类（基类）
    定义了通用接口：每个层都必须有 forward 和 backward
    """

    def forward(self, x):
        raise NotImplementedError  # 子类必须自己实现

    def backward(self, grad_output):
        raise NotImplementedError  # 子类必须自己实现


# ============================================================
# 任务 1: 让 Linear 继承 Layer
# 提示: class Linear(Layer):  ← 括号里写父类名
# ============================================================

class Linear(Layer):  # <-- 改这一行，让它继承 Layer
    """全连接层"""

    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T


# ============================================================
# 任务 2: 让 ReLU 和 Sigmoid 也继承 Layer
# ============================================================

class ReLU(Layer):  # <-- 改这一行
    """ReLU 激活函数"""

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0)


class Sigmoid(Layer):  # <-- 改这一行
    """Sigmoid 激活函数"""

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
    
class Softmax(Layer):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output 
    

class MSELoss:
    def forward(self, prediction, target):
        return np.mean((prediction - target) ** 2)

    def backward(self, prediction, target):
        return 2 * (prediction - target) / prediction.size

class CrossEntropyLoss:
    def forward(self, prediction, target):
        return -np.mean(np.sum(target * np.log(prediction + 1e-8), axis=1))
          # 加 1e-8 防止 log(0)
    def backward(self, prediction, target):
        return (prediction - target) / len(prediction)
          # 除以样本数是因为 forward 里用了 mean

class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
        self.softmax = Softmax()  # 十分类用 Softmax 替代 Sigmoid

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self, grad_output):
        grad = self.fc2.backward(grad_output)  # Softmax+CrossEntropy 的梯度直接传给 fc2
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)


# ============================================================
# 用真实数据训练：识别手写数字 0-9（十分类）
# ============================================================
if __name__ == "__main__":
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data / 16.0       # (1797, 64) 归一化
    y_labels = digits.target     # (1797,) 标签 0-9

    # One-hot 编码：数字 3 → [0,0,0,1,0,0,0,0,0,0]
    y = np.zeros((len(y_labels), 10))
    y[np.arange(len(y_labels)), y_labels] = 1.0

    print(f"数据集：{X.shape[0]} 个样本，{X.shape[1]} 维输入，{10} 个类别")

    np.random.seed(42)
    net = SimpleNet(input_size=64, hidden_size=32, output_size=10)
    loss_fn = CrossEntropyLoss()
    lr = 1.0

    for epoch in range(500):
        output = net.forward(X)
        loss = loss_fn.forward(output, y)
        grad = loss_fn.backward(output, y)
        net.backward(grad)

        net.fc1.W -= lr * net.fc1.grad_W
        net.fc1.b -= lr * net.fc1.grad_b
        net.fc2.W -= lr * net.fc2.grad_W
        net.fc2.b -= lr * net.fc2.grad_b

        if epoch % 100 == 0:
            predictions = np.argmax(output, axis=1)  # 取概率最大的类
            accuracy = np.mean(predictions == y_labels) * 100
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | 准确率: {accuracy:.1f}%")

    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == y_labels) * 100
    print(f"\n训练完成！最终准确率: {accuracy:.1f}%")
