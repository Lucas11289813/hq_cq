import torch
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-7, 7, 100)


def relu(x):
    return np.maximum(0, x)


def relu6(x):
    return np.minimum(np.maximum(0, x), 6)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hard_sigmoid(x):
    # 罗列条件
    case1 = x <= -3
    case2 = x >= 3
    case3 = ~(case1 | case2)
    # 为了不影响原数据此处创建一个新数组
    r = np.empty_like(x)
    r[case1] = 0
    r[case2] = 1
    r[case3] = x[case3] / 6 + 0.5
    return r


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def hard_tanh(x, max_val, min_val):
    # 罗列条件
    case1 = x > max_val
    case2 = x < min_val
    case3 = ~(case1 | case2)
    r = np.empty_like(x)
    r[case1] = max_val
    r[case2] = min_val
    r[case3] = x[case3]
    return r


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1).reshape(x.shape[0], -1).repeat(x.shape[1], axis=-1)


x = np.array([
    [-6, 7, 10],
    [2, 3, 4]
])

r = softmax(x)
print(r)
print(np.sum(r, axis=-1))


def log_softmax(x):
    return np.log(softmax(x))


print(log_softmax(x))
exit()

fig, ax = plt.subplots()

# ax.plot(x, relu(x), label='ReLU')
# ax.plot(x, relu6(x), label='ReLU6')
# ax.plot(x, sigmoid(x), label='Sigmoid')
# ax.plot(x, hard_sigmoid(x), label='Hardsigmoid')
# ax.plot(x, tanh(x), label='Tanh')
ax.plot(x, hard_tanh(x, 5, -5), label='Hardtanh')

ax.set_title('ReLU')
ax.set_ylim(-7, 7)
ax.set_xlim(-7, 7)
ax.set_xlabel('Input')
ax.set_ylabel('Output')

plt.grid()
# plt.axis('equal')
plt.show()
