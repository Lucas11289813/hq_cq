import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 3, 20)
y = torch.sin(x)

x_m = x[::2]
y_m = y[::2]
print(len(x), len(x_m))

# --- 训练模型 ---
# 思考：如何定义一个百分比，抽取里面的数据，比如我只要原来样本中的60%
# 有点：1、减少了训练量  2、可以一定程度避免局部最小值。

plt.plot(x, y, 'r-')
plt.plot(x_m, y_m, 'b--')
plt.show()
