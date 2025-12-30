import torch

x = torch.linspace(-7, 7, 100)

# relu
# 1. 使用 nn 模块声明 relu 激活函数
# 返回一个函数 relu
# relu = torch.nn.ReLU()
# relu = torch.nn.ReLU(inplace=True)  # 使用就地操作
# print(relu(x))
# print(x)

# 2. 使用 functional 中的函数
print(torch.nn.functional.relu(x))
print(x)
# 就地操作
# torch.nn.functional.relu_(x)
# print(x)
