import torch

classes = ["苹果", "香蕉", "芭蕉", "西瓜"]
# 输出四种类别的得分（苹果、香蕉、芭蕉、西瓜）
out = torch.tensor([[0.1, 3, 5, -5]])


def softmax(out):
    return torch.exp(out) / torch.sum(torch.exp(out))


result = softmax(out)
# 取出最大数值的下标位置
index = torch.argmax(result, dim=-1)
print(classes[index[0]])
