import torch


# 暂退法实现原理
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 0:
        return X
    if dropout == 1:
        return torch.zeros(X.shape)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1 - dropout)


X = torch.arange(0, 16, 1).reshape(4, 4)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
