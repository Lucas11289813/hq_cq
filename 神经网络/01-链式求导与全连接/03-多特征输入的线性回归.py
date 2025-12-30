import torch
import matplotlib.pyplot as plt

fig = plt.figure()

x1 = torch.linspace(0, 1, 20)
x2 = torch.linspace(0, 1, 20)
y = 3 * x1 + 2 * x2 + 1
y += torch.normal(0, 0.2, y.shape)

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot(x1, x2, y, 'ro-')

w1 = torch.tensor(0.1, requires_grad=True)
w2 = torch.tensor(0.1, requires_grad=True)
b = torch.tensor(0.2)
b.requires_grad = True
lr = 0.05

y_predict = w1 * x1 + w2 * x2 + b

epochs = 100
for epoch in range(epochs):
    y_predict = w1 * x1 + w2 * x2 + b
    loss = torch.mean((y_predict - y) ** 2)
    loss.backward()
    with torch.no_grad():
        slope_w1 = w1.grad
        slope_w2 = w2.grad
        slope_b = b.grad
        w1 -= lr * slope_w1
        w2 -= lr * slope_w2
        b -= lr * slope_b
        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()

    if epoch % 10 == 0:
        print(f"loss : {loss.item()}")

y_predict = w1 * x1 + w2 * x2 + b
ax2 = fig.add_subplot(122, projection="3d")
ax2.plot(x1.detach().numpy(), x2.detach().numpy(), y_predict.detach().numpy(), 'bv--')
plt.show()