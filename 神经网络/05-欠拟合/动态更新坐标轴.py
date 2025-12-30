import matplotlib.pyplot as plt
from random import random

x = [0]
loss = [1]
fig, ax = plt.subplots(1, 1)
line, = ax.plot(x, loss, 'ro-')

for i in range(1, 100):
    loss.append(random())
    if len(loss) > 8:
        loss.remove(loss[0])
    x.append(i)
    line.set_data(x, loss)

    ax.relim()
    ax.autoscale_view()
    plt.pause(0.5)
