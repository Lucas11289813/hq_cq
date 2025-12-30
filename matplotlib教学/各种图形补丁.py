import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 各种补丁的继承关系图
# https://matplotlib.org/stable/api/patches_api.html


fig, ax = plt.subplots()

# 矩形
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib-patches-rectangle
rectangle = patches.Rectangle((0.1, 0.1), 0.5, 0.3, edgecolor='black', facecolor='green')
ax.add_patch(rectangle)

# 圆
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Circle.html#matplotlib-patches-circle
circle = patches.Circle((0.5, 0.5), 0.2, edgecolor='black', facecolor='blue')
ax.add_patch(circle)
ax.set_aspect('equal')

# 椭圆
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Ellipse.html#matplotlib-patches-ellipse
ellipse = patches.Ellipse((0.5, 0.5), 0.6, 0.3, edgecolor='black', facecolor='purple')
ax.add_patch(ellipse)

# 多边形
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html#matplotlib-patches-polygon
vertices = [(1, 1), (2, 3), (4, 3), (5, 1)]
polygon = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='orange')
ax.add_patch(polygon)
# ax.set_xlim(0, 6)
# ax.set_ylim(0, 4)

# 弧线
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Arc.html#matplotlib-patches-arc
arc = patches.Arc((0.5, 0.5), 0.4, 0.4, angle=0, theta1=0, theta2=180, edgecolor='black')
ax.add_patch(arc)

plt.show()
