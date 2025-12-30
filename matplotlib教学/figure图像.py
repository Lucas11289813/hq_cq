import matplotlib
import matplotlib.pyplot as plt

# 指定后端
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['YouYuan']

# 创建一个图窗
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure
# fig = plt.figure(
#     # 图窗 id，没有就创建，有就查找并激活，若查找不到则创建
#     num=1,
#     # 窗口大小
#     figsize=(4, 2),
#     # 颜色
#     facecolor='lightskyblue',
#     # 分辨率 代表每英寸显示多少像素
#     dpi=200,
#     # 布局
#     layout='constrained'
# )
#
# # 添加图窗标题
# fig.suptitle('A nice Matplotlib Figure')
# # 添加一幅图
# ax = fig.add_subplot()
# ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')


# 多合一布局
# 这里的两个参数分别代表创建几行几列的图画
# 这里的 axs 是 2x3 个 Axes 对象
# fig, axs = plt.subplots(2, 3)
# # 通过索引设置对应图像
# for i in range(axs.shape[0]):
#     for j in range(axs.shape[1]):
#         axs[i, j].text(0.5, 0.5, f'{i + 1}行 {j + 1}列', ha='center', va='center')

# 复杂网格布局
# 此处 axs 是一个包含名字和 Axes 对象的字典
# fig, axs = plt.subplot_mosaic([
#     # 此处指定行列单元格
#     ['A', 'D', 'C'],
#     ['A', 'D', 'C'],
#     ['B', 'B', 'C'],
# ])
#
# fig.suptitle('图窗')
#
# for ax_name, ax in axs.items():
#     ax.set_title(ax_name)
#     ax.text(0.5, 0.5, ax_name, ha='center', va='center')


# 嵌套图窗
fig = plt.figure()
# 创建一行两列的子图窗
children = fig.subfigures(1, 2)
figL, figR = children
# 设置左右图窗
figL.set_facecolor('#ff0000')
figL.suptitle('figL')
# 创建两个 Axes
axl = figL.subplots(1, 2)
figR.set_facecolor('#0000ff')
figR.suptitle('figR')
axr = figR.subplots(2, 1)

# 保存 并指定分辨率
fig.savefig('my_save.png', dpi=300)

plt.show()
