from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
import sys

# 创建应用程序
app = QApplication(sys.argv)
# 在运行时，可以传递一些参数
# 如通过控制台命令python xxx.py 参数1 参数2... 运行时即可看到传递的参数
# print(sys.argv)
# 创建主窗口
window = QMainWindow()
# 设置窗口标题
window.setWindowTitle("主窗口")
# 调整窗口尺寸
window.resize(400, 400)
# 创建用于显示文字的标签
# label=QLabel(window)
# label.setText("hello pyqt!")
# 如果创建label时同时设置文本
QLabel("hello pyqt!", window)

# 显示主窗口
window.show()
# 应用程序启动
app.exec()
