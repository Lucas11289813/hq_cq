from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QTimer, QDateTime
import sys


# 创建一个类，继承QMainWindow
class MyWindow(QMainWindow):
    # 构造函数
    def __init__(self):
        super().__init__()
        # 通常在构造函数中进行界面初始化
        self.ui_init()
        # 调用业务代码
        self.connect_slot()

    # 如果界面中的组件比较多，最好创建一个函数定义
    def ui_init(self):
        # 此时的self就是当前窗口
        self.setWindowTitle('计时器Timer')
        self.resize(400, 400)
        # 添加一个计时器到窗口中
        self.timer = QTimer(self)
        # 设置计时器的时间间隔,这里1000表示毫秒
        # 表示每隔1秒触发一次计时器
        self.timer.start(1000)
        # 计时器的信号通常为timeout表示到时
        # timer.timeout.connect(lambda: print("计时器触发。。。"))
        self.label = QLabel(self)
        self.label.resize(300, 20)

    # 如果某个信号触发时执行的内容比较多，最好创建一个函数
    def timer_slot(self):
        # 获取当前时间，打印在窗口中
        now = QDateTime.currentDateTime()
        self.label.setText(now.toString("当前时间：yyyy/MM/dd HH:mm:ss"))

    # 通常定义一个函数，表示某个组件触发某个信号后执行指定的槽函数
    def connect_slot(self):
        # 信号触发时调用指定的槽函数，注意只需写函数名
        self.timer.timeout.connect(self.timer_slot)


if __name__ == '__main__':
    # 创建应用程序
    app = QApplication(sys.argv)
    # 创建自定义窗口
    window = MyWindow()
    window.show()
    app.exec()
