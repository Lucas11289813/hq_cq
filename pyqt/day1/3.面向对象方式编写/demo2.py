from PyQt5.QtWidgets import QApplication, QMainWindow, QCalendarWidget, QPushButton, QLabel
from PyQt5.QtCore import QTimer, QDateTime, QDate
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
        self.setWindowTitle('简易日历')
        self.resize(400, 400)
        # 添加一个计时器到窗口中
        self.timer = QTimer(self)
        # 设置计时器的时间间隔,这里1000表示毫秒
        # 表示每隔1秒触发一次计时器
        self.timer.start(1000)
        # 添加状态栏。任何窗口都有状态栏组件，所以无需额外创建添加
        self.status_bar = self.statusBar()
        # 创建日历组件
        self.calendar = QCalendarWidget(self)
        # 调整尺寸
        self.calendar.resize(self.width(), 300)
        # 添加按钮
        self.today_btn = QPushButton('回到今天', self)
        self.today_btn.move(self.width() - self.today_btn.width(),
                            self.height() - self.today_btn.height() - self.status_bar.height())
        # 添加显示日期标签
        self.date_lab = QLabel(self)
        self.date_lab.setGeometry(10,self.calendar.height()+30,self.width(),40)
        self.date_lab.lower()

    # 选择日历中的日期后，计算距今相隔的天数
    def choose_date_slot(self):
        # 获取选中的日期
        selected_date = self.calendar.selectedDate()
        days = QDate.currentDate().daysTo(selected_date)
        str = f"{selected_date.toString('yyyy/MM/dd')}距今相隔{abs(days)}天"
        self.date_lab.setText(str)

    # "回到今天"按钮的槽函数
    def today_btn_slot(self):
        self.calendar.setSelectedDate(QDate.currentDate())

    # 如果某个信号触发时执行的内容比较多，最好创建一个函数
    def timer_slot(self):
        # 获取当前时间，打印在窗口中
        now = QDateTime.currentDateTime()
        fmt_time = now.toString("当前时间：yyyy/MM/dd HH:mm:ss")
        self.status_bar.showMessage(fmt_time)

    # 通常定义一个函数，表示某个组件触发某个信号后执行指定的槽函数
    def connect_slot(self):
        # 信号触发时调用指定的槽函数，注意只需写函数名
        self.timer.timeout.connect(self.timer_slot)
        self.today_btn.clicked.connect(self.today_btn_slot)
        # 当日期改变时
        self.calendar.selectionChanged.connect(self.choose_date_slot)

if __name__ == '__main__':
    # 创建应用程序
    app = QApplication(sys.argv)
    # 创建自定义窗口
    window = MyWindow()
    window.show()
    app.exec()
