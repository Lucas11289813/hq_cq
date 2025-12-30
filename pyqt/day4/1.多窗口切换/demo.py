import sys
from PyQt5.QtWidgets import QApplication, QWidget
from MainWindow import Ui_MainWindow
from SubWindow import Ui_SubWindow
# 如果界面类名相同，可以通过"文件名.类名"区分
import MainWindow
import SubWindow

# 如果有多个窗口，需要创建多个类
# 主窗口
class MainWindow(QWidget, MainWindow.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.connect_signals()

    def subWinShow_slot(self):
        # 显示子窗口
        sub_window.show()
        # 关闭自身
        self.close()

    def connect_signals(self):
        self.pushButton.clicked.connect(self.subWinShow_slot)

# 子窗口
class SubWindow(QWidget, Ui_SubWindow):
    def __init__(self):
        super(SubWindow, self).__init__()
        self.setupUi(self)
        self.connect_signals()

    def mainWinShow_slot(self):
        # 显示主窗口
        main_window.show()
        # 关闭自身
        self.close()

    def connect_signals(self):
        self.pushButton.clicked.connect(self.mainWinShow_slot)

# 程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 创建所有的窗口
    main_window = MainWindow()
    sub_window = SubWindow()
    # 只显示其中一个窗口
    main_window.show()
    app.exec()
