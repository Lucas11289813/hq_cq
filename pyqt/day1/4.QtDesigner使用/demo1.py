from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap
from MyForm import Ui_MainWindow
import sys


# 1.导入通过QtDesinger设计的ui文件转换后的py文件
# 2.创建类，继承QMainWindow和自定义的窗体类
# 3.在构造函数中调用自定义窗体类中的初始化界面函数setupUi()

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        # 调用自定义模块中的初始化界面函数
        self.setupUi(self)
        # 调整按钮样式
        self.showPicBtn.setStyleSheet("color:red")
        self.connect_signals()

    def connect_signals(self):
        # 按钮点击
        self.showPicBtn.clicked.connect(self.showPicBtn_slot)

    def showPicBtn_slot(self):
        # QPixmap("图片路径")
        img = QPixmap("pic1.jpg")
        # 图片显示在label上,缩放与label一样尺寸
        self.picLab.setPixmap(img.scaled(self.picLab.size()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()
