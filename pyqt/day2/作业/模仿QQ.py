import sys

from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QPushButton
from PyQt5.QtGui import QRegExpValidator, QPixmap, QPalette, QBrush

from QQForm import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 修改输入框样式
        edit_qss = """
            border-radius:10px;
            font-size:20px;
        """
        self.qqEdit.setStyleSheet(edit_qss)
        self.pwdEdit.setStyleSheet(edit_qss)
        logBtn_qss = """
            border-radius:10px;
            background-color:#9ed6ff;
            color:#fff;
            font-size:18px;
        """
        self.logBtn.setStyleSheet(logBtn_qss)
        # 实现图片圆形显示
        self.headWidget.setStyleSheet("border-image:url(head.png);border-radius:50px;")
        # 定义正则表达式
        qq_regexp = QRegExp("[1-9]\d{4,9}")
        # 创建正则表达式验证器
        validator = QRegExpValidator(qq_regexp)
        # 给输入框添加验证器，进行实时验证
        self.qqEdit.setValidator(validator)

        back_img = QPixmap("background.jpg")
        # 创建调色板
        palette = QPalette()
        # 设置调色板背景为指定图片
        palette.setBrush(QPalette.Background, QBrush(back_img.scaled(self.size())))
        # 设置窗口的调色板
        self.setPalette(palette)
        # 去除默认标题栏
        self.setWindowFlags(Qt.CustomizeWindowHint)
        # 固定大小
        self.setFixedSize(self.size())
        # 退出按钮
        self.exitBtn = QPushButton("x", self)
        self.exitBtn.setGeometry(self.width() - 30, 10, 20, 20)
        self.exitBtn.setStyleSheet("border:none")
        self.exitBtn.clicked.connect(lambda: app.exit())

    def logBtn_slot(self):
        qq = self.qqEdit.text()
        pwd = self.pwdEdit.text()
        if qq == "123456" and pwd == "123qwe":
            pass
            QMessageBox.information(self, "提示", "登录成功")
        else:
            QMessageBox.warning(self, "提示", "密码错误")

    def connect_signals(self):
        self.logBtn.clicked.connect(self.logBtn_slot)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
