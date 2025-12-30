from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QApplication
from LoginWindow import Ui_LoginWindow
from MainWindow import Ui_MainWindow
import sys


# 登录窗口需要将输入的数据发送到主窗口
# 使用自定义信号发送
class LoginWindow(QWidget, Ui_LoginWindow):
    # 在类中自定义信号
    # 如果要传递参数，在这里定义参数的类型
    my_signal = pyqtSignal(str, str)

    def __init__(self):
        super(LoginWindow, self).__init__()
        self.setupUi(self)
        self.connect_signal()

    def emitArgs_slot(self):
        # 发送自定义信号emit(参数1,参数2)
        # 接收输入的内容后发送
        username = self.usernameEdit.text()
        password = self.passwordEdit.text()
        self.my_signal.emit(username, password)
        # 关闭自身
        self.close()

    def connect_signal(self):
        self.pushButton.clicked.connect(self.emitArgs_slot)


# 主窗口需要接收信号
class MainWindow(QWidget, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.connect_signal()

    # 接收信号时，将传递的信号定义在函数的形参中
    def getArgs_slot(self, username, password):
        print("收到了登录窗口发送的信号")
        # 将收到的值打印在界面中
        self.label.setText(f"用户名：{username}\t密码：{password}")
        self.show()

    def connect_signal(self):
        # 监听登录窗口发送的信号
        # 发送者.信号.connect(xxx)
        # 按钮.clicked.connect()
        login_window.my_signal.connect(self.getArgs_slot)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 注意创建对象的顺序:先创建发送信号的窗口
    login_window = LoginWindow()
    main_window = MainWindow()
    login_window.show()
    app.exec()
