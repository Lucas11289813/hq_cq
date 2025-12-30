from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QMessageBox
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
import sys


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_init()
        self.connect_signals()

    # 信号连接槽函数
    def connect_signals(self):
        self.reg_btn.clicked.connect(self.reg_btn_slot)
        self.clear_btn.clicked.connect(self.clear_btn_slot)

    # 界面
    def ui_init(self):
        self.setWindowTitle('用户注册')
        self.resize(400, 300)
        # 标签
        self.phone_lab = QLabel("手机号", self)
        self.phone_lab.move(50, 50)
        self.pwd_lab = QLabel("密码", self)
        self.pwd_lab.move(50, 100)
        # 单行文本框
        self.phone_edit = QLineEdit(self)
        self.phone_edit.setGeometry(150, 50, 200, 30)
        self.phone_edit.setPlaceholderText("请输入正确的手机号")
        # 定义正则表达式
        phone_regexp = QRegExp("1[3-9]\d{9}")
        # 创建正则表达式验证器
        validator = QRegExpValidator(phone_regexp)
        # 给输入框添加验证器，进行实时验证
        self.phone_edit.setValidator(validator)
        self.pwd_edit = QLineEdit(self)
        self.pwd_edit.setGeometry(150, 100, 200, 30)
        self.pwd_edit.setPlaceholderText("请输入6位数字密码")
        # 密码框密文显示
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        # 按钮
        self.reg_btn = QPushButton("注册", self)
        self.reg_btn.move(50, 200)
        self.clear_btn = QPushButton("清空", self)
        self.clear_btn.move(250, 200)

    # 注册按钮槽函数
    def reg_btn_slot(self):
        # 获取单行文本框信息
        phone = self.phone_edit.text()
        pwd = self.pwd_edit.text()
        # 输入格式验证
        # 1 3~9 9个数字 1[^012]\d{9}
        # qq号码 [^0]\d{4,9}
        phone_regexp = QRegExp("1[^012]\d{9}")
        pwd_regexp = QRegExp("\d{6}")
        if phone_regexp.exactMatch(phone):
            if pwd_regexp.exactMatch(pwd):
                QMessageBox.warning(self, "提示", "注册成功")
                # 保存信息到文件中
                with open("userinfo.txt", "a", encoding="utf-8") as file:
                    info = f"手机:{phone}\t密码:{pwd}\n"
                    file.write(info)
            else:
                QMessageBox.warning(self, "提示", "密码格式有误")
        else:
            QMessageBox.warning(self, "提示", "手机号格式有误")

    # 清空按钮槽函数
    def clear_btn_slot(self):
        self.phone_edit.clear()
        self.pwd_edit.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()
