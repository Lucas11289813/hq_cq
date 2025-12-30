import sys

from PyQt5.QtCore import QFile, QFileInfo
from PyQt5.QtGui import QColor

from MyForm import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QColorDialog, QFontDialog, QFileDialog, QMessageBox


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        # 设置文本输入框填满整个窗口
        self.setCentralWidget(self.textEdit)
        # 定义默认的标题名
        self.file_name = "无标题-记事本"
        self.setWindowTitle(self.file_name)
        # 记录当前文本
        self.current_text = self.textEdit.toPlainText()
        # 记录文件路径
        self.file_path = ""

        self.connect_signals()

    # 重写QMainWindow的关闭事件
    def closeEvent(self, a0):
        self.newAction_slot()

    # 自定义颜色选择器
    def colorAction_slot(self):
        # 创建一个颜色选择器
        color = QColorDialog.getColor(QColor(), self)
        self.textEdit.setTextColor(color)

    # 自定义字体选择器
    def fontAction_slot(self):
        font = QFontDialog.getFont(self)[0]
        self.textEdit.setCurrentFont(font)

    # 修改文件标题
    def update_title_slot(self):
        # 检测到当前内容和一开始打开的内容不同时
        if self.current_text != self.textEdit.toPlainText():
            # 修改标题
            self.setWindowTitle("*" + self.file_name)
        else:
            self.setWindowTitle(self.file_name)

    # 另存为
    def saveAsAction_slot(self):
        # 打开保存对话框
        self.file_path = QFileDialog.getSaveFileName(self, "另存为", "./", "*.txt")[0]
        # 检测到点击了保存按钮
        if len(self.file_path) > 0:
            self.saveFile(self.file_path)

    # 保存
    def saveAction_slot(self):
        # 第一次保存时，调用另存为
        if self.file_name == "无标题-记事本":
            self.saveAsAction_slot()
        else:  # 之后保存时，覆盖当前文件
            self.saveFile(self.file_path)

    # 提取保存和另存为中的公共代码
    def saveFile(self, file_path):
        file = QFile(file_path)
        file.open(QFile.WriteOnly)
        if file.isOpen():
            # 获取当前文本框中的内容
            self.current_text = self.textEdit.toPlainText()
            # 写入到文件
            file.write(self.current_text.encode("utf-8"))
            # 修改窗口标题
            self.file_name = QFileInfo(self.file_path).fileName()
            self.setWindowTitle(self.file_name)
            # 关闭
            file.close()

    # 打开文件
    def openAction_slot(self):
        # 弹出读取文件对话框，选择文件，得到文件路径
        self.file_path = QFileDialog.getOpenFileName(self, "打开", "./", "*.txt")[0]
        # 打开文件
        file = QFile(self.file_path)
        file.open(QFile.ReadOnly)
        if file.isOpen():
            # 读取文件内容
            data = file.readAll()
            # 以保存时的编码格式转换
            text = bytearray(data).decode("utf-8")
            # 填充到文本框中
            self.textEdit.setText(text)
            # 修改窗口标题
            self.file_name = QFileInfo(self.file_path).fileName()
            self.setWindowTitle(self.file_name)
            # 更新当前内容
            self.current_text = self.textEdit.toPlainText()
            # 关闭文件
            file.close()

    # 清空内容
    def clearEdit(self):
        self.textEdit.clear()
        self.file_name = "无标题-记事本"
        self.current_text = self.textEdit.toPlainText()
        self.setWindowTitle(self.file_name)

    # 新建文件
    def newAction_slot(self):
        # 新建时判断当前内容是否有改动，
        if self.current_text != self.textEdit.toPlainText():
            # 如果有改动，提示是否要保存
            # 自定义消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("请确认")
            msg_box.setText("要保存当前内容吗")
            msg_box.setIcon(QMessageBox.Information)
            yes_btn = msg_box.addButton("是", QMessageBox.YesRole)
            no_btn = msg_box.addButton("否", QMessageBox.NoRole)
            msg_box.exec()
            if msg_box.clickedButton() == yes_btn:
                # 调用保存
                self.saveAction_slot()
                # 清空内容
                self.clearEdit()
            elif msg_box.clickedButton() == no_btn:
                self.clearEdit()
            # res = QMessageBox.question(self, "请确认", "要保存当前内容吗",
            #                            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            # if res == QMessageBox.Yes:  # 保存
            #     # 调用保存
            #     self.saveAction_slot()
            #     # 清空内容
            #     self.clearEdit()
            # elif res == QMessageBox.No:  # 不保存
            #     # 清空内容
            #     self.clearEdit()
        else:
            self.clearEdit()

    def connect_signals(self):
        # 给剪切按钮设置trigger信号，表示被点击时执行，执行textEdit组件自带的cut函数
        self.cutAction.triggered.connect(self.textEdit.cut)
        self.copyAction.triggered.connect(self.textEdit.copy)
        self.pasteAction.triggered.connect(self.textEdit.paste)
        self.redoAction.triggered.connect(self.textEdit.redo)
        self.undoAction.triggered.connect(self.textEdit.undo)
        self.colorAction.triggered.connect(self.colorAction_slot)
        self.fontAction.triggered.connect(self.fontAction_slot)
        # 检测到文本有变化
        self.textEdit.textChanged.connect(self.update_title_slot)
        # 点击了另存为
        self.saveAsAction.triggered.connect(self.saveAsAction_slot)
        # 点击了保存
        self.saveAction.triggered.connect(self.saveAction_slot)
        # 打开
        self.openAction.triggered.connect(self.openAction_slot)
        # 新建
        self.newAction.triggered.connect(self.newAction_slot)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()
