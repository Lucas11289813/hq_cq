from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHeaderView, QTableWidgetItem, QTableWidget, \
    QMessageBox, QDialog

from InsertWindow import Ui_insertWindow
from MainWindow import Ui_MainWindow
from UpdateWindow import Ui_UpdateWindow
import sys
import json


# 创建主窗口类
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # tableWidget自适应
        self.setCentralWidget(self.tableWidget)
        # 定义所有键的集合
        self.keys = ["id", "name", "phone", "dept"]
        # 定义员工对象集合
        self.empList = []
        # 默认选择整行
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        # 阻止双击修改
        self.tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.connect_signals()
        self.data_init()

    # 加载数据
    def data_init(self):
        # 设计tableWidget的表头、列数
        headers = ["编号", "姓名", "电话", "部门"]
        # 设置表格列数
        self.tableWidget.setColumnCount(len(headers))
        # 设置表头
        self.tableWidget.setHorizontalHeaderLabels(headers)
        # 表头自适应宽度
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 读取data.json文件
        with open("data.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            # 根据读取到的数据数量设置表格的行数
            self.tableWidget.setRowCount(len(data))
            # 将数据填充到tableWidget中
            # self.tableWidget.setItem(行索引,列索引,QTableWidgetItem(数据))
            # data[索引]表示指定索引对象
            # data[索引][键]表示指定索引对象的指定键的值
            # self.tableWidget.setItem(0, 0, QTableWidgetItem(data[0]["name"]))
            # 将读取到的数据赋值给empList
            self.empList = data
            # 外层循环行数,这里遍历读取到的数据
            for row, emp in enumerate(data):
                # 内层循环列数，这里读取某个emp对象的某个键
                for col, key in enumerate(self.keys):
                    # emp[key]指定某个键的值
                    item = QTableWidgetItem(emp[key])
                    self.tableWidget.setItem(row, col, item)

    def showInsertWindow_slot(self):
        insert_window.show()

    # 获取信号中的数据
    def getEmpInfo_slot(self, name, phone, dept):
        # 获取数据后，更新data.json，更新table的数据
        # 将添加的数据封装为一个emp对象
        # 这里的id可以读取最后一个emp的id值后+1
        emp = {
            "id": str(int(self.empList[-1].get("id")) + 1),
            "name": name,
            "phone": phone,
            "dept": dept
        }
        # 添加到empList中
        self.empList.append(emp)
        # 通过重新写json更新data.json,这里需要获取最新的员工集合
        with open("data.json", "w", encoding="utf-8") as file:
            json.dump(self.empList, file)
        # 通过调用data_init更新table的数据
        self.data_init()
        # 关闭添加窗口
        insert_window.close()

    # 删除
    def deleteEmp_slot(self):
        # 判断当前是否有行被选中
        if len(self.tableWidget.selectedItems()) > 0:
            res = QMessageBox.warning(self, "提示", "确认要删除吗", QMessageBox.Ok | QMessageBox.Cancel)
            if res == QMessageBox.Ok:
                # 获取被选中的索引，调用pop删除
                self.empList.pop(self.tableWidget.currentRow())
                # 更新data.json
                with open("data.json", "w", encoding="utf-8") as file:
                    json.dump(self.empList, file)
                # 通过调用data_init更新table的数据
                self.data_init()
        else:
            QMessageBox.warning(self, "提示", "请先选择数据")

    # 编辑
    def editEmp_slot(self):
        if len(self.tableWidget.selectedItems()) > 0:
            # 弹出编辑窗口，传入当前选中的行的数据
            emp = self.empList[self.tableWidget.currentRow()]
            # 创建编辑窗口对象，传入数据
            update_window = UpdateWindow(emp)
            # 弹出编辑窗口，如果点击了编辑窗口的按钮
            if update_window.exec() == QDialog.Accepted:
                # 获取修改后的所有数据，更新json，更新表格
                emp = update_window.getData()
                self.empList[self.tableWidget.currentRow()] = emp
                with open("data.json", "w", encoding="utf-8") as file:
                    json.dump(self.empList, file)
                self.data_init()
        else:
            QMessageBox.warning(self, "提示", "请先选择数据")

    def connect_signals(self):
        self.insertEmpAction.triggered.connect(self.showInsertWindow_slot)
        self.deleteAction.triggered.connect(self.deleteEmp_slot)
        # 获取添加窗口信号
        insert_window.insert_emp_signal.connect(self.getEmpInfo_slot)
        # 编辑动作
        self.editAction.triggered.connect(self.editEmp_slot)


# 创建添加窗口类

class InsertWindow(QWidget, Ui_insertWindow):
    # 定义信号
    insert_emp_signal = pyqtSignal(str, str, str)

    def __init__(self):
        super(InsertWindow, self).__init__()
        self.setupUi(self)
        self.connect_signals()

    # 获取员工信息
    def getEmpInfo_slot(self):
        # 文本框
        name = self.nameEdit.text()
        phone = self.phoneEdit.text()
        # 组合框
        dept = self.comboBox.currentText()
        # 发送信号
        self.insert_emp_signal.emit(name, phone, dept)

    def connect_signals(self):
        self.pushButton.clicked.connect(self.getEmpInfo_slot)


# 选择某一行后，点击修改，流程为：
# 1.先读取要修改的原始数据。这里有可能读到的一行数据会很多，可以通过创建窗口的同时传参
# 2.编辑
# 3.修改
# # 创建修改窗口类，继承QDialog(对话框)，可以监听是否点击了确认
class UpdateWindow(QDialog, Ui_UpdateWindow):
    # 修改窗口在点击修改按钮后创建，创建的同时传递被选中的数据参数
    def __init__(self, data):
        super(UpdateWindow, self).__init__()
        self.setupUi(self)
        # 获取传入的数据后显示
        self.data = data
        self.showEmpInfo_slot()
        self.connect_signals()

    # 显示所选数据到窗口中
    def showEmpInfo_slot(self):
        emp = self.data
        self.idLab.setText(emp["id"])
        self.nameEdit.setText(emp["name"])
        self.phoneEdit.setText(emp["phone"])
        self.comboBox.setCurrentText(emp["dept"])

    # 定义获取当前数据的函数，用于提交给主窗口
    def getData(self):
        return {
            "id": self.idLab.text(),
            "name": self.nameEdit.text(),
            "phone": self.phoneEdit.text(),
            "dept": self.comboBox.currentText()
        }

    def connect_signals(self):
        # 点击了修改按钮时，表示接收
        self.updateBtn.clicked.connect(self.accept)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 注意创建对象的顺序：先创建被弹出的窗口
    insert_window = InsertWindow()
    main_window = MainWindow()
    main_window.show()
    app.exec()
