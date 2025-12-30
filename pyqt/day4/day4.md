# 1. 多窗口切换

通过QtDesinger设计多个界面分别保存，转换为py文件。

在主程序中同时创建这些界面py文件类

```python
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

```



# 2. 窗口切换传参

**核心知识点：自定义信号、发送信号**

* 自定义信号
  * 创建一个`pyqtSignal(参数1类型,参数2类型...)`对象
  * 创建对象的时候可以定义参数的数据类型，如`my_signal=pyqtSignal(str,int)`
* 发送信号
  * 通过`自定义信号对象.emit(参数1,参数2...)`发送信号
* 接收信号
  * `发送者.自定义信号名.connect(接收函数)`
  * 定义函数接收`def 函数名(self,传递的参数1,传递的参数2...)`

```python
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
    # 注意创建对象的顺序：先创建被弹出的窗口
    login_window = LoginWindow()
    login_window.show()
    main_window = MainWindow()
    app.exec()

```



# 员工管理系统

使用PyQt+Json形式保存数据。

## 窗口

* 主窗口QMainWindow
  * 展示数据
* 添加窗口QWidget
  * 输入信息
* 编辑窗口QDialog
  * 输入输出信息

## 核心功能

* 启动时，读取json文件，将数据打印在主窗口中。

* 添加
  * 主窗口通过点击添加，弹出添加窗口。
  * 在添加窗口中自定义信号发送添加的信息到主窗口。
* 删除
  * 主窗口中选择某条数据后点击删除
  * 使用列表删除功能

* 修改
  * 主窗口中通过选择数据后点击编辑，弹出修改窗口。
  * 创建修改窗口的同时，将所选行的数据传递给修改窗口类中。

## 实现过程

### 1.设计主窗口和添加窗口、读取原始数据、主窗口中弹出添加窗口

```python


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
 


# 创建添加窗口类

class InsertWindow(QWidget, Ui_insertWindow):
    def __init__(self):
        super(InsertWindow, self).__init__()
        self.setupUi(self)
 


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 注意创建对象的顺序：先创建被弹出的窗口
    insert_window = InsertWindow()
    main_window = MainWindow()
    main_window.show()
    app.exec()

```

### 2.在添加窗口中输入数据通过自定义信号发送参数

```python

# 创建主窗口类
class MainWindow(QMainWindow, Ui_MainWindow):

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




    def connect_signals(self):
        self.insertEmpAction.triggered.connect(self.showInsertWindow_slot)
        self.deleteAction.triggered.connect(self.deleteEmp_slot)
        # 获取添加窗口信号
        insert_window.insert_emp_signal.connect(self.getEmpInfo_slot)
       


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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 注意创建对象的顺序：先创建被弹出的窗口
    insert_window = InsertWindow()
    main_window = MainWindow()
    main_window.show()
    app.exec()
```

### 3.删除

```python
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

```

### 4.修改

修改可以是通过双击单元格进行行内编辑，也可以通过修改按钮，先读取数据后再修改。

* 行内编辑适合字段较少时修改，尤其是修改字符串、数值时。
* 如果字段多或用了单选、多选、组合框等情况，最好先弹出修改界面，再编辑修改。

这里使用先弹窗后修改。所以需要设计一个修改窗口。

主窗口

```python
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
```

修改窗口

```python
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
```

## 完整代码
```json

[
  {
    "id": "10001",
    "name": "\u738b\u6d77",
    "phone": "15266235478",
    "dept": "\u5e02\u573a\u90e8"
  },
  {
    "id": "10002",
    "name": "\u5f20\u94ed",
    "phone": "15247662358",
    "dept": "\u8fd0\u7ef4\u90e8"
  },
  {
    "id": "10003",
    "name": "\u8d75\u5b87",
    "phone": "15662352478",
    "dept": "\u7814\u53d1\u90e8"
  },
  {
    "id": "10004",
    "name": "\u5218\u6d9b",
    "phone": "15452662378",
    "dept": "\u5e02\u573a\u90e8"
  },
  {
    "id": "10005",
    "name": "aaaa",
    "phone": "bbbb",
    "dept": "\u5e02\u573a\u90e8"
  }
]
```
```python

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

```


# 作业

商品管理

* 商品：编号、名称、价格、库存...
* 上架、下架、修改商品

* 可选：购买功能
  * 选择某件商品，点击购买按钮，弹出窗口，输入数量，输出购买信息



客房管理

* 客房：房间号、类型、价格
* 添加客房、删除、修改客房信息
* 可选：入住、退房(修改客房状态)



宠物管理

* 宠物：品种、昵称、颜色...
* 添加、删除、修改宠物信息
* 可选：购买宠物、照看宠物...





...



