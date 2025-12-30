# 创建一个窗口，添加一个按钮
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
import sys

app = QApplication(sys.argv)
window = QMainWindow()
window.resize(400, 400)
# 创建按钮，添加到窗口中
button = QPushButton(window)
button.setText("按钮")
# # 调整尺寸resize(w,h)
# button.resize(100, 50)
# # 调整位置move(posx,posy)
# button.move(20, 200)

# 同时调整尺寸和位置setGeometry(posx,posy,w,h)
button.setGeometry(20, 150, 100, 50)
# 点击按钮时执行
# button.clicked.connect(lambda: print("按钮被点击"))

# 创建一个标签
label = QLabel(window)
label.move(200,150)
button.clicked.connect(lambda: label.setText("xxxx"))

window.show()
app.exec()
