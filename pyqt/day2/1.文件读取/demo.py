import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from MyForm import Ui_MainWindow
from torchvision.transforms import Resize, Compose, ToTensor
from PIL import Image
from FruitNN import MyNet
import torch


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.connect_signal()

    def chooseImgBtn_slot(self):
        # 打开文件对话框
        file_dialog = QFileDialog()
        # 限制打开的文件类型
        # 返回值为一个元组，包含了文件完整路径和限定格式
        res = file_dialog.getOpenFileName(filter="*.jpg *.png *.jpeg")
        # 这里只取路径部分
        file_path = res[0]
        if len(res) > 0:
            # 使用图片组件
            img = QPixmap(file_path)
            # 显示图片
            self.imgLab.setPixmap(img.scaled(self.imgLab.size()))
            # 使用Image打开图片
            image = Image.open(file_path)
            self.predict_slot(image)
        else:
            QMessageBox.warning(self, "提示", "请选择图片")

    def predict_slot(self, image):
        # 由于模型需要3通道图片，这里将图片转换为RGB模式
        image = image.convert("RGB")
        # 定义转换器，将图片转换为[1,3,100,100]形状的张量用于模型验证识别
        transformer = Compose(
            [
                Resize((100, 100)),
                ToTensor()
            ]
        )
        # 添加批次维度
        img_tensor = transformer(image).unsqueeze(0)
        # 创建模型对象
        model = MyNet()
        # 加载模型参数
        model.load_state_dict(torch.load("model_param.pt"))
        # 调用模型预测
        outputs = model.forward(img_tensor)
        y = torch.argmax(outputs)
        fruit_map = {
            0: "香蕉",
            1: "桑葚",
            2: "火龙果"
        }
        res = fruit_map[int(y)]
        # 拼接预测的结果
        self.resLab.setText(self.resLab.text() + res)

    def connect_signal(self):
        self.chooseImgBtn.clicked.connect(self.chooseImgBtn_slot)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()
