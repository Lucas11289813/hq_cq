import sys
from math import ceil

from PyQt5.QtCore import Qt, QFileInfo, QUrl, QTimer, QTime
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3

from MyForm import Ui_MainWindow
import os


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        # 去掉标题栏
        self.setWindowFlags(Qt.CustomizeWindowHint)
        # 固定大小
        self.setFixedSize(self.size())
        # 加载图片
        self.img_init()
        # 默认加载第一页
        self.stackedWidget.setCurrentIndex(0)
        # 隐藏控制器
        self.bottomFrame.hide()
        # 歌曲列表
        self.musicFileList = []
        # 定义播放器
        self.player = QMediaPlayer(self)
        # 当前歌曲播放状态
        self.isPlay = False
        # 定义计时器
        self.timer = QTimer(self)
        # 当前播放的歌曲
        self.currentMusic = None
        # 当前信息文字索引
        self.song_info_word_index = 0
        # 核心业务
        self.connect_signal()

    def minBtn_slot(self):
        self.showMinimized()

    def exitBtn_slot(self):
        res = QMessageBox.warning(self, "提示", "确认要退出吗", QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            self.close()

    def img_init(self):
        self.logoLab.setPixmap(QPixmap('imgs/logo.png').scaled(self.logoLab.size()))
        self.lab1.setPixmap(QPixmap('imgs/rnb.jpg').scaled(self.lab1.size()))
        self.lab2.setPixmap(QPixmap('imgs/hiphop.jpg').scaled(self.lab2.size()))
        self.lab3.setPixmap(QPixmap('imgs/rock.jpg').scaled(self.lab3.size()))
        self.lab4.setPixmap(QPixmap('imgs/pop1.jpg').scaled(self.lab4.size()))

    # 切换页面
    def myMusicBtn_slot(self):
        # 修改样式
        self.myMusicBtn.setStyleSheet("background-color:#fc3d49;color:#fff;")
        self.casualBtn.setStyleSheet("")
        # 修改stackedWidget组件的索引
        self.stackedWidget.setCurrentIndex(1)

    def casualBtn_slot(self):
        # 修改样式
        self.casualBtn.setStyleSheet("background-color:#fc3d49;color:#fff;")
        self.myMusicBtn.setStyleSheet("")
        # 修改stackedWidget组件的索引
        self.stackedWidget.setCurrentIndex(0)

    # 选择目录
    def chooseDirLink_slot(self):
        # 选择目录进行遍历
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录")
        if dir_path:
            index = 0
            # 遍历目录  获取.mp3文件
            for root, dirs, files in os.walk(dir_path):
                # 遍历每个子文件
                for file in files:
                    # 只获取.mp3文件
                    if file.endswith(".mp3"):
                        # 构建完整路径
                        file_path = os.path.join(root, file)
                        # 以文件的形式打开
                        music_file = QFileInfo(file_path)
                        # 保存到集合中
                        self.musicFileList.append(music_file)
                        # 获取歌曲的信息
                        audio = MP3(file_path, ID3=EasyID3)
                        title = audio.get("title", music_file.baseName())
                        artist = audio.get("artist", "[未知歌手]")
                        album = audio.get("album", '[未知专辑]')
                        length = ceil(audio.info.length)
                        duration = f"{length // 60:02}:{length % 60:02}"
                        index += 1
                        song_info = f"   {index}\t🎵  {title}\n\t{artist}\t{album}\t\t{duration}"
                        # 将信息添加到ListWidget组件中
                        self.listWidget.addItem(song_info)

    # 双击选择歌曲
    def chooseMusic_slot(self):
        # 获取被选中项的索引
        index = self.listWidget.currentRow()
        # 得到对应的音乐文件
        file = self.musicFileList[index]
        # 更新当前歌曲
        self.currentMusic = file
        # 设置当前歌曲信息要显示的起始索引
        self.song_info_word_index = 0
        # 通过播放器播放
        self.player.setMedia(QMediaContent(QUrl(file.absoluteFilePath())))
        # 播放
        self.player.play()
        # 读取播放的歌曲信息
        audio = MP3(file.absoluteFilePath(), ID3=EasyID3)
        title = audio.get("title", file.baseName())
        artist = audio.get("artist", "[未知歌手]")
        length = ceil(audio.info.length)
        duration = f"{length // 60:02}:{length % 60:02}"
        # 显示歌曲信息
        self.songInfoLab.setText(f"{title}--{artist}")
        # 显示时长
        self.durationLab.setText(duration)
        # 启动计时器
        self.timer.start(1000)
        # 修改歌曲播放状态
        self.isPlay = False
        # 点击播放按钮
        self.playBtn.click()
        # 修改进度条的最大值
        self.playerSlider.setRange(0, length * 1000)
        # 显示控制器
        self.bottomFrame.show()

    def timer_slot(self):
        # 获取当前播放器的进度,单位为毫秒
        pos = self.player.position()
        # 从00:00开始添加当前进度，将最新进度显示在label上
        pos_str = QTime(0, 0, 0, 0).addMSecs(pos).toString("mm:ss")
        self.timeLab.setText(pos_str)
        # 歌曲信息滑动显示
        audio = MP3(self.currentMusic.absoluteFilePath(), ID3=EasyID3)
        title = audio.get("title", self.currentMusic.baseName())
        artist = audio.get("artist", "[未知歌手]")
        info = f"{title}--{artist}"
        self.songInfoLab.setText(info[self.song_info_word_index:])
        self.song_info_word_index += 1

    # 播放按钮
    def playBtn_slot(self):
        if self.isPlay:
            self.player.pause()
            self.isPlay = False
            self.playBtn.setIcon(QIcon(":/res/imgs/icon/bofang.png"))
        else:
            self.player.play()
            self.isPlay = True
            self.playBtn.setIcon(QIcon(":/res/imgs/icon/zanting.png"))

    # 设置进度条的值
    def playerSlider_slot(self, pos):
        self.playerSlider.setValue(pos)

    # 设置当前播放器播放进度
    def player_set_position_slot(self, pos):
        self.player.setPosition(pos)

    # 设置音量
    def set_volume_slot(self, value):
        self.player.setVolume(value)
        if value == 0:
            self.volumeBtn.setIcon(QIcon(":/res/imgs/icon/jingyin.png"))
        elif value <= 50:
            self.volumeBtn.setIcon(QIcon(":/res/imgs/icon/yinliangjian.png"))
        else:
            self.volumeBtn.setIcon(QIcon(":/res/imgs/icon/yinliangjia.png"))

    def connect_signal(self):
        self.minBtn.clicked.connect(self.minBtn_slot)
        self.exitBtn.clicked.connect(self.exitBtn_slot)
        self.myMusicBtn.clicked.connect(self.myMusicBtn_slot)
        self.casualBtn.clicked.connect(self.casualBtn_slot)
        self.chooseDirLink.clicked.connect(self.chooseDirLink_slot)
        # ListWidget中的每一项双击
        self.listWidget.itemDoubleClicked.connect(self.chooseMusic_slot)
        # 播放暂停
        self.playBtn.clicked.connect(self.playBtn_slot)
        # 计时器
        self.timer.timeout.connect(self.timer_slot)
        # 当播放器当前歌曲进度发生变化时
        self.player.positionChanged.connect(self.playerSlider_slot)
        # 当手动调整进度条时
        self.playerSlider.sliderMoved.connect(self.player_set_position_slot)
        self.volumeSlider.sliderMoved.connect(self.set_volume_slot)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()
