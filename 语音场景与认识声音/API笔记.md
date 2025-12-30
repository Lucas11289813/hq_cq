# API 笔记

## 安装 torchaudio backend

使用 torchaudio.load 加载音频文件，需要先安装能够被加载的后端程序，例如: ffmpeg、soundfile 和 sox

其中 sox windows 不支持。

以下两种方式都可安装后端程序

```shell
# 安装 soundfile
pip3 install soundfile
# 安装 ffmpeg
conda install -c conda-forge "ffmpeg<7"
```

## torchaudio.load

https://pytorch.org/audio/stable/generated/torchaudio.load.html?highlight=load#torchaudio.load

从源加载音频数据

参数:

- uri: 音频路径
- frame_offset: 开始读取数据之前要跳过的帧数
- num_frames: 要读取的最大帧数。 -1 读取从 frame_offset 开始的所有剩余样本。如果给定文件中没有足够的帧，此函数可能会返回较少数量的帧。
- normalize: 归一化，若为 True 转换样本类型为 float32
- channels_first: 通道维度是否放在前面
- format: 可以覆盖后端程序检测到的格式信息
- buffer_size: 处理类文件对象时使用的缓冲区大小，以字节为单位。
- backend: 指定后端程序

返回值: 结果张量和采样率