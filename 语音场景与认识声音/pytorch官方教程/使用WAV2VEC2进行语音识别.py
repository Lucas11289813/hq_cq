# wav2vec 2.0 论文: https://arxiv.org/abs/2006.11477
# https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html
# 语音识别过程
# 1. 从音频波形中提取声学特征
# 2. 逐帧估计声学特征的类别
# 3. 从类概率序列生成假设

import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

#######################################
# 准备
#######################################

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SPEECH_FILE = '../Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'

#######################################
# 创建管道
#######################################

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)

print(model.__class__)

# torchaudio.load(SPEECH_FILE) 加载音频文件需要软件支持
# 运行 pip3 install soundfile 安装音频处理软件
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

#######################################
# 提取声学特征
#######################################

# 针对 ASR 任务进行微调的 Wav2Vec2 模型可以一步执行特征提取和分类，但为了教程的目的，我们在这里还展示了如何执行特征提取。
# 也就是说，官方的模型本来可以一步完成特征提取和分类，但是为了教学目的，这里单独编写特征提取代码作为演示
with torch.inference_mode():
    features, _ = model.extract_features(waveform)

# fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
# for i, feats in enumerate(features):
#     ax[i].imshow(feats[0].cpu(), interpolation="nearest")
#     ax[i].set_title(f"Feature from transformer layer {i + 1}")
#     ax[i].set_xlabel("Feature dimension")
#     ax[i].set_ylabel("Frame (time-axis)")
# fig.tight_layout()
# plt.show()

#######################################
# 特征分类
#######################################

with torch.inference_mode():
    # 模型可以一步到位执行特征提取和推理
    emission, _ = model(waveform)


# plt.imshow(emission[0].cpu().T, interpolation="nearest")
# plt.title("Classification result")
# plt.xlabel("Frame (time-axis)")
# plt.ylabel("Class")
# plt.tight_layout()
# print("Class labels.txt:", bundle.get_labels())
# plt.show()
# 我们可以看到，在整个时间线上，某些标签有强烈的迹象。

#######################################
# 生成转录本
#######################################

# 在本教程中，为了简单起见，我们将执行不依赖于此类外部组件的贪婪解码，并在每个时间步简单地选取最佳假设。因此，不使用上下文信息，并且只能生成一份转录本。
# 我们首先定义贪婪解码算法
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission):
        """Given a sequence emission over labels.txt, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        # emission 第一个维度是序列得时间维度，代表每帧
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        # 去重
        indices = torch.unique_consecutive(indices, dim=-1)
        # 将空白部分过滤掉
        indices = [i for i in indices if i != self.blank]
        # 将索引对应得字符拼接起来
        return "".join([self.labels[i] for i in indices])


# 现在创建解码器对象并解码转录本。
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

print(transcript)
