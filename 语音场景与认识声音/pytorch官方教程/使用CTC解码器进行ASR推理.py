# CTC 是 "Connectionist Temporal Classification"（连接主义时序分类）的缩写。CTC 是一种用于序列建模的技术，最初被用于语音识别，但后来也被应用于其他序列标记任务，如手写识别和自然语言处理。
# ASR 是 "Automatic Speech Recognition"（自动语音识别）的缩写。
# 教程中的 ctc_decoder 使用了 KenLM 库，而安装该库需要使用 vs installer 安装 c c++ 开发套件
import torch
import torchaudio
import time
from typing import List
import matplotlib.pyplot as plt
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files, CTCDecoderLM, CTCDecoderLMState
from torchaudio.utils import download_asset

print(torch.__version__)
print(torchaudio.__version__)

###############################################
# 声学模型和设置
###############################################

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
acoustic_model = bundle.get_model()

# 加载音频样本
# 音频对应文本为
# i really was very much afraid of showing him how much shocked i was at some parts of what he said
speech_file = '../1688-142285-0007.wav'

print(speech_file)

waveform, sample_rate = torchaudio.load(speech_file)
print(waveform)
print(waveform.shape)

# 采样率若何预训练模型不同，则修改采样率
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

###############################################
# 解码器的文件和数据
###############################################

# 加载 tokens
tokens = [label.lower() for label in bundle.get_labels()]
print(tokens)


###############################################
# 语言模型
###############################################

class CustomLM(CTCDecoderLM):
    """Create a Python wrapper around `language_model` to feed to the decoder."""

    def __init__(self, language_model: torch.nn.Module):
        CTCDecoderLM.__init__(self)
        self.language_model = language_model
        self.sil = -1  # index for silent token in the language model
        self.states = {}

        language_model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        with torch.no_grad():
            score = self.language_model(self.sil)

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if outstate not in self.states:
            score = self.language_model(token_index)
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.sil)


# 下载预训练模型
files = download_pretrained_files("librispeech-4-gram")

print(files)

###############################################
# 构造解码器
###############################################

LM_WEIGHT = 3.23
WORD_SCORE = -0.26

# 波束搜索解码器
beam_search_decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=files.tokens,
    lm=files.lm,
    nbest=3,
    beam_size=1500,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
)


# 贪婪解码器
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels.txt, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        # 找到最大值索引
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        # 去掉连续重复的索引
        indices = torch.unique_consecutive(indices, dim=-1)
        # 去掉 blank 空白符
        indices = [i for i in indices if i != self.blank]
        # 将 token 转换回文本 然后拼接起来
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()


greedy_decoder = GreedyCTCDecoder(tokens)

###############################################
# 运行推理
###############################################

# 真实转录文本
actual_transcript = "i really was very much afraid of showing him how much shocked i was at some parts of what he said"
actual_transcript = actual_transcript.split()

# 预测结果
emission, _ = acoustic_model(waveform)

# 贪婪解码
greedy_result = greedy_decoder(emission[0])
greedy_transcript = " ".join(greedy_result)
# 计算预测结果和真实值的差距
# wer 是一个语音识别准确度的指标
greedy_wer = torchaudio.functional.edit_distance(actual_transcript, greedy_result) / len(actual_transcript)

print(f"Transcript: {greedy_transcript}")
print(f"WER: {greedy_wer}")

# 使用波束搜索解码器
beam_search_result = beam_search_decoder(emission)
# 将解码结果连接成字符串
beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
beam_search_wer = torchaudio.functional.edit_distance(actual_transcript, beam_search_result[0][0].words) / len(
    actual_transcript
)

print(f"Transcript: {beam_search_transcript}")
print(f"WER: {beam_search_wer}")

# 对比看出 采用 ctc_decoder 解码的结果更接近真实结果
