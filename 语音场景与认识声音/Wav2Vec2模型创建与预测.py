import torch
import torchaudio
from torchaudio.models import wav2vec2_model

wf1, sr1 = torchaudio.load('./audios/1688-142285-0007.wav')
wf2, sr2 = torchaudio.load('./audios/376967__kathid__goodbye-high-quality.mp3')

print(wf1.shape)
print(wf2.shape)

wf2 = wf2.sum(dim=0)

# 储存两个音频的张量
wfs = torch.empty(2, 121776)
wfs[0, :wf1.shape[1]] = wf1[0]
wfs[1] = wf2

# 调用工厂函数构造模型
# api: https://pytorch.org/audio/stable/generated/torchaudio.models.wav2vec2_model.html?highlight=wav2vec2_model#torchaudio.models.wav2vec2_model
# 返回值 model 是 Wav2Vec2Model 类型的实例
model = wav2vec2_model(
    # 归一化方案，group_norm: 代表第一次卷积后给整体数据进行归一化，layer_norm: 代表使用层归一化
    extractor_mode='layer_norm',
    # 提取特征的卷积配置
    extractor_conv_layer_config=None,
    # 卷积是否采用偏置
    extractor_conv_bias=False,
    # 模型会根据此维度预测对应的概率个数，例如此处填1000，则代表词库中有1000个token对应模型输出
    encoder_embed_dim=1000,
    # 编码器全连接后的正则化概率
    encoder_projection_dropout=0.,
    # 卷积核大小
    encoder_pos_conv_kernel=3,
    # encoder_pos_conv_groups 必须整除 encoder_embed_dim
    encoder_pos_conv_groups=10,
    # 编码器的层数
    encoder_num_layers=3,
    # 多头注意力头数，头数必须整除 encoder_embed_dim 参数
    encoder_num_heads=4,
    # 编码器注意力的正则化
    encoder_attention_dropout=0.,
    # 前馈层中隐藏特征的维度
    encoder_ff_interm_features=1024,
    # 前馈神经网络的正则化概率
    encoder_ff_interm_dropout=0.,
    # 编码器的正则化
    encoder_dropout=0.,
    # 是否先归一化再执行注意力
    encoder_layer_norm_first=True,
    # 在训练期间丢弃每个编码器层的概率
    encoder_layer_drop=0.,
    # 用于微调的全连接层
    aux_num_out=None
)

# model 是 Wav2Vec2Model 模型实例
# api: https://pytorch.org/audio/stable/generated/torchaudio.models.Wav2Vec2Model.html#torchaudio.models.Wav2Vec2Model
# result: 输入文本概率分布
# lengths: 若音频时长不同时，代表每个音频的有效时长
result, lengths = model(
    # 音频数据
    wfs,
    # 当波形包含不同时长的音频时，可以设置此参数
    torch.tensor([112960, 121776])
)

print(result)
print(result.shape)

print(lengths)
