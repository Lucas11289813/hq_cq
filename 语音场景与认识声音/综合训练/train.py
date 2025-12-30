import torch.optim
from torch import nn

from dataset import waveform, tokens, sample_rate
import torchaudio.transforms as T
from model import w2v2

# 超参数
EPOCH = 100
lr = 1e-3

# 优化器
optim = torch.optim.Adam(w2v2.parameters(), lr=lr)

# 损失函数
# blank: 词库中 - 对应的索引
# reduction: 应用到输出损失的操作 none: 不做操作 mean: 在批次维度上平均 sum: 在批次维度上求和
# zero_infinity: 是否当损失值为无穷时，让梯度为 0。主要发生在输入太短而无法与目标对齐时。
loss_fn = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)

# 音频提取特征
# 梅尔频率倒谱系数 转换器
transform = T.MFCC(
    sample_rate=sample_rate,
    # 变换器输出的个数，类似卷积的输出通道数
    # n_mfcc 不能大于 n_mels
    n_mfcc=23,
    # n_mels: 梅尔滤波器的个数
    melkwargs={'n_fft': 400, 'win_length': 256, 'hop_length': 256, 'n_mels': 23, 'power': 2}
)

# (N, T, features)
data = transform(waveform)
# 这里因为变换后的形状和音频输入 waveform 的形状 (N, T) 不同，所以需要变形
data = data.reshape(1, -1)
target_lengths = torch.tensor([len(tokens)])

w2v2.load_state_dict(torch.load('model.pt'))
w2v2.train()

total_loss = 0.
count = 0

for epoch in range(EPOCH):
    optim.zero_grad()
    # (N, T, C)
    emission, _ = w2v2(data, None)
    # 因为损失函数要求输入对数概率，所以此处使用 log_softmax 求对数概率
    emission = emission.log_softmax(-1)
    # 因为损失函数要求的输入形状为 (T, N, C)，T 输入长度(时间轴的分块) N 批次 C 分类数
    # 所以需要变形
    emission = emission.transpose(0, 1)
    input_lengths = torch.tensor([emission.shape[0]])
    # 求损失
    # emission: 模型预测的概率
    # tokens: 音频对应的文本标签
    # input_lengths: 输入给损失函数的数据长度，等于 emission 中 T 的大小
    # target_lengths: 对应 tokens 的长度
    loss = loss_fn(emission, torch.tensor([tokens]), input_lengths=input_lengths, target_lengths=target_lengths)
    total_loss += loss.item()
    count += 1
    loss.backward()
    optim.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(w2v2.state_dict(), 'model.pt')
