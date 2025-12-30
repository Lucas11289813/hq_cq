from torchaudio.models import wav2vec2_model
from dataset import vocab

w2v2 = wav2vec2_model(
    extractor_mode='layer_norm',
    extractor_conv_layer_config=None,
    extractor_conv_bias=False,
    # 模型会根据此维度预测对应的概率个数，例如此处填1000，则代表词库中有1000个token对应模型输出
    encoder_embed_dim=len(vocab),
    encoder_projection_dropout=0.,
    encoder_pos_conv_kernel=3,
    # encoder_pos_conv_groups 必须整除 encoder_embed_dim
    encoder_pos_conv_groups=3,
    encoder_num_layers=3,
    # 多头注意力头数，头数必须整除 encoder_embed_dim 参数
    encoder_num_heads=3,
    encoder_attention_dropout=0.,
    encoder_ff_interm_features=1024,
    encoder_ff_interm_dropout=0.,
    encoder_dropout=0.,
    encoder_layer_norm_first=True,
    encoder_layer_drop=0.,
    aux_num_out=None
)
