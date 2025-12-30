import torch
import torch.nn.functional as F

from model_RNNCell import EncoderDecoderModel
from dataset import TranslateDataset

model = EncoderDecoderModel(9, 20, teacher_forcing_ratio=0)
model.load_state_dict(torch.load('weights/model_rnncell.pt', map_location='cpu', weights_only=True))
model.eval()

ds = TranslateDataset()

src, tgt = ds[0]
_tgt = F.one_hot(tgt, 9)

with torch.inference_mode():
    y, h = model(src, _tgt)

print(y.argmax(-1))
