import torch
from torch import nn
from torchtext.vocab import vocab
from collections import OrderedDict
from Transformer简单实现 import Transformer

orderedDict = OrderedDict({
    'how': 500,
    'are': 500,
    'you': 500,
    '?': 500
})

voc = vocab(orderedDict, min_freq=1, specials=['<pad>', '<unk>', '<sos>', '<eos>'], special_first=True)
voc.set_default_index(voc['<unk>'])

src = 'how are you <pad> <pad> <pad>'.split(' ')
tgt = '<sos> how are you ? <eos>'.split(' ')

src_idx = torch.tensor(voc.lookup_indices(src)).unsqueeze(0)
tgt_idx = torch.tensor(voc.lookup_indices(tgt)).unsqueeze(0)

src_key_padding_mask = torch.zeros_like(src_idx, dtype=torch.float)
src_key_padding_mask[src_idx == voc['<pad>']] = float('-inf')

model = Transformer(len(voc), 100, 2, 3, voc['<pad>'])
model.train()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

label = tgt_idx.reshape(-1)

for epoch in range(100):
    optimizer.zero_grad()
    y = model(src_idx, tgt_idx, src_key_padding_mask=src_key_padding_mask)
    y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
    loss = loss_fn(y, label)
    print(loss.item())
    loss.backward()
    optimizer.step()
