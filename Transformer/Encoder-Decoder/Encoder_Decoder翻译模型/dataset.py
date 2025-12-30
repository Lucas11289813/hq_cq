import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class TranslateDataset(Dataset):
    def __init__(self):
        super().__init__()
        all_text = '<SOS> <EOS> i love you 我 爱 你 们'
        self.words = all_text.split(' ')

        input_text = 'i love you'
        input_idx = [self.words.index(word) for word in input_text.split(' ')]
        # one-hot 编码
        self.inputs = F.one_hot(torch.tensor(input_idx), len(self.words))
        label_text = '我 爱 你 们 <EOS>'
        label_idx = [self.words.index(word) for word in label_text.split(' ')]
        self.labels = torch.tensor(label_idx)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.inputs.to(torch.float), self.labels


if __name__ == '__main__':
    ds = TranslateDataset()
    print(ds[0])
