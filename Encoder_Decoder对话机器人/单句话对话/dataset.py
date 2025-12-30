import torch
from torch.utils.data import Dataset
from FlagEmbedding import FlagModel


class ChatDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.embedding = FlagModel(r'D:\projects\py-projects\bge-small-zh', use_fp16=True)
        src = '你叫什么名字？'
        tgt = '我就是传说中的法外狂徒张三。'
        self.src = self.embedding.tokenizer(src, add_special_tokens=False)['input_ids']
        self.tgt = self.embedding.tokenizer(tgt)['input_ids'][:-1]
        self.label = self.embedding.tokenizer(tgt)['input_ids'][1:]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        # 模型的输入 标签
        return (torch.tensor(self.src), torch.tensor(self.tgt)), torch.tensor(self.label)


if __name__ == '__main__':
    ds = ChatDataset()
    print(ds[0])
