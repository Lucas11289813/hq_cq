import torch
from FlagEmbedding import FlagModel
from torch.utils.data import Dataset, DataLoader
from modelscope.msdatasets import MsDataset
import traceback


class ChatDataset(Dataset):
    def __init__(self):
        super().__init__()
        _ds = MsDataset.load('Moemuu/Muice-Dataset')
        self.embedding = FlagModel(r'D:\projects\py-projects\bge-small-zh', use_fp16=True)
        self.ds = _ds['train']
        # 输入数据
        # (src, tgt)
        self.inputs = []
        # 标签数据
        self.labels = []
        try:
            for i in range(len(self.ds)):
                d = self.ds[i]
                src = d['prompt']
                tgt = d['respond']
                if len(tgt) >= 50:
                    continue
                # 处理数据
                # 我们将 src 和 tgt 都补充成 50 个字长

                # 分词
                src_idx = self.embedding.tokenizer(src, add_special_tokens=False)['input_ids']
                # 需要补充多少个 [PAD]
                pad_num = 50 - len(src_idx)
                src_idx = torch.tensor([*src_idx, *[0 for i in range(pad_num)]])
                # 创建 padding_mask
                src_key_padding_mask = torch.zeros(50)
                src_key_padding_mask[src_idx == 0] = float('-inf')

                # 分词
                _tgt_idx = self.embedding.tokenizer(tgt)['input_ids']
                # 去掉结尾
                tgt_idx = _tgt_idx[:-1]
                # 要填充的长度
                pad_num = 50 - len(tgt_idx)
                tgt_idx = torch.tensor([*tgt_idx, *[0 for i in range(pad_num)]])
                # 创建padding掩码
                tgt_key_padding_mask = torch.zeros(50)

                tgt_key_padding_mask[tgt_idx == 0] = float('-inf')

                # 处理label
                label = _tgt_idx[1:]
                pad_num = 50 - len(label)
                label = torch.tensor([*label, *[0 for i in range(pad_num)]])

                # 保存数据
                self.inputs.append((src_idx, src_key_padding_mask, tgt_idx, tgt_key_padding_mask))
                self.labels.append(label)
        except:
            print('error')
            # 打印调用栈信息
            traceback.print_exc()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        labels = self.labels[idx]
        return inputs, labels


if __name__ == '__main__':
    ds = ChatDataset()
    print(len(ds))
    print(ds[0])
    dl = DataLoader(ds, batch_size=10, shuffle=True)
    for i, (inputs, labels) in enumerate(dl):
        print(inputs)
        print(labels)
        break

