import torch

from model import ChatBot

model = ChatBot()
model.load_state_dict(torch.load('model.pt'))
model.eval()


# _src = '你叫什么名字？'
# 流式输出对话结果
def stream(_src):
    # 目标序列以 101 开头
    _tgt = [101]

    # 对输入进行分词
    src_idx = model.embedding.tokenizer(_src, add_special_tokens=False)['input_ids']
    src = torch.tensor([src_idx])

    max_len = 30

    for i in range(max_len):
        # 将目标序列回复成文本
        tgt = torch.tensor([_tgt])
        y = model(src, tgt)
        idx = y.argmax(-1)

        last_idx = idx[0][-1].item()

        # 检查最后一个字是否是结束符
        if last_idx == 102:
            break

        tokens = model.embedding.tokenizer.convert_ids_to_tokens(idx[0].numpy())

        yield tokens[-1]

        # 追加模型预测的最后一个字到目标序列
        _tgt.append(last_idx)


# 聊天
def chat(src):
    chunks = []
    for chunk in stream(src):
        chunks.append(chunk)
    return ''.join(chunks)


if __name__ == '__main__':
    # for chunk in stream('你叫什么名字？'):
    #     # chunk: 模型每次返回的文本块
    #     print(chunk, end='')
    # print()
    # print(chat('你叫什么名字？'))
    while True:
        src = input('Human: ')
        print('AI: ', end='')
        for chunk in stream(src):
            print(chunk, end='')
        print()
