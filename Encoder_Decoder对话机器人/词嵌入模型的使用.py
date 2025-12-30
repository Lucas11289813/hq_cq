# 词嵌入模型
# 作用: 将文本转换成向量
# 用在哪里?
# 1. 训练和使用语言模型的时候，但是类似 GPT 这样的模型会使用自己的词嵌入模型
# 2. 知识库中，做文本检索的时候
# 官网: https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md

# 安装: pip install -U FlagEmbedding
# pip install peft

# 导入模型类
from FlagEmbedding import FlagModel

t1 = '你好，请问你叫什么名字？'
t2 = '你好吗？请问你叫啥名？'

# 构造词嵌入模型
model = FlagModel(r'D:\projects\py-projects\bge-small-zh', use_fp16=True)

# 密集编码
r1 = model.encode(t1)
print(r1)
print(r1.shape)
r2 = model.encode(t2)

# 相似度得分
score = r1 @ r2.T
print(score)

# 分词
# add_special_tokens: 是否添加特殊符号，例如: [CLS] [SEP]
r1 = model.tokenizer(t1, add_special_tokens=False)
print(r1)

# 获取词库的大小
print(len(model.tokenizer))

# 解码
t = model.tokenizer.decode(r1['input_ids'])
print(t)

# 转换 idx list 变成 token 列表
t = model.tokenizer.convert_ids_to_tokens(r1['input_ids'])
print(t)

# 编码
word_vector = model.encode(t)
print(word_vector.shape)
