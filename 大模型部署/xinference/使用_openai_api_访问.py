# 安装 chatglm-cpp[api]
# pip install chatglm-cpp[api]

# 启动服务
# set MODEL=./models/chatglm-ggml.bin && uvicorn chatglm_cpp.openai_api:app --host 127.0.0.1 --port 8000

# 安装 openai 库
# pip install openai


import openai
import numpy as np

# 创建 Client 客户端 实例
client = openai.Client(
    # 开发者密钥，需要随便填写一个
    api_key='token123',
    # 访问请求时的基础网络路径
    base_url='http://192.168.128.134:9997/v1'
)

# api 文档: https://platform.openai.com/docs/api-reference/chat/create
# 向服务器发起请求
# response = client.chat.completions.create(
#     # 访问 default 模型
#     model='llama3-3b',
#     # 消息
#     messages=[
#         {"role": "system", "content": "你是一个人工智能助手叫做张三"},
#         {"role": "user", "content": "你叫什么名字？"}
#     ],
#     # 因为我们使用的是openai服务器的模拟器
#     # 所以并非所有参数都有效
#     # 是否流式输出
#     stream=True
# )
#
# print(response)
# for chunk in response:
#     print(chunk)


# https://platform.openai.com/docs/api-reference/embeddings
response = client.embeddings.create(
    input="我爱中国",
    model="my-bge-small-zh"  # 选择适合的模型
)

print(response)
# 词嵌入的结果，是列表类型
print(response.data[0].embedding)
# 长度取决于模型的嵌入维度，此处 bge-small-zh 的嵌入长度为 512
print(len(response.data[0].embedding))

embed1 = np.array(response.data[0].embedding)

response = client.embeddings.create(
    input="我爱中华",
    model="my-bge-small-zh"
)

embed2 = np.array(response.data[0].embedding)

# 相似度得分
print(embed1 @ embed2.T)
