# chatglm.cpp

chatglm.cpp 是 github 上的一个开源项目，是对 chatglm 的 c++ 实现

该项目可以量化原生的 chatglm，以达到用更低的电脑资源来更高效的运行大模型的目的

可以将该项目作为原生 ChatGLM 的平替

[github 地址](https://github.com/li-plus/chatglm.cpp)

chatglm.cpp 也是一个大模型引擎，该项目受 llama.cpp 的启发而创建，他们运行的模型格式为 ggml 模型

> 当我使用 ubuntu 22 搭建 llama.cpp 时，发现其二进制命令行工具需要在更高版本的 ubuntu 中才能运行（Ubuntu 23.10 以上）

## 搭建环境

克隆项目

```shell
# 注意：需要添加 --recursive
git clone --recursive https://github.com/li-plus/chatglm.cpp.git
```

创建 conda 环境

```shell
conda create --name chatglm_cpp python=3.10
```

安装依赖

```shell
pip install -U pip
pip install torch tabulate tqdm transformers accelerate sentencepiece
```

## 转换量化模型

```shell
# -i THUDM/chatglm-6b 是本地模型参数路径
python chatglm_cpp/convert.py -i THUDM/chatglm-6b -t q4_0 -o models/chatglm-ggml.bin
```

最后会输出一个 GGML 格式的模型文件 `models/chatglm-ggml.bin`

## 构建和运行（非必要）

> 注意: 为了运行 cmake，需要提前安装 c++ 编译环境和 cmake
> 这里使用的微软 `vs_BuildTools.exe` 和 `cmake-3.30.3-windows-x86_64.zip`


构建后会生成一个 `main.exe` 文件，然后就可以使用命令行待用大模型程序

构建命令如下:

```shell
cmake -B build
cmake --build build -j --config Release
```

构建好后运行命令行开始对话

```shell
# -p 提示词
./build/bin/main -m models/chatglm-ggml.bin -p 你好
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
# 互动模式启动程序
./build/bin/main -m models/chatglm-ggml.bin -i
# 查看帮助信息
./build/bin/main -h
```

## 加载预转换的 GGML 模型

首先安装 `chatglm.cpp` 项目自身

```shell
pip install .
```

将安装后生成的 `build/lib.win-amd64-cpython-310/chatglm_cpp/_C.cp310-win_amd64.pyd` 复制到 `chatglm_cpp/_C.cp310-win_amd64.pyd`

接下来就可以创建脚本，导入量化后的模型了

创建脚本 `load_ggml_demo.py` 如下:

```python
import chatglm_cpp

pipeline = chatglm_cpp.Pipeline("models/chatglm-ggml.bin")
messages = pipeline.chat([chatglm_cpp.ChatMessage(role="user", content="你好")])
print(messages)
# 打印结果: ChatMessage(role="assistant", content="你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。", tool_calls=[])
```

## LangChain API

我们在使用 LangChain 时，需要一个模型服务程序来提供问答服务功能，这里我们使用 `chatglm.cpp` 来提供问答功能

先安装服务器依赖如下：

```shell
pip install chatglm-cpp[api]
```

启动 `LangChain API` 服务器

```shell
set MODEL=./models/chatglm-ggml.bin && uvicorn chatglm_cpp.langchain_api:app --host 127.0.0.1 --port 8000
```

测试服务器是否正常运行

```shell
curl http://127.0.0.1:8000 -H "Content-Type: application/json" -d "{\"prompt\": \"你好\"}"
```

服务搭建好了，接下来需要使用 `LangChain` 来调用服务器，先安装 `LangChain` 依赖

```shell
pip install -U langchain-community
```

编写脚本 `langchain_demo.py` 如下

```python
from langchain_community.llms import ChatGLM

llm = ChatGLM(endpoint_url="http://127.0.0.1:8000", max_token=2048, top_p=0.7, temperature=0.95, with_history=False)
print(llm.invoke("你好"))
```

运行查看结果

## OpenAI API

可以直接用 `chatglm.cpp` 替代 `OpenAI` 的 API Server，但是因为毕竟用的不是 `ChatGPT` 模型，所以对话过程中的一些数据接口是不能等价于 `OpenAI` 的接口的

```shell
# linux 系统命令
MODEL=./models/chatglm-ggml.bin uvicorn chatglm_cpp.openai_api:app --host 127.0.0.1 --port 8000
# windows 系统命令
set MODEL=./models/chatglm-ggml.bin && uvicorn chatglm_cpp.openai_api:app --host 127.0.0.1 --port 8000
```