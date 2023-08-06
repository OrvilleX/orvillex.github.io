# Llama-2安装运行

## 一、低现存运行

由于Llama官方最低参数要求的运行环境也比较高，为了确保我们能够在低现存的配置环境下依然可以运行并使用，为此我们需要结合[Llama-2](https://github.com/facebookresearch/llama)以及[Llama.cpp](https://github.com/ggerganov/llama.cpp)集合进行训练以及运行。

### 1.1 Llama-2 安装  

首先需要下载模型权重和分词器，请访问[Meta AI网站](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)并接受我们的许可证。一旦您的请求获得批准，您将通过电子邮件收到签名的 URL。

通过clone该[项目](https://github.com/facebookresearch/llama)后，运行 ​​download.sh 脚本，并在提示开始下载时传递提供的 URL。确保复制 URL 文本本身，右键单击 URL 时不要使用“复制链接地址”选项。如果复制的 URL 文本以https://download.llamameta.net开头，则您复制正确。如果复制的 URL 文本以：https://l.facebook.com开头，则您复制的方式错误。

### 1.2 Llama.cpp 安装

首先我们需要将该项目继续clone，完成clone后对项目进行编译。

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

完成上述项目的编译后，我们需要将Llama对应的权重文件放置在`models`文件夹，确保`tokenizer_checklist.chk`与`tokenizer.model`文件均在文件夹下，然后执行下述命令。

```bash
python3 -m pip install -r requirements.txt

python3 convert.py models/Llama-2-7B/

./quantize ./models/Llama-2-7B/ggml-model-f16.bin ./models/Llama-2-7B/ggml-model-q4_0.bin q4_0

./main -m ./models/Llama-2-7B/ggml-model-q4_0.bin -n 128
```

上述我们可以完成对模型的转换以及验证，我们希望像Chat-GPT一样进行交互操作则需要按照下述方式执行对应的指令。

```bash
# 以默认参数运行 7B 模型
./examples/chat.sh

# 以自定义参数运行 7B 模型
./main -m ./models/Llama-2-7B/ggml-model-q4_0.bin -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```