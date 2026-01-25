# CS336 2025 春季 作业 1：基础（Assignment 1: Basics）

如需查看作业的完整说明，请参考作业讲义：
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

如果你在作业讲义或代码中发现任何问题，欢迎通过 GitHub Issue 提出，或直接提交 Pull Request 进行修复。

## 环境配置（Setup）

### 环境管理

我们使用 `uv` 来管理项目环境，以确保可复现性、可移植性以及易用性。

你可以在这里安装 uv（推荐）：
👉 https://github.com/astral-sh/uv
，或者使用以下方式安装：`pip install uv`/`brew install uv`

我们强烈建议你阅读 uv 的项目管理指南：
👉 https://docs.astral.sh/uv/guides/projects/#managing-dependencies （真的很值得一看！）

安装完成后，你可以使用以下命令运行仓库中的任意 Python 文件：

```sh
uv run <python_file_path>
```

uv 会在需要时自动解析并激活对应的环境。

## 运行单元测试

使用以下命令运行所有单元测试：

```sh
uv run pytest
```

在初始状态下，所有测试都会因为 `NotImplementedError` 而失败。
要将你的实现与测试连接起来，请完成以下文件中的函数实现：[./tests/adapters.py](./tests/adapters.py)

### 下载数据集

请下载 TinyStories 数据集以及 OpenWebText 的子集：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz


# Hugging Face 在国内经常直连不稳定，尤其是大文件
# 把链接里的 huggingface.co 换成 https://hf-mirror.com
# TinyStoriesV2-GPT4-train.txt | 2.1G
# TinyStoriesV2-GPT4-valid.txt | 21M
# owt_train.txt.gz             | 4.3G
# owt_valid.txt.gz             | 107M
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz


cd ..
```