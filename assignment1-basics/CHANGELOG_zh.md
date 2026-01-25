# 更新日志（Changelog）

我们对作业代码或 PDF 所做的所有更改，都会记录在本文件中。

## [1.0.6] 2025-08-28
- 讲义：修复 RoPE 公式中的一个错误
- 代码：修复 adapters 中的类型标注
- 代码：重新锁定依赖版本
- 代码：小幅代码格式整理

## [1.0.5] 2025-04-15
- 代码：新增提交脚本，修复拼写错误
- 代码：使用 `uv_build` 作为包构建系统
- 讲义：修复 RoPE 索引问题
- 代码：修复截断语言模型输入的测试
- 代码：使用 Ruff 进行代码格式化和 lint
- 代码：简化快照测试（snapshot testing）
- 代码：使所有内容兼容 `ty` 类型系统

## [1.0.4] 2025-04-08

### 新增（Added）
- 讲义：新增并行化 pretokenization 的指导，并提供用于分块（chunking）的示例代码
- 讲义：新增在 pretokenization 前移除特殊 token 的说明（应当在它们处进行切分）

### 修改（Changed）
- 讲义：修复在 MPS 后端编译模型的命令

## [1.0.3] 2025-04-07

### 新增（Added）
- 代码：新增在训练 BPE 时移除特殊 token 的测试

### 修改（Changed）
- 讲义：修复 RoPE 的 off-by-one 错误
- 代码：修复 Intel Mac 相关问题，在 macOS x86_64 上支持 Python 3.11 和 PyTorch 2.2.2

## [1.0.2] 2025-04-03

### 新增（Added）
- 代码：补充 Linear 和 Embedding 的缺失测试

### 修改（Changed）
- 讲义：修复 RMSNorm 接口定义
- 讲义：在 RMSNorm 和 SwiGLU 的说明中加入数值稳定性提示
- 讲义：澄清 BPE 示例中使用的是按空白符拆分的 naive pretokenization；你的实现仍应使用提供的正则表达式
- 讲义：修复 RMSNorm 的 docstring

## [1.0.1] 2025-04-02

### 修改（Changed）
- 代码：为 RoPE 测试增加一定容差
- 代码：确保测试时模型加载在 CPU 上
- 代码：修复 Tensor 类型注解在 VSCode 中的 docstring 可读性问题（请修复这个，@Microsoft）

## [1.0.0] 2025-04-01

### 新增（Added）
- 代码：使用 uv 进行环境管理
- 代码：Tensor 类型标注
- 代码：快照测试（仅解答版可用，使用 `--update-snapshots` 更新）
- 代码/讲义：SwiGLU
- 代码/讲义：RoPE
- 讲义：einops
- 代码/讲义：新增消融实验
- 讲义：低资源训练建议
- 讲义：从零实现 Linear 和 Embedding

### 修改（Changed）
- 讲义：全面重排格式、结构和措辞
- 讲义：重新定义注意力机制
- 讲义：GeLU → SiLU
- 讲义：参数初始化方式调整
- 讲义：移除 dropout
- 讲义：重绘所有示意图
- 讲义：新增实验
- 讲义：补充 BPE 相关解释、系统分析以及对解答中“奇怪行为”的说明

### 修复（Fixed）
- 代码：修复在未启用 CUDA 的环境中 `test_get_batch` 抛出错误的问题
- 讲义：澄清梯度裁剪的范数是对**所有参数**统一计算的
- 代码：修复梯度裁剪测试中比较了错误张量的问题
- 代码：修复在存在多个参数、部分参数无梯度时的梯度范数计算逻辑

## [0.1.6] 2024-04-13

### 修复（Fixed）
- 讲义：将 TinyStories 预期运行时间修改为 30–40 分钟
- 讲义：补充如何使用 `np.memmap` 或 `np.load` 的 `mmap_mode` 参数
- 代码：修复 `get_tokenizer()` 的 docstring
- 讲义：明确 `main_experiment` 题目应使用与 TinyStories 相同的设置
- 代码：将 LayerNorm 的所有提及替换为 RMSNorm

## [0.1.5] 2024-04-06

### 修改（Changed）
- 讲义：澄清在 BPE 合并中偏好字典序更大的合并对，具体为使用 tuple 比较

### 修复（Fixed）
- 讲义：修复 TinyStories 的训练 token 总数，应为 327,680,000
- 代码：修复 `run_get_lr_cosine_schedule` 返回值 docstring 的拼写错误
- 代码：修复 `test_tokenizer.py` 中的拼写错误

## [0.1.4] 2024-04-04

### 修改（Changed）
- 代码：在非 Linux 系统上跳过 Tokenizer 的内存相关测试（RLIMIT_AS 支持不一致）
- 代码：放宽端到端 Transformer 前向测试的 atol
- 代码：在模型相关测试中移除 dropout，以提升跨平台确定性
- 代码：在 `run_multihead_self_attention` adapter 中新增 `attn_pdrop`
- 代码：澄清 adapters 中 `{q,k,v}_proj` 的维度顺序
- 代码：放宽交叉熵测试的 atol
- 代码：移除 `test_get_lr_cosine_schedule` 中不必要的 warning

### 修复（Fixed）
- 讲义：修复 `Tokenizer.__init__` 的函数签名，补充 `self`
- 讲义：说明 `Tokenizer.from_files` 应为类方法
- 讲义：澄清 `adamwAccounting` 中列出的模型超参数
- 讲义：澄清 `adamwAccounting` (b) 使用的是 GPT-2 XL **形状**的模型，而非字面意义上的 GPT-2 XL
- 讲义：将 softmax 相关题目移动到首次介绍 softmax 的位置（Scaled Dot-Product Attention，第 3.4.3 节）
- 讲义：移除 AdamW 伪代码中冗余的初始化步骤（t = 0）
- 讲义：补充 BPE 训练所需资源说明

## [0.1.3] 2024-04-02

### 修改（Changed）
- 讲义：在 `adamWAccounting` (d) 中定义 MFU，并说明反向传播通常假设 FLOPs 为前向的两倍
- 讲义：提供当 `Tokenizer.decode` 接收到无效 UTF-8 字节 ID 时的期望行为提示

## [0.1.2] 2024-04-02

### 新增（Added）
- 讲义：补充排行榜提交相关说明

## [0.1.1] 2024-04-01

### 新增（Added）
- 代码：在 README.md 中添加说明，鼓励提交 Pull Request 和 Issue

### 修改（Changed）
- 讲义：在 pre-tokenization 动机中补充关于仅在标点符号不同的 token 的处理说明
- 讲义：移除每个小节后的总分说明
- 讲义：说明大语言模型（如 LLaMA、GPT-3）常用 AdamW 的 beta 值为 (0.9, 0.95)，不同于 PyTorch 默认的 (0.9, 0.999)
- 讲义：明确 `adamw` 题目的提交内容
- 代码：将 `test_serialization::test_checkpoint` 重命名为 `test_serialization::test_checkpointing` 以与讲义保持一致
- 代码：略微放宽 `test_train_bpe_speed` 的时间限制

### 修复（Fixed）
- 代码：修复 `train_bpe` 测试中合并规则与词表不匹配的问题  
  - 原因在于参考实现（对齐 HuggingFace）会将不可读字节映射为可打印 Unicode 字符，我们之前错误地在该映射后进行字典序比较
- 讲义：修复推荐 TinyStories 超参数模型中非 Embedding 参数数量的错误（第 7.2 节）
- 讲义：将 `decoding` 题目中的 `<|endofsequence|>` 替换为 `<|endoftext|>`
- 代码：修复 setup 命令（`pip install -e .'[test]'`），提升 zsh 兼容性
- 讲义：修复多处拼写和格式问题

## [0.1.0] 2024-04-01

初始版本发布（Initial release）。
