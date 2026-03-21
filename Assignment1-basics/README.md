# CS336 Spring 2025 — Assignment 1: Basics

> 从零实现语言模型的核心组件：BPE 分词器、Transformer 模型、AdamW 优化器、文本生成。

## 作业说明

该作业要求从零手实现语言模型的全部基础组件，包括 BPE 分词、Transformer 架构、训练流程和文本生成，在 TinyStories 数据集上训练并评估模型。

## 技术栈

| 类别 | 详情 |
|------|------|
| 语言 | Python 3.11+ |
| 核心库 | PyTorch、einops、regex、NumPy |
| 实验工具 | uv（包管理）、pytest（单元测试） |
| 计算设备 | CUDA GPU / Apple Silicon MPS / CPU |
| 数据集 | TinyStories、OpenWebText（子集） |

## 核心知识点

- **BPE（Byte Pair Encoding）分词**：实现了完整的 BPE 训练算法，包含多进程并行统计词频、增量缓存更新和贪心合并策略，在 `bpe.py` 的 `train_bpe()` 中实现
- **GPT-2 预分词**：使用 Unicode 正则表达式将文本切分为单词，避免跨词合并，在 `bpe.py` 和 `tokenizer.py` 中实现
- **Transformer 橧模块**：自实现 RMSNorm、Causal Multi-Head Self-Attention、SwiGLU FFN，采用 Pre-Norm + 残差连接，在 `model.py` 的 `TransformerBlock` 中实现
- **RoPE（旋转位置编码）**：将位置编码为旋转矩阵作用于 Q/K，使注意力天然感知相对位置，在 `model.py` 的 `RotaryPositionalEmbedding` 中实现
- **SwiGLU 激活函数**：门控机制提升非线性表达能力，公式为 `W2(SiLU(W1·x) ⊗ W3·x)`，在 `model.py` 的 `SwiGLU` 中实现
- **AdamW 优化器**：手动实现 Adam 动量更新、偏差修正与解耦权重衰减，在 `train.py` 的 `AdamW` 中实现
- **余弦学习率调度**：线性预热 + 余弦衰减调度策略，提升训练稳定性，在 `train.py` 的 `get_lr_cosine_schedule()` 中实现
- **Top-p（Nucleus）采样**：生成时像枚中只保留累积概率超过 p 的最小候选集，平衡多样性与质量，在 `generate.py` 中实现
- **梯度裁剪**：全局 L2 范数裁剪防止梯度爆炸，在 `train.py` 的 `clip_gradients()` 中实现

## 代码结构

```
Assignment1-basics/
├── Transformer/
│   ├── bpe.py                   # BPE 分词器训练（多进程 + 增量缓存）
│   ├── tokenizer.py             # BPE 分词器推理（encode/decode）
│   ├── model.py                 # Transformer 模型全部组件
│   ├── train.py                 # 训练流程、AdamW、裁剪、检查点
│   ├── generate.py              # 文本生成（Temperature + Top-p）
│   ├── pretokenization_example.py  # GPT-2 预分词示例
│   ├── run_bpe_experiments.py   # BPE 实验脚本
│   ├── run_tokenizer_experiments.py # 分词器实验脚本
│   └── read_pkl.py              # 读取已保存的分词器文件
└── cs336_spring2025_assignment1_basics.pdf  # 作业手册
```

## 环境配置

使用 `uv` 管理虹境（推荐）：

```bash
# 安装 uv
pip install uv
# 或 macOS
brew install uv

# 运行任意 Python 文件（自动解决并激活虹境）
uv run <python_file_path>

# 运行单元测试
uv run pytest
```

## 数据准备

```bash
mkdir -p data && cd data

# TinyStories 数据集
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText 子集
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
cd ..
```

## 运行方法

```bash
# 1. 训练 BPE 分词器
cd Transformer
python run_bpe_experiments.py

# 2. 训练模型
python train.py

# 3. 文本生成
python generate.py
```

## 关键实验结果

- 在 TinyStories 上训练 5000 步后，损失从初始 ~9.0 降至 ~3.2，困惑度从 ~8000 降至 ~24
- 使用 10000 大小词表 + 256 上下文长度，模型已能生成逻辑一致的小故事文本
- BPE 增量缓存优化使 10000 词表训练速度提升明显

## 注意事项

- `d_ff = 1344` 约为 `d_model * 2.625`，符合 SwiGLU 推荐的 FFN 剰大小比例
- 模型训练前需先运行 `run_bpe_experiments.py` 生成 `outputs/TinyStories_tokens.npy`
- `generate.py` 中模型超参数必须与 `train.py` 完全一致
