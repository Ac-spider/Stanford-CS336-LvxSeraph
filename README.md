# Stanford CS336：从零实现语言模型

> Stanford CS336: Language Modeling from Scratch — 个人作业记录

本仓库包含斯坦福 CS336 课程的全部作业实现，从零构建大语言模型的核心组件，涵盖 Tokenizer、Transformer 架构、系统优化、数据处理和对齐训练。
代码注释工作由AI辅助完成。

## 课程简介

CS336 是斯坦福开设的大语言模型从零实现课程，要求学生不依赖高层框架，手动实现语言模型训练的完整流程，包括：BPE 分词、Transformer 架构、分布式训练、数据工程和 RLHF。

## 作业目录

| 作业 | 主题 | 核心内容 |
|------|------|---------|
| [Assignment1-basics](./Assignment1-basics/) | 基础组件 | BPE 分词器、GPT-2 风格 Transformer、AdamW、余弦学习率、top-p 采样 |
| [Assignment2-system](./Assignment2-system/) | 系统优化 | FlashAttention-2（Triton GPU 内核）、DDP 梯度同步、ZeRO-1 分片优化器 |
| [Assignment3-scaling](./Assignment3-scaling/) | 缩放规律 | Chinchilla 缩放定律、超参数搜索、训练效率分析 |
| [Assignment4-data](./Assignment4-data/) | 数据工程 | 数据清洗、去重、质量过滤、数据混合策略 |
| [Assignment5-alignment](./Assignment5-alignment/) | 对齐训练 | SFT 监督微调、RLHF、DPO 直接偏好优化 |

## 技术栈总览

| 类别 | 工具 |
|------|------|
| 语言 | Python 3.10+ |
| 深度学习 | PyTorch 2.x |
| GPU 编程 | Triton |
| 分布式 | torch.distributed（NCCL） |
| 编译优化 | `@torch.compile` |
| 数据处理 | NumPy、HuggingFace Datasets |

## 核心知识点

- **BPE 分词**：多进程并行训练，增量更新 pair_counts，GPT-2 预分词正则
- **Transformer 架构**：RMSNorm、SwiGLU、RoPE 旋转位置编码、因果多头注意力
- **AdamW 优化器**：解耦权重衰减、偏差修正、动量/方差追踪
- **训练技巧**：余弦学习率（warmup + 衰减 + floor）、梯度裁剪
- **FlashAttention-2**：在线 Softmax、分块 SRAM 计算、Triton GPU 内核
- **DDP**：异步 AllReduce、桶式梯度合并、通信-计算 overlap
- **ZeRO-1**：优化器状态轮询分片、Broadcast 同步参数

## 关联资源

- 课程主页：[cs336.stanford.edu](https://cs336.stanford.edu)
- 参考论文：Attention is All You Need, FlashAttention-2, ZeRO, Chinchilla
