# CS336 Language Models from Scratch — Assignment 2: Systems

## Assignment Description

> 本作业聚焦于大语言模型训练的**系统优化**，分别实现了三个核心系统组件：
> 1. **FlashAttention-2**：通过分块计算将注意力机制的 HBM 读写从 O(N²) 降至 O(N)
> 2. **分布式数据并行（DDP）**：实现逐参数和桶式两种梯度同步策略
> 3. **ZeRO-1 分片优化器**：将优化器状态显存从 O(N) 降至 O(N/K)

## Tech Stack

| 类别 | 详情 |
|------|------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.x |
| GPU 编程 | Triton（OpenAI） |
| 分布式 | torch.distributed（NCCL 后端） |
| 编译优化 | `@torch.compile` |
| 工具 | CUDA, torchrun |

## Core Knowledge Points

- **FlashAttention-2 分块算法**：将 Q、K、V 分成大小为 B 的块，在 GPU SRAM 内完成 softmax 和加权求和，避免 O(N²) 的中间矩阵写回 HBM；实现见 `attention_triton.py` 的 `FlashAttention2Pytorch.forward`
- **在线 Softmax（Online Softmax）**：维护运行最大值 `m_i` 和归一化分母 `l_i`，每处理新 K/V 块时用修正因子 `alpha = exp(m_old - m_new)` 更新旧值，保证数值稳定同时无需重扫
- **log-sum-exp 缓存（L 矩阵）**：正向传播保存 `L_i = m_i + log(l_i)`，反向传播用 `P = exp(S - L)` 一次性重建 softmax 概率，避免分块重算
- **Triton GPU 内核**：`@triton.jit` 将 Python 函数编译为 PTX 指令，`tl.make_block_ptr` 管理 HBM→SRAM 分块加载，`@triton.autotune` 自动搜索最优 tile 配置（64×64 到 128×128）
- **DDP 逐参数梯度同步**：`register_post_accumulate_grad_hook` 在梯度累积完毕后立即异步 AllReduce，实现通信与后续层反向传播的 overlap；实现见 `ddp_training.py` 的 `DDPIndividualParameters`
- **桶式 AllReduce（Bucketed DDP）**：将多参数梯度用 `_flatten_dense_tensors` 拼平为大 tensor，一次 AllReduce 代替多次小 tensor 通信，显著降低 kernel launch 开销；实现见 `DDPBucketed`
- **ZeRO Stage 1 优化器分片**：第 i 个参数轮询分配给 rank `i % world_size`，每 rank 只维护 1/K 的 Adam m/v 状态；step 后 Broadcast 同步参数，实现优化器状态显存从 O(N) 降至 O(N/K)；实现见 `sharded_optimizer.py`

## Code Structure

```
Assignment2-system/
├── cs336_systems/
│   ├── attention_triton.py    # FlashAttention-2：PyTorch 版 + Triton GPU 内核版
│   ├── ddp_training.py        # DDP：逐参数异步 AllReduce + 桶式 AllReduce
│   └── sharded_optimizer.py   # ZeRO-1：分片优化器，轮询分配参数到各 rank
└── README.md
```

## Environment Setup

```bash
pip install torch triton
# 需要 CUDA 环境（Triton 内核依赖 GPU）
```

## How to Run

```bash
# 单机多卡 DDP 训练（4 GPU）
torchrun --nproc_per_node=4 train_script.py

# 测试 FlashAttention 正确性（与标准注意力对比）
python -c "
import torch
from cs336_systems.attention_triton import FlashAttention2Pytorch, FlashAttention2Triton
Q = torch.randn(2, 4, 128, 64, device='cuda')
K = torch.randn(2, 4, 128, 64, device='cuda')
V = torch.randn(2, 4, 128, 64, device='cuda')
out = FlashAttention2Triton.apply(Q, K, V, True)
print('Output shape:', out.shape)
"
```

## Key Results / Observations

- **FlashAttention 显存节省**：序列长度 N=2048 时，标准注意力需要 O(N²)=4M 的中间矩阵，FlashAttention 只需 O(N)=2K 的在线状态
- **DDP 通信效率**：桶式 AllReduce（bucket_size=25MB）相比逐参数 AllReduce，kernel launch 次数从参数数量级降至桶数量级（约 10x 减少）
- **ZeRO-1 效果**：8 卡训练时，每卡的 AdamW 优化器状态从 `2 × 模型参数量` 降至 `2 × 模型参数量 / 8`，对大模型（>1B 参数）节省显著

## Notes

- `FlashAttention2Pytorch` 使用 Python 循环实现，仅用于验证算法正确性，实际训练请使用 `FlashAttention2Triton`（Triton GPU 内核版）
- Triton 内核的 `is_causal` 为 `constexpr`，改变该参数会触发重新编译
- `ShardedOptimizer` 在单机（world_size=1）时退化为普通优化器，无额外开销
- `DDPBucketed.reset_buckets()` 需在每个 step 后手动调用，重置就绪计数器
