import os
import math
import torch
import numpy as np
from typing import Callable, Iterable, Optional
import gc

from model import TransformerLM

# ─────────────────────────────────────────
# 交叉熵损失（手动实现，数值稳定版）
# Purpose: 计算语言模型的训练损失，衡量预测分布与真实标签的差异
# Key concept: 交叉熵 = -log P(correct_token)，等价于负对数似然
#   数值技巧：先减去最大 logit，防止 exp 溢出（log-sum-exp trick）
# ─────────────────────────────────────────
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # log-sum-exp trick：减去最大值保持数值稳定
    log_max = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_shifted = logits - log_max
    # 计算 log(Σ exp(logit_i))，即归一化常数的对数
    log_sum_up = torch.log(torch.sum(torch.exp(logits_shifted), -1))

    # 取出目标 token 对应的 logit（shifted 后）
    logits_target = torch.gather(logits_shifted, -1, targets.unsqueeze(-1)).squeeze(-1)
    # 交叉熵 = log_normalizer - log_target_logit
    loss = log_sum_up - logits_target
    return loss.mean()  # 对 batch 和序列位置取均值


def compute_perplexity(loss: torch.Tensor) -> float:
    """
    由平均交叉熵损失计算困惑度（Perplexity）。
    Key concept: 困惑度 = exp(loss)，衡量语言模型对文本的「困惑程度」，越低越好
    """
    return math.exp(loss.item())


# ─────────────────────────────────────────
# AdamW 优化器（手动实现）
# Purpose: 训练 Transformer 参数，结合自适应学习率与解耦权重衰减
# Key concept: AdamW = Adam + 解耦权重衰减
#   Adam 核心：维护梯度的一阶矩（动量 m）和二阶矩（自适应缩放 v）
#   偏差修正：早期步骤 m/v 被初始化为 0 导致偏小，需除以 (1-β^t) 修正
#   解耦权重衰减：直接对参数乘以 (1-lr*wd)，不通过梯度，效果优于 L2 正则
# ─────────────────────────────────────────
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f'Damn! lr:{lr}<0')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data  # 当前梯度
                state = self.state[p]

                # 首次更新：初始化状态变量
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # 一阶矩（梯度指数移动平均）
                    state['v'] = torch.zeros_like(p.data)  # 二阶矩（梯度平方指数移动平均）

                state['step'] += 1
                t = state['step']
                m = state['m']
                v = state['v']

                # 更新一阶矩：m = β1*m + (1-β1)*g
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                # 更新二阶矩：v = β2*v + (1-β2)*g²
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # 偏差修正：消除 m、v 初始化为 0 带来的低估
                bias_corrections1 = 1 - beta1 ** t
                bias_corrections2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_corrections2) / bias_corrections1)

                # 解耦权重衰减：直接缩小参数，不经过梯度（AdamW 与 Adam 的核心区别）
                p.data.mul_(1 - lr * weight_decay)
                # Adam 参数更新：p -= step_size * m / (√v + eps)
                p.data.addcdiv_(m, v.sqrt() + eps, value=-step_size)

        return loss


# ─────────────────────────────────────────
# 余弦退火学习率调度（带线性预热）
# Purpose: 动态调整学习率，先预热再余弦衰减，提升训练稳定性和最终性能
# Key concept: Cosine Annealing with Warmup
#   阶段1（预热）：[0, T_w) 线性从 0 升到 alpha_max
#   阶段2（余弦衰减）：[T_w, T_c] 从 alpha_max 余弦衰减到 alpha_min
#   阶段3（平稳）：(T_c, ∞) 保持 alpha_min
# ─────────────────────────────────────────
def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        # 线性预热阶段：学习率从 0 线性增长到 alpha_max
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        # 余弦衰减阶段：平滑地从 alpha_max 降至 alpha_min
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (
                alpha_max - alpha_min)
    else:
        # 学习率下界：保持最小学习率不再降低
        return alpha_min


# ─────────────────────────────────────────
# 梯度裁剪（全局范数裁剪）
# Purpose: 防止梯度爆炸，保证训练稳定性
# Key concept: 全局梯度范数裁剪——若所有参数梯度的 L2 范数超过 max_norm，则等比缩放
#   公式：if ‖g‖ > max_norm: g *= max_norm / ‖g‖
# ─────────────────────────────────────────
@torch.no_grad()
def clip_gradients(parameters, max_norm: float, eps: float = 1e-6):
    # 过滤出有梯度的参数
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return
    # 计算所有参数梯度拼接后的全局 L2 范数
    total_norm = torch.norm(torch.stack([torch.norm(p.grad, 2) for p in parameters]), 2)

    if total_norm > max_norm:
        # 按比例缩放所有梯度，使全局范数恰好等于 max_norm
        scale = max_norm / (total_norm + eps)
        for p in parameters:
            p.grad.mul_(scale)


# ─────────────────────────────────────────
# 随机批次采样
# Purpose: 从预处理好的 numpy 数组中随机采样训练批次
# Key concept: 语言模型数据对：输入 x 为位置 [i, i+L)，目标 y 为位置 [i+1, i+L+1)
#   即 y[t] 是 x[t] 的下一个 token（自回归训练目标）
# ─────────────────────────────────────────
def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str):
    # 随机采样 batch_size 个起始索引，留出 context_length 的余量
    idx = torch.randint(0, len(data) - context_length - 1, (batch_size,))

    # x：输入序列，y：目标序列（x 右移一位）
    x = torch.stack([
        torch.from_numpy(data[i:i + context_length].astype(np.int64))
        for i in idx
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + context_length + 1].astype(np.int64))
        for i in idx
    ])

    return x.to(device), y.to(device)


# ─────────────────────────────────────────
# 检查点保存与加载
# Purpose: 定期保存训练状态，支持断点续训
# Key concept: 检查点 = 模型权重 + 优化器状态 + 当前迭代步数
# ─────────────────────────────────────────
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    """加载检查点并恢复模型和优化器状态，返回已训练的迭代步数。"""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


def free_gpu_memory():
    """
    清理无用变量并释放 GPU 显存缓存。
    Purpose: 训练前确保显存干净，避免碎片化导致 OOM。
    Key concept: PyTorch 显存缓存——PyTorch 不立即归还已释放显存给 OS，需手动清理。
    """
    # 触发 Python 垃圾回收，强制回收不再引用的张量
    gc.collect()

    if torch.cuda.is_available():
        # 清空 PyTorch 的 CUDA 缓存分配器
        torch.cuda.empty_cache()
        # 重置显存峰值统计，便于后续监控
        torch.cuda.reset_peak_memory_stats()
        print("CUDA 显存缓存已释放！")
    elif torch.backends.mps.is_available():
        # 兼容 Apple Silicon (M1/M2/M3) 的 MPS 后端
        torch.mps.empty_cache()
        print("MPS 显存缓存已释放！")
    else:
        print("当前使用 CPU，无显存需清理。")


def main():
    free_gpu_memory()

    # ── 超参数配置 ──
    vocab_size = 10000       # BPE 词表大小
    context_length = 256     # 序列最大长度（上下文窗口）
    d_model = 512            # 隐藏层维度
    d_ff = 1344              # FFN 中间层维度（约 2.625 × d_model，符合 SwiGLU 推荐比例）
    num_layers = 4           # Transformer 层数
    num_heads = 16           # 注意力头数（d_k = d_model / num_heads = 32）
    batch_size = 256         # 每步训练的样本数

    alpha_max = 5e-4         # 学习率峰值
    alpha_min = 1e-5         # 学习率下界
    T_w = 700                # 预热步数
    max_steps = 5000         # 总训练步数

    # 自动检测计算设备：优先 NVIDIA GPU (cuda)，其次 Apple Silicon (mps)，最后 CPU
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"使用设备: {device}")

    # 加载预分词后的 token 数组（内存映射模式，节省 RAM）
    data_path = "outputs/TinyStories_tokens.npy"
    if not os.path.exists(data_path):
        print("未找到数据，请先运行数据预处理脚本。")
        return
    data = np.load(data_path, mmap_mode='r')  # mmap 模式：按需读取，不全量载入内存

    # 构建模型并移动到目标设备
    model = TransformerLM(
        vocab_size=vocab_size, context_length=context_length,
        num_layers=num_layers, d_model=d_model,
        num_heads=num_heads, d_ff=d_ff, device=device
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=alpha_max, weight_decay=0.1)

    os.makedirs("checkpoints", exist_ok=True)
    model.train()

    # ── 主训练循环 ──
    for step in range(1, max_steps + 1):
        # 动态更新学习率（余弦调度）
        lr = get_lr_cosine_schedule(step, alpha_max, alpha_min, T_w, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 采样批次并前向传播
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 反向传播：清零梯度 -> 计算梯度 -> 裁剪 -> 更新参数
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
        optimizer.step()

        # 每 200 步打印训练进度
        if step % 200 == 0:
            perplexity = compute_perplexity(loss)
            print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.4f} | LR: {lr:.6f}")

        # 每 1000 步保存检查点
        if step % 1000 == 0:
            save_checkpoint(model, optimizer, step, f"checkpoints/model_step_{step}.pt")
            print(f"已保存检查点至 checkpoints/model_step_{step}.pt")


if __name__ == '__main__':
    main()
