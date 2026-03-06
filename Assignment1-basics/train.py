import os
import math
import torch
import numpy as np
from typing import Callable, Iterable, Optional

from model import TransformerLM

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_max = torch.max(logits,dim=-1,keepdim=True)[0]
    logits_shifted = logits - log_max
    log_sum_up = torch.log(torch.sum(torch.exp(logits_shifted),-1))

    logits_target = torch.gather(logits_shifted,-1,targets.unsqueeze(-1)).squeeze(-1)
    loss =log_sum_up - logits_target
    return loss.mean()

def compute_perplexity(loss:torch.Tensor) -> float:
    return math.exp(loss.item())

class AdamW(torch.optim.Optimizer):
    def __init__(self,params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f'Damn!lr:{lr}<0')
        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params,defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1,beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']
                m = state['m']
                v = state['v']

                m.mul_(beta1).add_(g,alpha=1-beta1)
                v.mul_(beta2).addcmul_(g,g,value=1-beta2)

                bias_corrections1 = 1 - beta1**t
                bias_corrections2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_corrections2)/bias_corrections1)

                # 先进行解耦的权重衰减
                p.data.mul_(1 - lr * weight_decay)
                # 再进行 Adam 动量更新
                p.data.addcdiv_(m, v.sqrt() + eps, value=-step_size)

        return loss


def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:

    if t < T_w:
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (
                    alpha_max - alpha_min)
    else:
        return alpha_min


@torch.no_grad()
def clip_gradients(parameters, max_norm: float, eps: float = 1e-6):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return
    total_norm = torch.norm(torch.stack([torch.norm(p.grad,2) for p in parameters]),2)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in parameters:
            p.grad.mul_(scale)


def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str):
    idx = torch.randint(0,len(data)-context_length-1,(batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i:i+context_length].astype(np.int64))
        for i in idx
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+context_length+1].astype(np.int64))
        for i in idx
    ])

    return x.to(device),y.to(device)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    checkpoint = {
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "iteration" : iteration
    }
    torch.save(checkpoint,out)


def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']

import gc
def free_gpu_memory():
    """清理无用变量并释放显卡缓存"""
    # 1. 触发 Python 的垃圾回收，强制回收所有不再使用的变量
    gc.collect()

    # 2. 清空 PyTorch 内部的 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 重置显存峰值统计（可选，方便后续性能监控）
        torch.cuda.reset_peak_memory_stats()
        print("CUDA 显存缓存已释放！")

    # 3. 兼容 Apple Silicon 的 MPS 缓存清理
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("MPS 显存缓存已释放！")

    else:
        print("当前使用 CPU，无显存需清理。")

def main():
    free_gpu_memory()
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    num_layers = 4
    num_heads = 16
    batch_size = 256

    alpha_max = 5e-4
    alpha_min = 1e-5
    T_w = 700
    max_steps = 5000

    # 自动检测计算设备：优先使用 NVIDIA GPU (cuda)，其次 Apple Silicon (mps)，最后回退到 CPU
    device = "cuda" if torch.cuda.is_available() else "mps" \
        if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")

    data_path = "outputs/TinyStories_tokens.npy"
    if not os.path.exists(data_path):
        print("未找到数据，请先运行数据预处理脚本。")
        return
    data = np.load(data_path,mmap_mode='r')

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device
    )
    model.to(device)
    optimizer = AdamW(model.parameters(),lr = alpha_max,weight_decay=0.1)

    os.makedirs("checkpoints",exist_ok=True)
    model.train()

    for step in range(1,max_steps+1):
        lr = get_lr_cosine_schedule(step,alpha_max,alpha_min,T_w,max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x,y = get_batch(data,batch_size, context_length, device)

        logits = model(x)
        loss = cross_entropy(logits,y)

        optimizer.zero_grad()
        loss.backward()

        clip_gradients(model.parameters(), max_norm=1.0)

        optimizer.step()

        if step % 200 == 0:
            perplexity = compute_perplexity(loss)
            print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f}| Perplexity: {perplexity:.4f} | LR: {lr:.6f}")

        if step % 1000 == 0:
            save_checkpoint(model, optimizer, step, f"checkpoints/model_step_{step}.pt")
            print(f"已保存检查点至 checkpoints/model_step_{step}.pt")

if __name__ == '__main__':
    main()
























