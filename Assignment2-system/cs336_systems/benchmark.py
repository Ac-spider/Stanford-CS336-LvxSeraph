import sys
import argparse
import timeit
import torch

import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

# 将提供的绝对路径加入系统路径，确保能够顺利导入内部模块
#sys.path.append(r"C:\Users\liu_j\Desktop\SJTU\AI\LLM\CS336\assignment2-systems-main\cs336-basics")
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy,clip_gradient

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def get_model_and_data(size_name, batch_size, seq_len, device="cuda"):
    """
    初始化指定尺寸的模型，并生成用于基准测试的随机输入(x)和目标标签(y)
    """
    if size_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown size: {size_name}")

    cfg = MODEL_CONFIGS[size_name]
    vocab_size = 10000

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0  # 旋转位置编码 (RoPE) 的基数参数
    ).to(device)

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    return model, x, y

def run_step(model, x, y, optimizer, is_backward, amp_dtype):
    """
    执行单步的前向传播（可选包含反向传播和优化器更新）。
    """
    device = x.device

    autocast_ctx = torch.autocast(device_type='cuda',dtype=amp_dtype) if amp_dtype else nullcontext()

    with nvtx.range("Forward Pass"):
        with autocast_ctx:
            logits = model(x)

            loss = cross_entropy(logits,y)

    if is_backward:
        with nvtx.range("Backward Pass"):
            optimizer.zero_grad()
            loss.backward()

        with nvtx.range("Optimizer Step"):
            clip_gradient(model.parameters(), max_norm=1.0)
            optimizer.step()

    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=5,)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="fwd_bwd", choices=["fwd", "fwd_bwd"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--profile_memory", action="store_true", help="是否导出 PyTorch 内存分配历史的快照以供分析")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else None
    print(f"Initialize {args.model_size} Model (Batch={args.batch_size}, Seq={args.seq_len}, Dtype={args.dtype})...")

    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map[args.dtype]
    is_backward = (args.mode == "fwd_bwd")

    model, x, y = get_model_and_data(args.model_size, args.batch_size, args.seq_len, device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    print(f"run {args.warmup_steps}steps to warm up...")
    with nvtx.range("Warmup"):
        for _ in range(args.warmup_steps):
            run_step(model, x, y, optimizer, is_backward, amp_dtype)

    if args.profile_memory:
        print("开启 PyTorch 内存记录仪...")
        torch.cuda.memory._record_memory_history(max_entries=100000)

    print(f"执行 {args.measure_steps} 步性能测量...")
    times = []

    with nvtx.range("Measurement"):
        for i in range(args.measure_steps):
            start_time = timeit.default_timer()

            # 使用 NVTX 对每次迭代单独打标签，方便在 Nsight UI 中看到每次 iteration 的详细拆分
            with nvtx.range(f"Iteration_{i}"):
                run_step(model, x, y, optimizer, is_backward, amp_dtype)

            # 计算这一步的总耗时并保存
            times.append(timeit.default_timer() - start_time)

    if args.profile_memory:
        snapshot_file = f"memory_snapshot_{args.model_size}_{args.seq_len}.pickle"
        # 导出包含了所有内存分配/释放事件的 pickle 文件
        torch.cuda.memory._dump_snapshot(snapshot_file)
        # 关闭内存记录仪以停止消耗额外性能
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"内存快照已保存至: {snapshot_file} (可拖入 https://pytorch.org/memory_viz 查看)")

    avg_time = sum(times) / len(times)
    std_time = torch.tensor(times).std().item() if len(times) > 1 else 0.0
    print("-" * 40)
    print(f"Finish! | Mode: {args.mode} | Dtype: {args.dtype.upper()}")
    print(f"Avg_time: {avg_time:.4f} seconde/step | std: {std_time:.4f} second")
    print("-" * 40)


if __name__ == '__main__':
    main()






