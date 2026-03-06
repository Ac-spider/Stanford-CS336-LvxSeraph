import sys
import argparse
import timeit
import torch
# nvtx (NVIDIA Tools Extension) 用于在 GPU 性能分析工具（如 Nsight Systems）的时间轴上打标签，方便定位代码段
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

# 将提供的绝对路径加入系统路径，确保能够顺利导入内部模块
sys.path.append(r"C:\Users\liu_j\Desktop\SJTU\AI\LLM\CS336\assignment2-systems-main\cs336-basics")
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

# 模型规模配置字典，定义了不同量级 Transformer 的超参数
# d_model: 隐藏层维度, d_ff: 前馈神经网络内部维度, num_layers: Transformer 块的层数, num_heads: 注意力头数
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
        raise ValueError(f"未知的模型尺寸: {size_name}")

    cfg = MODEL_CONFIGS[size_name]
    vocab_size = 10000  # 词表大小设为 10000

    # 实例化自定义的 Transformer 语言模型，并将其移动到指定设备（通常是 GPU）
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0  # 旋转位置编码 (RoPE) 的基数参数
    ).to(device)

    # 构造随机输入与目标，模拟真实的 Token ID 输入
    # 形状为 (batch_size, seq_len)，数值范围在 [0, vocab_size-1] 之间
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    return model, x, y


def run_step(model, x, y, optimizer, is_backward, amp_dtype):
    """
    执行单步的前向传播（可选包含反向传播和优化器更新）。
    """
    device = x.device
    # 如果指定了混合精度类型 (fp16/bf16)，则启用 autocast 上下文管理器；否则使用 nullcontext（即不改变默认的 fp32 行为）
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) \
        if amp_dtype else nullcontext()

    # 前向传播标记，会在 Nsight Systems 等工具中显示为一段名为 "Forward Pass" 的时间带
    with nvtx.range("Forward Pass"):
        # 自动混合精度上下文：在这个区域内的计算会自动选择最合适的精度（如矩阵乘法用 fp16/bf16，累加用 fp32）以加速并节省显存
        with autocast_ctx:
            # logits 形状: (batch_size, seq_len, vocab_size)
            logits = model(x)

            # 计算交叉熵损失。PyTorch 的 cross_entropy 要求输入 logits 为 2D (N, C)，目标 y 为 1D (N)
            # 所以使用 view(-1) 将 batch_size 和 seq_len 维度展平合并
            loss = cross_entropy(
                logits,
                y
            )

    # 如果处于完整训练模式（包含反向传播）
    if is_backward:
        with nvtx.range("Backward Pass"):
            # 清空上一步的梯度，防止梯度累加
            optimizer.zero_grad()
            # 根据 loss 计算各参数的梯度
            loss.backward()

        with nvtx.range("Optimizer Step"):
            # 根据计算出的梯度更新模型权重
            optimizer.step()

    # 核心步骤：由于 GPU 的计算是异步的，Python CPU 线程提交完 GPU 任务后会立刻继续执行。
    # 为了保证使用 timeit 等工具测出的时间是准确的 GPU 执行时间，必须调用 synchronize 强制 CPU 等待当前 GPU 流上的所有任务完成。
    torch.cuda.synchronize()


def main():
    # 设置命令行参数解析器，方便在不修改代码的情况下测试不同配置
    parser = argparse.ArgumentParser(description="Transformer 模型性能与内存基准测试")
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="正式测速前的预热步数，用于消除 GPU 初始化和 CUDA 上下文创建的延迟")
    parser.add_argument("--measure_steps", type=int, default=10, help="正式测量的步数")
    parser.add_argument("--mode", type=str, default="fwd_bwd", choices=["fwd", "fwd_bwd"],
                        help="测试纯前向传播，还是完整的前向+反向传播")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="计算精度")
    parser.add_argument("--profile_memory", action="store_true", help="是否导出 PyTorch 内存分配历史的快照以供分析")
    args = parser.parse_args()

    # 确定计算设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"初始化 {args.model_size} 模型 (Batch={args.batch_size}, Seq={args.seq_len}, Dtype={args.dtype})...")

    # 获取模型和测试数据
    model, x, y = get_model_and_data(args.model_size, args.batch_size, args.seq_len, device)
    # 无论是否进行反向传播，都初始化 AdamW 优化器备用
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # 将字符串参数映射为实际的 PyTorch dtype 对象
    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map[args.dtype]
    is_backward = (args.mode == "fwd_bwd")

    # 预热阶段 (Warmup)
    # GPU 在刚启动、首次分配显存或首次执行特定 kernel 时会有额外开销。
    # 执行几次预热步骤可以确保这些初始化开销不被计入最终的性能测试结果中。
    print(f"执行 {args.warmup_steps} 步预热...")
    with nvtx.range("Warmup"):
        for _ in range(args.warmup_steps):
            run_step(model, x, y, optimizer, is_backward, amp_dtype)

    # 内存 Profiling 设置
    # 开启 PyTorch 的内存分配记录器，追踪底层 CUDA allocator 的行为（如 malloc, free）
    if args.profile_memory:
        print("开启 PyTorch 内存记录仪...")
        torch.cuda.memory._record_memory_history(max_entries=100000)

    # 测量阶段 (Measurement)
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

    # 保存内存快照
    if args.profile_memory:
        snapshot_file = f"memory_snapshot_{args.model_size}_{args.seq_len}.pickle"
        # 导出包含了所有内存分配/释放事件的 pickle 文件
        torch.cuda.memory._dump_snapshot(snapshot_file)
        # 关闭内存记录仪以停止消耗额外性能
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"内存快照已保存至: {snapshot_file} (可拖入 https://pytorch.org/memory_viz 查看)")

    # 统计输出
    # 计算平均每步耗时和标准差，标准差可以反映 GPU 运行时的稳定性
    avg_time = sum(times) / len(times)
    std_time = torch.tensor(times).std().item() if len(times) > 1 else 0.0
    print("-" * 40)
    print(f"测试完成 | 模式: {args.mode} | 精度: {args.dtype.upper()}")
    print(f"平均耗时: {avg_time:.4f} 秒/步 | 标准差: {std_time:.4f} 秒")
    print("-" * 40)


if __name__ == "__main__":
    main()