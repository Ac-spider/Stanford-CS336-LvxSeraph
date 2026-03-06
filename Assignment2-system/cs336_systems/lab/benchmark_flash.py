import sys
import torch
import triton
import pandas as pd
import gc

# 请根据你的实际路径调整
sys.path.append(r"C:\Users\liu_j\Desktop\SJTU\AI\LLM\CS336\assignment2-systems-main\cs336-basics")
from cs336_basics.model import scaled_dot_product_attention
from attention_triton import FlashAttention2Triton


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


def run_benchmark():
    batch_size = 1
    n_heads = 8  # 使用 8 个注意力头进行模拟测试

    # 定义搜索空间
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_heads = [16, 32, 64, 128]
    dtypes = [torch.float32, torch.bfloat16]

    results = []

    # 获取我们的 Triton 算子
    flash_triton = FlashAttention2Triton.apply

    print("开始运行 FlashAttention vs PyTorch 性能对比测试...")
    print("注意: 较大的 Seq Length 可能会在原生 PyTorch 中导致 OOM (显存溢出)，这是正常现象，脚本会捕获并标记为 'OOM'。\n")

    for dtype in dtypes:
        for d in d_heads:
            for seq in seq_lens:
                config_name = f"Seq={seq}, d={d}, {str(dtype).split('.')[-1]}"
                print(f"Testing: {config_name}")

                # 初始化随机输入
                q = torch.randn(batch_size, n_heads, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                k = torch.randn(batch_size, n_heads, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                v = torch.randn(batch_size, n_heads, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                dout = torch.randn_like(q)

                # PyTorch 的因果掩码
                mask = torch.tril(torch.ones(seq, seq, device='cuda', dtype=torch.bool))

                row = {
                    "Dtype": str(dtype).split('.')[-1],
                    "Seq_Len": seq,
                    "d_head": d,
                }

                # ==========================================
                # 1. 测算 PyTorch 原生实现
                # ==========================================
                clear_cache()
                try:
                    # Fwd
                    row["Torch_Fwd(ms)"] = triton.testing.do_bench(
                        lambda: scaled_dot_product_attention(q, k, v, mask),
                        quantiles=[0.5]
                    )[0]

                    # 准备 Bwd 的中间变量
                    out_pt = scaled_dot_product_attention(q, k, v, mask)
                    row["Torch_Bwd(ms)"] = triton.testing.do_bench(
                        lambda: out_pt.backward(dout, retain_graph=True),
                        quantiles=[0.5]
                    )[0]

                    # Fwd + Bwd
                    def pt_fwd_bwd():
                        out = scaled_dot_product_attention(q, k, v, mask)
                        out.backward(dout)

                    row["Torch_E2E(ms)"] = triton.testing.do_bench(pt_fwd_bwd, quantiles=[0.5])[0]

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        row["Torch_Fwd(ms)"] = "OOM"
                        row["Torch_Bwd(ms)"] = "OOM"
                        row["Torch_E2E(ms)"] = "OOM"
                    else:
                        raise e

                # ==========================================
                # 2. 测算 Triton FlashAttention-2
                # ==========================================
                clear_cache()
                try:
                    # Fwd
                    row["Triton_Fwd(ms)"] = triton.testing.do_bench(
                        lambda: flash_triton(q, k, v, True),
                        quantiles=[0.5]
                    )[0]

                    out_tr = flash_triton(q, k, v, True)
                    row["Triton_Bwd(ms)"] = triton.testing.do_bench(
                        lambda: out_tr.backward(dout, retain_graph=True),
                        quantiles=[0.5]
                    )[0]

                    def tr_fwd_bwd():
                        out = flash_triton(q, k, v, True)
                        out.backward(dout)

                    row["Triton_E2E(ms)"] = triton.testing.do_bench(tr_fwd_bwd, quantiles=[0.5])[0]

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        row["Triton_Fwd(ms)"] = "OOM"
                        row["Triton_Bwd(ms)"] = "OOM"
                        row["Triton_E2E(ms)"] = "OOM"
                    else:
                        raise e

                # 避免残留梯度影响后续测试
                q.grad, k.grad, v.grad = None, None, None
                results.append(row)

    # 格式化并输出表格
    df = pd.DataFrame(results)

    # 转换为 Markdown 格式便于放入作业文档
    print("\n" + "=" * 80)
    print("Benchmarking Results (Markdown)")
    print("=" * 80)
    print(df.to_markdown(index=False))

    with open("benchmark_flash.md", "w", encoding="utf-8") as f:
        f.write("# Benchmarking Results\n\n")
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()