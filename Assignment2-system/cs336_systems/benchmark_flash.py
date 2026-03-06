import sys
import torch
import triton
import pandas as pd
import gc

sys.path.append("/mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/CS336/assignment2-systems-main/cs336-basics")
from cs336_basics.model import scaled_dot_product_attention
from attention_triton import FlashAttention2Triton

flash_triton = FlashAttention2Triton.apply

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()

def run_benchmark():
    batch_size = 1
    n_heads = 8

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_heads = [16, 32, 64,128]
    dtypes = [torch.float32, torch.bfloat16]

    results = []

    print("开始运行 FlashAttention vs PyTorch 性能对比测试...")

    for dtype in dtypes:
        for d in d_heads:
            for seq in seq_lens:
                config_name = f"Seq={seq}, d={d}, dtype={str(dtype).split('.')[-1]}"
                print(f"Testing: {config_name}")

                q = torch.randn(batch_size, n_heads, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                k = torch.randn(batch_size, n_heads, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                v = torch.randn(batch_size, n_heads, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                dout = torch.randn_like(q)

                mask = torch.tril(torch.ones((seq,seq),device='cuda',dtype=torch.bool))

                row = {
                    "Dtype": str(dtype).split('.')[-1],
                    "Seq_Len": seq,
                    "d_head": d,
                }


                # 1. 测算 PyTorch 原生实现
                clear_cache()
                try:
                    # Fwd
                    row["Torch_Fwd(ms)"] = triton.testing.do_bench(
                        lambda: scaled_dot_product_attention(q, k, v, mask),
                        quantiles=[0.5]
                    )

                    #Bwd
                    out_pt = scaled_dot_product_attention(q, k, v, mask)
                    row["Torch_Bwd(ms)"] = triton.testing.do_bench(
                        lambda: out_pt.backward(dout, retain_graph=True),
                        quantiles=[0.5]
                    )

                    # Fwd + Bwd
                    def pt_fwd_bwd():
                        out = scaled_dot_product_attention(q, k, v, mask)
                        out.backward(dout)

                    row["Torch_E2E(ms)"] = triton.testing.do_bench(pt_fwd_bwd, quantiles=[0.5])


                except Exception as e:


                    error_msg = str(e).lower()

                    # 同时匹配 PyTorch 的全局显存 OOM 和 Triton 的共享内存溢出

                    if "out of memory" in error_msg or "out of resource" in error_msg:

                        pass

                    else:

                        raise e


                # 2. 测算 Triton FlashAttention-2
                clear_cache()
                try:
                    # Fwd
                    row["Triton_Fwd(ms)"] = triton.testing.do_bench(
                        lambda: flash_triton(q, k, v, True),
                        quantiles=[0.5]
                    )

                    out_tr = flash_triton(q, k, v, True)
                    row["Triton_Bwd(ms)"] = triton.testing.do_bench(
                        lambda: out_tr.backward(dout, retain_graph=True),
                        quantiles=[0.5]
                    )

                    def tr_fwd_bwd():
                        out = flash_triton(q, k, v, True)
                        out.backward(dout)

                    row["Triton_E2E(ms)"] = triton.testing.do_bench(tr_fwd_bwd, quantiles=[0.5])


                except Exception as e:

                    error_msg = str(e).lower()

                    # 同时匹配 PyTorch 的全局显存 OOM 和 Triton 的共享内存溢出

                    if "out of memory" in error_msg or "out of resource" in error_msg:

                        pass

                    else:

                        raise e

                # 避免残留梯度影响后续测试
                q.grad, k.grad, v.grad = None, None, None
                results.append(row)

                try:
                    del q, k, v, dout, mask
                    # 如果有定义的话一并删除
                    if 'out_pt' in locals(): del out_pt
                    if 'out_tr' in locals(): del out_tr
                except NameError:
                    pass
                clear_cache()

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("Benchmarking Results (Markdown)")
    print("=" * 80)
    print(df.to_markdown(index=False))

    with open("benchmark_flash.md", "w", encoding="utf-8") as f:
        f.write("# Benchmarking Results\n\n")
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    run_benchmark()
