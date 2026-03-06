import math
import torch
import triton
import triton.language as tl


# 全局的 compiled 函数用于加速反向传播的执行
@torch.compile
def _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale):
    """
    执行带重计算的 FlashAttention 反向传播
    """
    # 1. 计算 D = rowsum(dO * O) -> 形状: (..., seq_q, 1)
    D = (dO * O).sum(dim=-1, keepdim=True)

    # 2. 重新计算注意力分数 S = Q @ K^T / sqrt(d) -> 形状: (..., seq_q, seq_k)
    S = (Q @ K.transpose(-2, -1)) * scale

    # 应用因果掩码
    if is_causal:
        seq_q = Q.shape[-2]
        seq_k = K.shape[-2]
        q_idx = torch.arange(seq_q, device=Q.device).unsqueeze(1)
        k_idx = torch.arange(seq_k, device=Q.device).unsqueeze(0)
        mask = q_idx >= k_idx
        # 用 -1e6 填充掩码位置，防止在混合精度下产生 NaN
        S = torch.where(mask, S, float('-1e6'))

    # 3. 重新计算注意力概率 P = exp(S - L)
    # L 的形状是 (..., seq_q)，需要 unsqueeze 以进行广播
    P = torch.exp(S - L.unsqueeze(-1))

    if is_causal:
        # 为了绝对准确，将掩码部分强制置为 0
        P = torch.where(mask, P, torch.tensor(0.0, dtype=P.dtype, device=P.device))

    # 4. 计算关于 V 的梯度: dV = P^T @ dO
    dV = P.transpose(-2, -1) @ dO

    # 5. 计算关于 P 的梯度: dP = dO @ V^T
    dP = dO @ V.transpose(-2, -1)

    # 6. 计算关于 S 的梯度: dS = P * (dP - D)
    dS = P * (dP - D)

    # 7. 计算关于 Q 和 K 的梯度
    dQ = (dS @ K) * scale
    dK = (dS.transpose(-2, -1) @ Q) * scale

    return dQ, dK, dV


# ==============================================================================
# (a) 纯 PyTorch 实现的 FlashAttention-2 前向与反向传播 (对照组)
# ==============================================================================
class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_heads, N_q, d = Q.shape
        _, _, N_k, _ = K.shape

        B_q = 32
        B_k = 32

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, n_heads, N_q), device=Q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)

        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(T_q):
                    q_start = i * B_q
                    q_end = min((i + 1) * B_q, N_q)
                    Qi = Q[b, h, q_start:q_end, :]

                    m_i = torch.full((q_end - q_start,), float('-inf'), device=Q.device)
                    l_i = torch.zeros((q_end - q_start,), device=Q.device)
                    O_i = torch.zeros((q_end - q_start, d), device=Q.device)

                    for j in range(T_k):
                        k_start = j * B_k
                        k_end = min((j + 1) * B_k, N_k)

                        if is_causal and k_start >= q_end:
                            continue

                        Kj = K[b, h, k_start:k_end, :]
                        Vj = V[b, h, k_start:k_end, :]

                        S_ij = (Qi @ Kj.transpose(-2, -1)) * scale

                        if is_causal:
                            q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                            k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                            mask = q_idx >= k_idx
                            S_ij = torch.where(mask, S_ij, float('-1e6'))

                        m_ij, _ = torch.max(S_ij, dim=-1)
                        m_new = torch.maximum(m_i, m_ij)

                        alpha = torch.exp(m_i - m_new)
                        beta = torch.exp(S_ij - m_new.unsqueeze(1))

                        l_i = l_i * alpha + torch.sum(beta, dim=-1)

                        P_ij = beta.to(V.dtype)
                        O_i = O_i * alpha.unsqueeze(1) + (P_ij @ Vj)

                        m_i = m_new

                    O[b, h, q_start:q_end, :] = O_i / l_i.unsqueeze(1)
                    L[b, h, q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = 1.0 / math.sqrt(Q.shape[-1])

        # 直接调用预编译的辅助函数
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale)

        return dQ, dK, dV, None


# ==============================================================================
# (b & c) FlashAttention-2 Triton 内核与 Autograd 封装
# ==============================================================================
@triton.autotune(
        configs=[
            triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4),
            triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_warps=8),
            triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 128}, num_warps=4),
            triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8),
        ],
        key=['N_QUERIES', 'N_KEYS', 'D'],
    )
@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
):
    batch_index = tl.program_id(1)
    query_tile_index = tl.program_id(0)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    m_i = tl.full([Q_TILE_SIZE], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    O_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        loop_end = (query_tile_index + 1) * Q_TILE_SIZE
        num_key_tiles = tl.cdiv(tl.minimum(loop_end, N_KEYS), K_TILE_SIZE)

    for j in range(num_key_tiles):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_ij = tl.dot(q, k, out_dtype=tl.float32) * scale

        if is_causal:
            q_offset = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offset = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offset[:, None] >= k_offset[None, :]
            S_ij = tl.where(mask, S_ij, -1e6)

        m_ij = tl.max(S_ij, 1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(S_ij - m_new[:, None])

        l_i = l_i * alpha + tl.sum(beta, 1)

        P_ij = beta.to(v.type.element_ty)
        O_i = O_i * alpha[:, None] + tl.dot(P_ij, v, out_dtype=tl.float32)

        m_i = m_new
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_ptr.type.element_ty), boundary_check=(0, 1))

    l_offsets = batch_index * stride_lb + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    l_mask = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES
    tl.store(L_ptr + l_offsets, L_i, mask=l_mask)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_size, n_heads, seq_len_q, d = q.shape
        _, _, seq_len_k, _ = k.shape

        q = q.view(batch_size * n_heads, seq_len_q, d)
        k = k.view(batch_size * n_heads, seq_len_k, d)
        v = v.view(batch_size * n_heads, seq_len_k, d)

        o = torch.empty_like(q)
        L = torch.empty((batch_size * n_heads, seq_len_q), device=q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        grid = (triton.cdiv(seq_len_q, Q_TILE_SIZE), batch_size * n_heads)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        flash_fwd_kernel[grid](
            q, k, v,
            o, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1),
            seq_len_q, seq_len_k,
            scale,
            is_causal=is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        o = o.view(batch_size, n_heads, seq_len_q, d)
        L = L.view(batch_size, n_heads, seq_len_q)
        q = q.view(batch_size, n_heads, seq_len_q, d)
        k = k.view(batch_size, n_heads, seq_len_k, d)
        v = v.view(batch_size, n_heads, seq_len_k, d)

        ctx.save_for_backward(q, k, v, o, L)
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = 1.0 / math.sqrt(Q.shape[-1])

        # 调用与 Pytorch 版本相同的编译后辅助函数
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale)

        return dQ, dK, dV, None