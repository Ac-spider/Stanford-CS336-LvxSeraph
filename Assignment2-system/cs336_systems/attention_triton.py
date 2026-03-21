import math
import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────
# FlashAttention-2 反向传播（PyTorch 编译版）
# Purpose: 复用正向传播保存的 L（log-sum-exp）直接重建 P，避免再次分块循环
# Key concept: D_i = rowsum(dO ⊙ O) 是每行的"标量修正项"，用于计算 dS = P⊙(dP-D)
#   公式推导：
#     dV  = P^T @ dO
#     dP  = dO @ V^T
#     dS  = P ⊙ (dP - D)   ← softmax 的 Jacobian-vector product
#     dQ  = dS @ K * scale
#     dK  = dS^T @ Q * scale
# ─────────────────────────────────────────
@torch.compile()
def _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale):
    # D_i：每个 query 位置的修正标量，等于 dO 与 O 的逐元素积按行求和
    D = (dO * O).sum(dim=-1, keepdim=True)

    # 重建注意力得分 S = Q @ K^T * scale
    S = (Q @ K.transpose(-2, -1)) * scale
    if is_causal:
        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        # 构造因果掩码：query 位置 >= key 位置才允许关注
        q_idx = torch.arange(n_q,device=Q.device).unsqueeze(-1)
        k_idx = torch.arange(n_k,device=Q.device).unsqueeze(0)
        mask = q_idx>=k_idx
        S = torch.where(mask,S, float('-1e6'))

    # 利用正向保存的 L（log-sum-exp）重建 softmax 概率矩阵 P
    # P_ij = exp(S_ij - L_i)，L 已包含 max 和 log(sum_exp)
    P = torch.exp(S - L.unsqueeze(-1))

    if is_causal:
        P = torch.where(mask, P, 0)

    # 反向传播各梯度计算
    dV = P.transpose(-2, -1) @ dO          # dV：来自 O = P @ V 的梯度
    dP = dO @ V.transpose(-2, -1)          # dP：来自 O = P @ V 对 P 的梯度
    dS = P * (dP - D)                      # dS：softmax 反向，利用 D 消去 Jacobian 中的求和项
    dQ = (dS @ K) * scale                  # dQ：缩放点积注意力对 Q 的梯度
    dK = (dS.transpose(-2, -1) @ Q) * scale  # dK：对 K 的梯度

    return dQ,dK,dV


# ─────────────────────────────────────────
# FlashAttention-2 PyTorch 实现（分块 + 在线 Softmax）
# Purpose: 用 Python 循环模拟 FlashAttention 的分块计算，验证算法正确性
# Key concept: 显存复杂度 O(N) vs 标准注意力 O(N²)
#   在线 Softmax 维护两个累计量：
#     m_i：当前见到的最大值（用于数值稳定）
#     l_i：归一化分母的累计和
#   每次处理新 K/V 块时用 alpha=exp(m_old-m_new) 修正旧值
# ─────────────────────────────────────────
class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_causual=False):
        batch,n_heads,N_q,d = Q.shape
        _,_,N_k,_ = K.shape

        # 分块大小：Q 块和 K/V 块各 32 行
        B_q = 32
        B_k = 32

        # 计算块数（向上取整）
        T_q = math.ceil(N_q/B_q)
        T_k = math.ceil(N_k/B_k)

        # 输出矩阵 O 和 log-sum-exp 缓存 L（反向传播使用）
        O = torch.zeros_like(Q)
        L = torch.zeros((batch,n_heads,N_q),device=Q.device,dtype=torch.float32)
        scale = 1/math.sqrt(d)  # 注意力缩放因子 1/√d

        for b in range(batch):
            for n in range(n_heads):
                # 外循环：遍历每个 Q 块（每块独立完成自己的 softmax）
                for i in range(T_q):
                    q_start = i * B_q
                    q_end = min((i+1)*B_q,N_q)

                    q_i = Q[b,n,q_start:q_end,:]
                    # 在线 Softmax 状态初始化
                    l_i = torch.zeros((q_end-q_start,),device=Q.device)          # 归一化分母
                    m_i =torch.full((q_end-q_start,),float('inf'),device=Q.device)  # 运行最大值
                    o_i = torch.zeros((q_end-q_start,d),device=Q.device)          # 累计输出

                    # 内循环：遍历每个 K/V 块，增量更新 softmax 状态
                    for j in range(T_k):
                        k_start = j * B_k
                        k_end = min((j+1)*B_k,N_k)

                        # 因果掩码：跳过完全在当前 Q 块右侧的 K 块
                        if is_causual and k_start >= q_end:
                            continue

                        K_j = K[b,n,k_start:k_end,:]
                        V_j = V[b, n, k_start:k_end, :]

                        # 计算当前块的注意力得分
                        S_ij = q_i @ K_j.transpose(-2,-1) * scale

                        if is_causual:
                            # 精细因果掩码：逐元素判断 query_idx >= key_idx
                            q_idx = torch.arange(q_start,q_end,device=Q.device).unsqueeze(-1)
                            k_idx = torch.arange(k_start,k_end,device=Q.device).unsqueeze(0)
                            mask = q_idx>=k_idx
                            S_ij = torch.where(mask,S_ij,-1e6)

                        # 在线 Softmax 更新
                        m_ij,_= torch.max(S_ij,-1)           # 当前块的最大值
                        m_new = torch.maximum(m_i,m_ij)      # 更新全局最大值

                        # alpha：用于修正旧累计量（参考点从 m_i 变为 m_new）
                        alpha = torch.exp(m_i-m_new)
                        # beta：当前块相对于新参考点的 exp 值
                        beta = torch.exp(S_ij - m_new.unsqueeze(-1))

                        # 更新归一化分母和累计输出
                        l_i = alpha * l_i + beta
                        P_ij = beta.to(V.dtype)
                        o_i = alpha.unsqueeze(-1) * o_i + P_ij@V_j

                        m_i = m_new

                    # 保存 log-sum-exp：L = log(l) + m（用于反向传播重建 P）
                    L[b,n,q_start:q_end] = torch.log(l_i) + m_i
                    # 归一化输出
                    O[b,n,q_start:q_end,:] = o_i/l_i.unsqueeze(-1)

        # 保存反向传播所需张量
        ctx.save_for_backward(Q,K,V,L,O)
        ctx.is_causual = is_causual

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = 1.0 / math.sqrt(Q.shape[-1])

        # 调用预编译的辅助函数完成反向传播（复用 L 避免重新分块）
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale)

        return dQ, dK, dV, None


# ─────────────────────────────────────────
# FlashAttention-2 Triton GPU 内核（前向）
# Purpose: 在 GPU SRAM 内完成分块注意力计算，HBM 读写从 O(N²) 降至 O(N)
# Key concept:
#   - tl.make_block_ptr：声明 HBM→SRAM 的分块指针，自动处理越界填充
#   - @triton.autotune：自动搜索最优 tile 大小（Q_TILE_SIZE × K_TILE_SIZE）
#   - tl.dot：在 SRAM 内执行矩阵乘法（利用 Tensor Core）
#   - tl.advance：移动块指针到下一个 K/V 块
# ─────────────────────────────────────────
@triton.autotune(
        configs=[
            # 搜索不同 tile 大小与 warp 数量的最优组合
            triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4),
            triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_warps=8),
            triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 128}, num_warps=4),
            triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8),
        ],
        key=['N_QUERIES', 'N_KEYS', 'D'],  # 根据序列长度和维度选择最优配置
    )
@triton.jit
def flash_fwd_kernel(
        # HBM 中各矩阵的基地址指针
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        # 各维度步长（元素数），用于计算多维索引偏移
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        # 编译期常量（constexpr）：允许编译器展开循环、优化寄存器分配
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,  # Q 块大小（由 autotune 决定）
        K_TILE_SIZE: tl.constexpr,  # K/V 块大小（由 autotune 决定）
):
    # 每个 Triton program 处理一个 (batch*head, Q_tile) 的组合
    batch_index = tl.program_id(1)        # 第二维：batch × head 的展平索引
    query_tile_index = tl.program_id(0)   # 第一维：Q 块索引

    # 构造 Q 块指针：指向当前 program 负责的 Q 行
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # 行优先连续存储
    )

    # K 块指针：shape 为 (D, N_KEYS) 即转置形式，配合 tl.dot(q, k) = Q @ K^T
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),  # 列优先，配合转置形状
    )

    # V 块指针：从 K/V 序列头部开始，随内循环推进
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # O 块指针：用于将最终结果写回 HBM
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # 在线 Softmax 状态（存于 SRAM 寄存器，避免 HBM 读写）
    m_i = tl.full([Q_TILE_SIZE],-float("inf"),dtype=tl.float32)  # 运行最大值
    l_i = tl.zeros([Q_TILE_SIZE],dtype=tl.float32)               # 归一化分母
    O_i = tl.zeros([Q_TILE_SIZE,D],dtype=tl.float32)             # 累计输出

    # 将当前 Q 块从 HBM 加载到 SRAM（boundary_check 处理边界填充）
    q = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")

    # 确定内循环次数：因果模式下只需处理到当前 Q 块结束位置
    num_key_tiles = tl.cdiv(N_KEYS,K_TILE_SIZE)
    if is_causal:
        loop_end = (query_tile_index+1)*Q_TILE_SIZE
        num_key_tiles = tl.cdiv(tl.minimum(loop_end, N_KEYS), K_TILE_SIZE)

    # 内循环：逐 K/V 块更新在线 Softmax 状态（所有中间结果留在 SRAM）
    for j in range(num_key_tiles):
        # 从 HBM 加载当前 K/V 块到 SRAM
        k = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        v = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")

        # 计算注意力得分：S = Q @ K^T * scale（tl.dot 利用 Tensor Core）
        S_ij = tl.dot(q,k,out_dtype=tl.float32)*scale

        if is_causal:
            # 因果掩码：将未来位置置为 -1e6（近似 -∞）
            q_offset = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offset = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offset[:,None]>=k_offset[None,:]
            S_ij = tl.where(mask,S_ij,-1e6)

        # 在线 Softmax 更新（同 PyTorch 版算法）
        m_ij = tl.max(S_ij,-1)
        m_new = tl.maximum(m_i,m_ij)

        alpha = tl.exp(m_i-m_new)           # 旧累计量的修正因子
        beta = tl.exp(S_ij-m_new[:,None])   # 当前块相对于新最大值的 exp

        l_i = l_i * alpha + tl.sum(beta,-1)
        P_ij = beta.to(v.type.element_ty)
        O_i = O_i * alpha[:, None] + tl.dot(P_ij, v, out_dtype=tl.float32)

        m_i = m_new

        # 推进块指针到下一个 K/V 块（避免重新计算偏移量）
        K_block_ptr = tl.advance(K_block_ptr,(0,K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr,(K_TILE_SIZE,0))

    # 最终归一化并计算 log-sum-exp
    O_i = O_i / l_i[:,None]
    L_i = m_i + tl.log(l_i)  # L = log(l) + m，紧凑形式存储 log-sum-exp

    # 将结果写回 HBM（只写一次，这正是 FlashAttention 节省带宽的关键）
    tl.store(O_block_ptr,O_i.to(O_ptr.type.element_ty),boundary_check=(0,1))

    # 存储 L（逐元素写入，需手动计算偏移和掩码）
    l_offsets = batch_index*stride_lb + query_tile_index*Q_TILE_SIZE +tl.arange(0,Q_TILE_SIZE)
    l_mask = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES
    tl.store(L_ptr+l_offsets,L_i,mask = l_mask)


# ─────────────────────────────────────────
# FlashAttention-2 Triton 封装类
# Purpose: 将 Triton 内核包装为标准 PyTorch autograd Function
# Key concept: 将 (batch, n_heads, seq, d) 展平为 (batch*n_heads, seq, d)
#   以利用 Triton 内核的二维 grid：grid[0]=Q块数, grid[1]=batch*heads
# ─────────────────────────────────────────
class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_size, n_heads, seq_len_q, d = q.shape
        _, _, seq_len_k, _ = k.shape

        # 合并 batch 和 head 维度，简化 Triton 内核的 grid 设计
        q = q.view(batch_size * n_heads, seq_len_q, d)
        k = k.view(batch_size * n_heads, seq_len_k, d)
        v = v.view(batch_size * n_heads, seq_len_k, d)

        o = torch.empty_like(q)
        L = torch.empty((batch_size * n_heads, seq_len_q), device=q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)

        # grid：第 0 维 = Q 块数，第 1 维 = batch * heads
        grid = lambda META: (triton.cdiv(seq_len_q,META['Q_TILE_SIZE']),batch_size*n_heads)

        # 确保内存连续（Triton 要求连续内存布局）
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # 启动 Triton 内核（传入步长而非形状，支持非连续张量）
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
        )

        # 恢复原始四维形状
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

        # 调用与 PyTorch 版本相同的编译后辅助函数（两个实现共享反向传播逻辑）
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale)

        return dQ, dK, dV, None
