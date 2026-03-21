import math
import torch
import torch.nn as nn
from torch import einsum
import einops

# ─────────────────────────────────────────
# 自定义线性层（无偏置）
# Purpose: 替代 nn.Linear，使用 Xavier 均匀初始化的截断正态分布
# Key concept: 权重初始化——std = sqrt(2/(fan_in+fan_out)) 平衡前向激活方差与反向梯度方差
# ─────────────────────────────────────────
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # W 形状为 (out, in)，便于矩阵乘法
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # Xavier 初始化标准差：平衡输入输出维度
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3, b=3)

    def forward(self, x):
        # einsum 实现通用批量矩阵乘法：...i × oi -> ...o
        return einsum('...i,oi->...o', x, self.W)


# ─────────────────────────────────────────
# 词嵌入层
# Purpose: 将离散 token ID 映射为连续向量空间中的稠密表示
# Key concept: Embedding——查表操作，可理解为 one-hot × 权重矩阵的高效实现
# ─────────────────────────────────────────
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # weight 形状：(词表大小, 嵌入维度)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 直接用 token_ids 作索引，等价于查表
        return self.weight[token_ids]


# ─────────────────────────────────────────
# RMS 归一化
# Purpose: 替代 LayerNorm，更简洁且效果相当；被 LLaMA/GPT-NeoX 等模型广泛采用
# Key concept: RMSNorm——只做均方根归一化（不减均值），计算量更小，无中心化偏置
#   公式：x_norm = x / sqrt(mean(x²) + eps) * γ
# ─────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps  # 防止除以零的小常数
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))  # 可学习缩放因子 γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        # 强制在 float32 下计算，避免 fp16 精度损失
        x = x.to(torch.float32)

        # 计算均方根：沿最后一维求均值后开方
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight  # 归一化后乘以可学习缩放参数

        return result.to(in_type)  # 恢复原始数据类型


# ─────────────────────────────────────────
# SwiGLU 前馈网络
# Purpose: Transformer 中的 FFN 子层，提升非线性表达能力
# Key concept: SwiGLU 激活函数——Swish(x1) × x3，门控机制让网络自适应选择信息
#   公式：output = W2(SiLU(W1·x) ⊗ W3·x)，其中 SiLU(x) = x·σ(x)
# ─────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, d_ff, d_model, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 门控路径投影
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 值路径投影
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # 输出投影，降维回 d_model

    def forward(self, x):
        x1 = self.w1(x)  # 门控分支
        x3 = self.w3(x)  # 值分支

        # SiLU 激活（Sigmoid Linear Unit）：x * σ(x)
        silu = x1 * torch.sigmoid(x1)

        # 门控相乘后投影回原始维度
        return self.w2(silu * x3)


# ─────────────────────────────────────────
# 旋转位置编码（Rotary Positional Embedding, RoPE）
# Purpose: 将位置信息编码为旋转矩阵，使注意力分数天然感知相对位置
# Key concept: RoPE——对 Q/K 的每对相邻维度施加以位置为角度的旋转变换
#   原理：q_i·k_j 的内积仅依赖 (i-j)，实现相对位置编码
#   公式：[x1, x2] -> [x1·cos(θ) - x2·sin(θ), x1·sin(θ) + x2·cos(θ)]
# ─────────────────────────────────────────
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        # 计算每对维度的旋转频率：inv_freq[i] = 1 / theta^(2i/d_k)
        inv_freq = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        # t: 位置序列 [0, 1, ..., max_seq_len-1]
        t = torch.arange(0, max_seq_len, device=device)
        # 外积得到每个位置、每对维度的旋转角度矩阵
        freqs = einsum('i,j->ij', t, inv_freq)  # 形状：(max_seq_len, d_k/2)

        # 预计算 cos/sin 并缓存（不作为参数更新）
        self.register_buffer('cos_cache', freqs.cos(), persistent=False)
        self.register_buffer('sin_cache', freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 按 token 位置取出对应的 cos/sin 值
        cos = self.cos_cache[token_positions]  # (..., seq, d_k/2)
        sin = self.sin_cache[token_positions]

        # 将最后一维按对 reshape，便于旋转操作
        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # (..., seq, d_k/2, 2)
        x1, x2 = x_reshaped.unbind(dim=-1)          # 拆分每对中的两个分量

        # 旋转变换：(-x2, x1) 即对应旋转矩阵的另一列
        rotated = torch.stack((-x2, x1), dim=-1)

        # 旋转后的向量：x*cos + rotated*sin
        out = (x_reshaped * cos.unsqueeze(-1) + rotated * sin.unsqueeze(-1))

        return out.view_as(x)  # 恢复原始形状


# ─────────────────────────────────────────
# 数值稳定的 Softmax
# Purpose: 将注意力分数转为概率分布
# Key concept: 减去最大值避免 exp 溢出（数学上等价，但数值更稳定）
#   公式：softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
# ─────────────────────────────────────────
def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]  # 取最大值用于数值稳定
    x_exp = torch.exp(x - x_max)                    # 减去最大值后再取 exp
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


# ─────────────────────────────────────────
# 缩放点积注意力
# Purpose: 计算注意力权重并加权聚合 Value
# Key concept: Scaled Dot-Product Attention
#   公式：Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
#   除以 √d_k 防止内积值随维度增大而过大，导致梯度消失
# ─────────────────────────────────────────
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor = None) -> torch.Tensor:
    d_k = q.size(-1)
    # 计算注意力分数：Q·Kᵀ / √d_k
    scores = einsum('...nd,...md -> ...nm', q, k) / math.sqrt(d_k)

    if mask is not None:
        # 将被 mask 位置设为 -inf，softmax 后趋近于 0（因果遮蔽）
        scores = scores.masked_fill(~mask, float('-inf'))

    # softmax 得到注意力权重
    attention_scores = softmax(scores, dim=-1)

    # 加权求和 Value，得到输出表示
    return einsum('...nm,...md -> ...nd', attention_scores, v)


# ─────────────────────────────────────────
# 因果多头自注意力
# Purpose: Transformer 的核心组件，让每个位置关注其前方所有位置（因果约束）
# Key concept: Multi-Head Attention——将 Q/K/V 分割为多个头并行计算，再拼接输出
#   因果遮蔽（causal mask）确保位置 i 只能看到位置 ≤ i 的信息（自回归特性）
# ─────────────────────────────────────────
class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int,
                 theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的键/查询维度

        # 合并 Q/K/V 投影为一个矩阵，提高计算效率
        self.qkv_proj = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        # 输出投影：将所有头的输出拼接后线性变换
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

        # 旋转位置编码（共享于所有头）
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

        # 因果遮蔽矩阵：下三角为 True，上三角为 False（预计算，不随训练更新）
        self.register_buffer(
            'mask',
            torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)),
            persistent=False
        )

    def forward(self, x, token_positions):
        seq_len = x.size(-2)

        # 一次投影得到 Q/K/V，沿最后一维三等分
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 将序列维度和头维度分离：(batch, seq, d_model) -> (batch, heads, seq, d_k)
        q = einops.rearrange(q, '... seq (h d) -> ... h seq d', h=self.num_heads)
        k = einops.rearrange(k, '... seq (h d) -> ... h seq d', h=self.num_heads)
        v = einops.rearrange(v, '... seq (h d) -> ... h seq d', h=self.num_heads)

        # 为头维度扩展 token_positions，对 Q/K 施加 RoPE
        pos_extended = token_positions.unsqueeze(-2)
        q = self.rope(q, pos_extended)
        k = self.rope(k, pos_extended)

        # 截取对应序列长度的因果遮蔽矩阵
        mask = self.mask[:seq_len, :seq_len]
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 将多头输出拼接回 d_model 维度
        out = einops.rearrange(out, '... h seq d -> ... seq (h d)', h=self.num_heads)

        return self.w_o(out)


# ─────────────────────────────────────────
# Transformer Block（Pre-Norm 结构）
# Purpose: 堆叠多层构成完整 Transformer，每层包含注意力子层和 FFN 子层
# Key concept: Pre-Norm + 残差连接——先归一化再做子层计算，训练更稳定
#   公式：x = x + Attn(RMSNorm(x))
#         x = x + FFN(RMSNorm(x))
# ─────────────────────────────────────────
class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)  # 注意力前的归一化
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)  # FFN 前的归一化
        self.fnn = SwiGLU(d_ff, d_model, device, dtype)

    def forward(self, x, token_positions):
        # 残差连接1：注意力子层（Pre-Norm）
        x = x + self.attn(self.norm1(x), token_positions)
        # 残差连接2：前馈子层（Pre-Norm）
        x = x + self.fnn(self.norm2(x))
        return x


# ─────────────────────────────────────────
# 完整的 Transformer 语言模型
# Purpose: 给定 token ID 序列，预测每个位置下一个 token 的概率分布（自回归 LM）
# Key concept: 语言建模——最大化 P(x_{t+1} | x_1, ..., x_t)，等价于最小化交叉熵损失
# ─────────────────────────────────────────
class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        # 词嵌入层：token ID -> d_model 维向量
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        # 堆叠 num_layers 个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        # 最终输出前的 RMSNorm
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        # 语言模型头：d_model -> vocab_size（输出每个 token 的 logit）
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        # 生成位置编号：[0, 1, ..., seq_len-1]，扩展到 batch 维度
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand_as(token_ids)

        # 词嵌入
        x = self.embedding(token_ids)
        # 逐层 Transformer Block
        for block in self.blocks:
            x = block(x, token_positions)
        # 最终归一化
        x = self.norm(x)
        # 线性投影到词表大小，得到每个位置的 logit
        logits = self.lm_head(x)

        return logits  # 形状：(batch, seq_len, vocab_size)
