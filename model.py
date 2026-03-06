import math
import torch
import torch.nn as nn
from torch import einsum
import einops

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.W,mean=0,std=std,a=-3,b=3)
    def forward(self,x):
        return einsum('...i,oi->...o',x,self.W)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        result = x/rms * self.weight

        return result.to(in_type)

class SwiGLU(nn.Module):
    def __init__(self,d_ff,d_model,device = None,dtype = None):
        super().__init__()
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)

    def forward(self,x):
        x1 = self.w1(x)
        x3 = self.w3(x)

        silu = x1 * torch.sigmoid(x1)

        return self.w2(silu * x3)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        inv_freq = 1 / (theta**(torch.arange(0,d_k,2,device=device)/d_k))
        t = torch.arange(0,max_seq_len,device=device)
        freqs = einsum('i,j->ij',t,inv_freq)

        self.register_buffer('cos_cache',freqs.cos(),persistent=False)
        self.register_buffer('sin_cache',freqs.sin(),persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_reshaped = x.view(*x.shape[:-1],-1,2)
        x1,x2 = x_reshaped.unbind(dim=-1)
        rotated = torch.stack((-x2,x1),dim=-1)

        out = (x_reshaped * cos.unsqueeze(-1) + rotated * sin.unsqueeze(-1))

        return out.view_as(x)

def softmax(x,dim):
    x_max = torch.max(x,dim=dim,keepdim=True)[0]
    x_exp = torch.exp(x-x_max)
    return x_exp / torch.sum(x_exp,dim=dim,keepdim=True)

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor = None) -> torch.Tensor:
    d_k = q.size(-1)
    scores = einsum('...nd,...md -> ...nm',q,k)/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask,float('-inf'))

    attention_scores = softmax(scores,dim=-1)

    return einsum('...nm,...md -> ...nd',attention_scores,v)


class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0, device=None,
                 dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model,d_model*3,device=device,dtype=dtype)
        self.w_o = Linear(d_model,d_model,device=device,dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta,self.d_k,max_seq_len,device = device)

        self.register_buffer('mask',torch.tril(torch.ones((max_seq_len,max_seq_len),dtype = torch.bool)),persistent=False)

    def forward(self,x,token_positions):
        seq_len = x.size(-2)

        qkv = self.qkv_proj(x)

        q,k,v = torch.chunk(qkv,3,dim=-1)

        q = einops.rearrange(q,'... seq (h d) -> ... h seq d', h = self.num_heads)
        k = einops.rearrange(k, '... seq (h d) -> ... h seq d', h=self.num_heads)
        v = einops.rearrange(v, '... seq (h d) -> ... h seq d', h=self.num_heads)

        pos_extended = token_positions.unsqueeze(-2)
        q = self.rope(q,pos_extended)
        k = self.rope(k,pos_extended)

        mask = self.mask[:seq_len,:seq_len]
        out = scaled_dot_product_attention(q,k,v,mask=mask)
        out = einops.rearrange(out,'... h seq d ->... seq (h d)',h = self.num_heads)

        return self.w_o(out)


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, device=None,
                 dtype=None):
        super().__init__()
        self.d_model = d_model

        self.norm1 = RMSNorm(d_model,device = device,dtype = dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model,num_heads, max_seq_len, device = device,dtype = dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.fnn = SwiGLU(d_ff, d_model, device, dtype)

    def forward(self,x,token_positions):

        x = x + self.attn(self.norm1(x),token_positions)

        x = x + self.fnn(self.norm2(x))

        return x


class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.embedding = Embedding(vocab_size,d_model,device,dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,max_seq_len = context_length,device=device,dtype = dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model,vocab_size,device,dtype)

    def forward(self,token_ids):
        seq_len = token_ids.size(1)
        token_positions = torch.arange(seq_len,device=token_ids.device).unsqueeze(0).expand_as(token_ids)

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x,token_positions)
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

























