
import torch
from torch.masked import masked_tensor
from transformers import PreTrainedTokenizerBase
from typing import Callable, List,Literal


# ─────────────────────────────────────────
# tokenize_prompt_and_output
# Purpose: 将一批 (prompt, output) 字符串对拼接、分词、对齐，
#          并生成 response_mask——仅标记 output 部分为 1，
#          使 SFT 只对 response token 计算 loss，不惩罚 prompt 部分。
# Key concept:
#   拼接后做 shift：input_ids = concat[:-1]，labels = concat[1:]，
#   这样位置 i 的 input 预测位置 i+1 的 label（语言模型标准做法）。
#   response_mask 的起始位置为 len(p_ids) - 1（因为 shift 后 prompt
#   最后一个 token 的预测目标才是 output 第一个 token）。
# ─────────────────────────────────────────
def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:

    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    for p_str,o_str in zip(prompt_strs,output_strs):
        # prompt 添加特殊 token（如 BOS），output 不重复添加
        p_ids = tokenizer.encode(p_str,add_special_tokens=True)
        o_ids = tokenizer.encode(o_str,add_special_tokens=False)

        # 拼接完整序列
        concat_ids = p_ids + o_ids

        # response_mask：
        #   - prompt 部分（shift 后共 len(p_ids)-1 个位置）标记为 0
        #   - output 部分（len(o_ids) 个位置）标记为 1
        #   SFT 仅对 mask=1 的 token 计算交叉熵损失
        mask = [0] * (len(p_ids) - 1) + [1] * len(o_ids)
        # shift：input[i] 预测 label[i+1]
        input_ids = concat_ids[:-1]
        labels    = concat_ids[1:]

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_response_mask.append(mask)

    # 批次内按最长序列 padding（右 padding）
    max_len = max(len(ids) for ids in batch_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_mask = []

    for i_ids, l_ids, m in zip(batch_input_ids, batch_labels, batch_response_mask):
        pad_len = max_len - len(i_ids)
        # padding 位置的 mask 设为 0，不参与 loss 计算
        padded_input_ids.append(i_ids + [pad_id] * pad_len)
        padded_labels.append(l_ids    + [pad_id] * pad_len)
        padded_mask.append(m          + [0]      * pad_len)

    return{
    "input_ids":      torch.tensor(padded_input_ids, dtype=torch.long),
    "labels":         torch.tensor(padded_labels,    dtype=torch.long),
    "response_mask":  torch.tensor(padded_mask,      dtype=torch.long)
    }


# ─────────────────────────────────────────
# compute_entropy
# Purpose: 计算 logits 对应的信息熵 H(p) = -Σ p(x) log p(x)
# Key concept:
#   使用 log-sum-exp 技巧保证数值稳定性：
#     log Σ exp(z_i) = max(z) + log Σ exp(z_i - max(z))
#   从而避免 exp(large_number) 溢出或 exp(-large_number) 下溢。
# ─────────────────────────────────────────
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:

    # 减去每个位置的最大值（log-sum-exp trick 的第一步）
    max_logits = torch.max(logits,dim=-1,keepdim=True)[0]

    # exp(z - max(z))，数值范围被限制在 (0, 1]
    exp_logits = torch.exp(logits-max_logits)

    # Σ exp(z_i - max(z))
    sum_exp = torch.sum(exp_logits,dim=-1,keepdim=True)
    # log Σ exp(z_i - max(z))
    log_sum_exp = torch.log(sum_exp)

    # log p(x) = z_x - max(z) - log Σ exp(z_i - max(z))
    log_probs = logits - max_logits - log_sum_exp

    # p(x) = exp(log p(x))
    probs = torch.exp(log_probs)

    # H = -Σ p(x) * log p(x)，在 vocab 维度上求和
    entropy = -torch.sum(probs * log_probs,dim=-1)

    return entropy


# ─────────────────────────────────────────
# get_response_log_probs
# Purpose: 模型前向传播，提取每个位置真实 token 的 log_prob；
#          可选返回每个位置的 token 级别熵（用于监控训练过程中的多样性）。
# Key concept:
#   与 dpo_train.py 中 get_sequence_log_prob 类似，
#   但这里不做 shift（shift 已在 tokenize_prompt_and_output 中处理），
#   直接用 labels 作为 gather 的索引。
#   同样用 log-sum-exp trick 保证数值稳定。
# ─────────────────────────────────────────
def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    # 前向传播，获取所有位置的 logit：(batch, seq_len, vocab_size)
    logits = model(input_ids).logits

    # log-sum-exp 数值稳定化
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp    = torch.sum(exp_logits, dim=-1, keepdim=True)

    # log P(token | context) = logit - max - log Σ exp(logit - max)
    log_probs_all = logits - max_logits - torch.log(sum_exp)  # (batch, seq_len, vocab_size)

    # gather：只取 labels 对应的真实 token 的 log prob
    # labels shape: (batch, seq_len) → unsqueeze → (batch, seq_len, 1)
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)  # → (batch, seq_len)

    result = {
        "log_probs": log_probs
    }

    # 可选：返回每个 token 位置的预测熵，用于追踪策略的"不确定性"
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


# ─────────────────────────────────────────
# masked_normalize
# Purpose: 将 tensor 中 mask=1 的元素求和后除以归一化常数，
#          用于 SFT loss 计算时屏蔽 prompt 及 padding 位置。
# ─────────────────────────────────────────
def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:

    # 将 mask 广播乘到 tensor，屏蔽不需要的位置
    masked_tensor = tensor * mask.to(tensor.dtype)

    if dim is None:
        # 全局求和（跨 batch 和 seq_len）
        summed = torch.sum(masked_tensor)
    else:
        # 沿指定维度求和
        summed = torch.sum(masked_tensor,dim=dim)

    # 除以归一化常数（例如有效 token 数量或样本数）
    normalized = summed / normalize_constant

    return normalized


# ─────────────────────────────────────────
# sft_microbatch_train_step
# Purpose: SFT 单个微批次的训练步骤：
#          1. 对 response 位置的 log_prob 求和（masked sum）
#          2. 取负得到交叉熵损失
#          3. 除以 gradient_accumulation_steps 后 backward
# Key concept:
#   梯度累积（Gradient Accumulation）：
#     将一个大 batch 分成多个小批次（microbatch）依次前向/反向，
#     累积梯度后统一更新参数，等效于更大的 batch_size，但节省显存。
#     每次 backward 前将 loss 除以累积步数，保证梯度尺度不变。
# ─────────────────────────────────────────
def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # 对 response 部分的 log_prob 求 masked 加权和（忽略 prompt 和 padding）
    sum_log_probs = masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )

    # SFT 损失 = -log P(response | prompt)，即负对数似然
    loss = -sum_log_probs

    # 梯度累积：将 loss 等比缩小，累积多步后梯度等效于完整 batch
    scaled_loss = loss / gradient_accumulation_steps

    # 反向传播（累积梯度，不清零）
    scaled_loss.backward()

    metadata = {
        "sft_loss":loss.detach(),
        "scaled_sft_loss": scaled_loss.detach()
    }

    return scaled_loss, metadata


# ─────────────────────────────────────────
# log_generations
# Purpose: 评估阶段记录模型生成的样本，
#          计算 reward / 长度 / 熵等聚合指标并返回供 wandb 记录。
# ─────────────────────────────────────────
def log_generations(
        prompts: List[str],
        responses: List[str],
        ground_truths: List[str],
        reward_fn: Callable[[str, str], dict[str, float]],
        token_entropies: List[float] = None,
        step: int = 0
) -> dict:
    """
    Log generations from the model, including rewards, token entropy, and lengths.
    Returns a dictionary of aggregated metrics that you can log to wandb.
    """
    total_rewards = []
    format_rewards = []
    answer_rewards = []

    response_lengths = []
    correct_lengths = []
    incorrect_lengths = []

    # 若未提供熵，默认全 0
    if token_entropies is None:
        token_entropies = [0.0] * len(prompts)

    for i, (p, r, gt, entropy) in enumerate(zip(prompts, responses, ground_truths, token_entropies)):
        # 1. 计算 Reward（含 format_reward 和 answer_reward 分项）
        reward_info = reward_fn(r, gt)
        t_reward = reward_info.get("reward", 0.0)
        f_reward = reward_info.get("format_reward", 0.0)
        a_reward = reward_info.get("answer_reward", 0.0)

        total_rewards.append(t_reward)
        format_rewards.append(f_reward)
        answer_rewards.append(a_reward)

        # 2. 统计回复长度（字符级），并按正确/错误分组
        length = len(r)
        response_lengths.append(length)
        if a_reward > 0:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

        # 仅打印前 2 个样本，用于人工 sanity check
        if i < 2:
            print(f"--- Step {step} | Sample {i + 1} ---")
            print(f"Prompt: {p[:100]}...")
            print(f"Response: {r}")
            print(f"Ground Truth: {gt}")
            print(f"Rewards: Total={t_reward}, Format={f_reward}, Answer={a_reward}")
            print(f"Avg Entropy: {entropy:.4f}")
            print("-" * 30)

    # 3. 聚合统计：均值（空列表返回 0.0 避免除零）
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    metrics = {
        "eval/reward_total":       safe_mean(total_rewards),
        "eval/reward_format":      safe_mean(format_rewards),
        "eval/reward_answer":      safe_mean(answer_rewards),
        "eval/avg_token_entropy":  safe_mean(token_entropies),
        "eval/avg_response_len":   safe_mean(response_lengths),
        "eval/avg_correct_len":    safe_mean(correct_lengths),
        "eval/avg_incorrect_len":  safe_mean(incorrect_lengths),
    }

    return metrics


# ─────────────────────────────────────────
# compute_group_normalized_rewards
# Purpose: GRPO 的组内奖励归一化，将原始奖励转换为 advantage。
# Key concept:
#   对每个 prompt 采样 G 个（group_size）回复，
#   在组内计算奖励的均值（和可选的标准差）：
#     advantage_i = (r_i - mean(r_group)) / (std(r_group) + ε)
#   这相当于组内的相对优势估计，消除了不同 prompt 绝对难度差异的影响，
#   使策略梯度信号更稳定。
# ─────────────────────────────────────────
def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:

    raw_rewards_list = []

    # 逐条调用奖励函数，收集原始标量奖励
    for resp,gt in zip(rollout_responses,repeated_ground_truths):
        reward_dict = reward_fn(resp,gt)
        raw_rewards_list.append(reward_dict.get('reward',0))

    raw_rewards = torch.tensor(raw_rewards_list,dtype=torch.float32)

    # 将奖励重塑为 (n_prompts, group_size) 便于组内统计
    reshaped_rewards = raw_rewards.view(-1,group_size)

    # 组内均值：shape (n_prompts, 1)
    group_means = reshaped_rewards.mean(dim=-1,keepdim=True)

    if normalize_by_std:
        # 除以组内标准差（有偏估计，unbiased=False），避免方差很小时梯度爆炸
        # advantage_eps 作为分母下界防止除零
        group_stds = reshaped_rewards.std(dim=-1,unbiased=False,keepdim=True)
        advantage_reshaped = (reshaped_rewards-group_means)/group_stds
    else:
        # 仅减均值，不除标准差（reinforce_with_baseline 的基线减法）
        advantage_reshaped = reshaped_rewards - group_means

    # 展平回 (rollout_batch_size,)
    advantages = advantage_reshaped.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_max":  raw_rewards.max().item(),
        "reward_min":  raw_rewards.min().item(),
    }

    return advantages,raw_rewards,metadata


# ─────────────────────────────────────────
# compute_naive_policy_gradient_loss
# Purpose: REINFORCE 策略梯度损失（逐 token）
# Key concept:
#   REINFORCE 目标（最大化期望奖励）：
#     J(θ) = E[R * log π_θ(a|s)]
#   对应的损失（取负号转为最小化）：
#     L = -R * log π_θ(a|s)  （per-token）
#   其中 R 可以是原始奖励（no_baseline）或组归一化 advantage（reinforce_with_baseline）。
#   advantage 通过减均值/基线来降低梯度方差。
# ─────────────────────────────────────────
def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:

    # advantage 扩展到 seq_len 维度：(batch,) → (batch, 1) 广播到 (batch, seq_len)
    adv_expanded = raw_rewards_or_advantages.unsqueeze(-1)
    # 每个 token 的损失 = -advantage * log_prob
    per_token_loss = -adv_expanded*policy_log_probs

    return per_token_loss


# ─────────────────────────────────────────
# compute_grpo_clip_loss
# Purpose: PPO Clip 目标（GRPO 使用的裁剪策略梯度）
# Key concept:
#   PPO Clip 目标（最大化，取负号转最小化）：
#     L_clip = -E[ min( ratio * A,  clip(ratio, 1-ε, 1+ε) * A ) ]
#   其中 ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log_π_new - log_π_old)
#   裁剪机制限制了新旧策略的偏离程度，提升训练稳定性。
#   clip_fraction 监控有多少 token 触发了裁剪。
# ─────────────────────────────────────────
def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # advantage 扩展到 seq_len 维度
    adv_expanded = advantages.unsqueeze(-1)

    # 计算重要性采样比率 ratio = π_new / π_old = exp(log_new - log_old)
    ratio = torch.exp(policy_log_probs-old_log_probs)

    # 未裁剪项：ratio * advantage
    unclipped_term = ratio * adv_expanded

    # 裁剪项：将 ratio 限制在 [1-ε, 1+ε] 后再乘 advantage
    clipped_term = torch.clamp(ratio,1-cliprange,1+cliprange)*adv_expanded

    # 取两者中较小值（保守更新），然后取负号转为最小化
    per_token_loss = -torch.min(unclipped_term,clipped_term)

    # 统计触发裁剪的 token 比例（用于监控训练稳定性）
    is_clipped    = (clipped_term < unclipped_term).to(torch.float32)
    clip_fraction = is_clipped.mean()

    metadata = {
        "clip_fraction":clip_fraction.detach(),
    }

    return per_token_loss,metadata


# ─────────────────────────────────────────
# compute_policy_gradient_loss
# Purpose: 损失函数分发器（dispatcher），
#          根据 loss_type 调用对应的损失计算函数。
# Key concept:
#   三种模式：
#   - no_baseline：纯 REINFORCE，用原始奖励作为权重
#   - reinforce_with_baseline：REINFORCE + 基线减法（组内减均值），降低方差
#   - grpo_clip：PPO Clip，额外需要旧策略的 log_probs 和 cliprange
# ─────────────────────────────────────────
def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    便捷 Wrapper：分发到具体的损失函数。
    """
    metadata = {}

    if loss_type == "no_baseline":
        # 纯 REINFORCE：直接用原始奖励加权 log_prob
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        # REINFORCE + 基线：用组归一化后的 advantage 替换原始奖励
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        # PPO Clip：需要旧策略 log_probs 和 cliprange
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"
        loss, clip_meta = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata.update(clip_meta)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata


# ─────────────────────────────────────────
# masked_mean
# Purpose: 对 mask=1 的元素计算加权均值，
#          屏蔽 padding 和 prompt 位置，保证 loss 的规模不受序列长度影响。
# ─────────────────────────────────────────
def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:

    # 将整数 mask 转为与 tensor 相同的浮点类型
    mask = mask.to(tensor.dtype)

    # 屏蔽无效位置
    masked_tensor = tensor * mask

    if dim is None:
        # 全局均值（跨 batch 和 seq_len）
        sum_tensor   = torch.sum(masked_tensor)
        count_tensor = torch.sum(mask)
    else:
        # 沿指定维度均值（如 dim=1 对每条样本的 token 求均值）
        sum_tensor   = torch.sum(masked_tensor,dim=dim)
        count_tensor = torch.sum(mask)

    # clamp 防止分母为零（全 padding 的极端情况）
    count_tensor = torch.clamp(count_tensor,min=1e-8)

    mean_tensor = sum_tensor / count_tensor

    return mean_tensor


# ─────────────────────────────────────────
# grpo_microbatch_train_step
# Purpose: GRPO 单个微批次训练步骤：
#          1. 调用 compute_policy_gradient_loss 得到逐 token 损失
#          2. masked_mean 对每条样本的有效 token 求均值
#          3. 对 batch 内所有样本再求均值得到标量 loss
#          4. 除以 gradient_accumulation_steps 后 backward
# Key concept:
#   与 sft_microbatch_train_step 的区别：
#   - SFT 用 masked_normalize（求和），GRPO 用 masked_mean（求均值），
#     使 loss 尺度对序列长度不敏感。
#   - GRPO 的 loss 还依赖 advantage，需在外部提前计算好并传入。
# ─────────────────────────────────────────
def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # 计算逐 token 的策略梯度损失，shape: (batch, seq_len)
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 对每条样本在 seq_len 维度上做 masked mean（dim=1），得到 (batch,) 的 per-sample loss
    per_example_loss = masked_mean(per_token_loss,response_mask,dim=1)

    # 对 batch 内所有样本取均值，得到标量 loss
    loss = torch.mean(per_example_loss)

    # 梯度累积缩放，等效于更大 batch_size 的梯度更新
    scaled_loss = loss / gradient_accumulation_steps

    # 反向传播（不清零，累积多步后在外部 optimizer.step()）
    scaled_loss.backward()

    metadata['loss'] = loss.detach
    metadata['scaled_loss'] = scaled_loss.detach

    return scaled_loss,metadata
