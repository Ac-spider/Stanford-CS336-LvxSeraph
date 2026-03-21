import torch
from transformers import PreTrainedTokenizerBase
from typing import Literal
from typing import Callable, List


def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    for p_str, o_str in zip(prompt_strs, output_strs):
        # 1. 独立分词
        # 问题部分通常保留特殊的起始 token (如 BOS)
        p_ids = tokenizer.encode(p_str, add_special_tokens=True)
        # 回答部分追加在问题之后，不应该添加开头的特殊 token
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)

        # 2. 拼接总序列
        concat_ids = p_ids + o_ids

        # 3. 切片构建 input_ids 和 shifted labels
        # 截掉最后一个 token 作为输入
        input_ids = concat_ids[:-1]
        # 截掉第一个 token 作为标签 (即预测下一个 token)
        labels = concat_ids[1:]

        # 4. 构建 response_mask
        # 模型前 len(p_ids) - 1 个目标是在预测 prompt 本身的 token，这些不需要计算 loss (设为 0)
        # 紧接着的 len(o_ids) 个目标是预测 response 的 token，需要计算 loss (设为 1)
        mask = [0] * (len(p_ids) - 1) + [1] * len(o_ids)

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_response_mask.append(mask)

    # 5. Padding (右侧填充对齐 Batch 内最大长度)
    max_len = max(len(ids) for ids in batch_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_mask = []

    for i_ids, l_ids, m in zip(batch_input_ids, batch_labels, batch_response_mask):
        pad_len = max_len - len(i_ids)

        padded_input_ids.append(i_ids + [pad_id] * pad_len)
        padded_labels.append(l_ids + [pad_id] * pad_len)
        padded_mask.append(m + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_mask, dtype=torch.long)
    }


import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
                containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length).
        The entropy for each next-token prediction.
    """
    # 为了数值稳定，找到最后一个维度（词表维度）的最大值
    # keepdim=True 保证形状为 (batch_size, sequence_length, 1)，以便广播
    max_logits = torch.max(logits, dim=-1, keepdim=True).values

    # 减去最大值后计算 exp
    exp_logits = torch.exp(logits - max_logits)

    # 计算 sum(exp) 和 log(sum(exp))
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)

    # 手动计算 log_softmax = x - max_x - log(sum(exp(x - max_x)))
    log_probs = logits - max_logits - log_sum_exp

    # 概率 p = exp(log_probs)
    probs = torch.exp(log_probs)

    # 计算熵：-sum(p * log_p)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy


def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities from a causal language model,
    and optionally the entropy of the model's next-token distribution.
    """
    # 1. 运行前向传播获取 logits
    outputs = model(input_ids)
    logits = outputs.logits  # (batch_size, sequence_length, vocab_size)

    # 2. 手动计算所有 token 的 log_softmax (同样使用 LogSumExp 保证数值稳定)
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)

    log_probs_all = logits - max_logits - torch.log(sum_exp)

    # 3. 使用 gather 提取 labels 对应位置的 log 概率
    # labels 形状为 (batch_size, sequence_length)
    # 提取后 squeeze 掉最后一个维度，恢复为 (batch_size, sequence_length)
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {
        "log_probs": log_probs
    }

    # 4. 可选：复用我们刚刚写的函数来计算熵 [cite: 297, 298]
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
    """
    # 1. 屏蔽非目标元素：将 tensor 与 mask 相乘
    # 确保 mask 和 tensor 的数据类型一致，避免报错
    masked_tensor = tensor * mask.to(tensor.dtype)

    # 2. 沿指定维度求和
    if dim is None:
        # 如果 dim 为 None，则对整个张量求和
        summed = torch.sum(masked_tensor)
    else:
        # 否则沿指定维度求和
        summed = torch.sum(masked_tensor, dim=dim)

    # 3. 归一化
    normalized = summed / normalize_constant

    return normalized


def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    """
    # 1. 计算被 mask 的 response token 的对数概率之和 (并归一化)
    # SFT 的目标是最大化这些 token 的 log-prob
    sum_log_probs = masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )

    # 2. 将最大化问题转为最小化问题，加上负号得到 NLL Loss
    loss = -sum_log_probs

    # 3. 为梯度累积调整 loss 比例
    scaled_loss = loss / gradient_accumulation_steps

    # 4. 反向传播 (此时 PyTorch 会为模型参数累加梯度)
    scaled_loss.backward()

    # 5. 组装 metadata 用于后续的日志记录
    metadata = {
        "sft_loss": loss.detach(),
        "scaled_sft_loss": scaled_loss.detach()
    }

    return scaled_loss, metadata


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

    # 如果没有传入 entropies，用 0.0 占位
    if token_entropies is None:
        token_entropies = [0.0] * len(prompts)

    for i, (p, r, gt, entropy) in enumerate(zip(prompts, responses, ground_truths, token_entropies)):
        # 1. 计算 Reward
        reward_info = reward_fn(r, gt)
        t_reward = reward_info.get("reward", 0.0)
        f_reward = reward_info.get("format_reward", 0.0)
        a_reward = reward_info.get("answer_reward", 0.0)

        total_rewards.append(t_reward)
        format_rewards.append(f_reward)
        answer_rewards.append(a_reward)

        # 2. 统计长度
        length = len(r)
        response_lengths.append(length)
        if a_reward > 0:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

        # 可以选择只打印前几个示例进行肉眼 sanity check
        if i < 2:
            print(f"--- Step {step} | Sample {i + 1} ---")
            print(f"Prompt: {p[:100]}...")
            print(f"Response: {r}")
            print(f"Ground Truth: {gt}")
            print(f"Rewards: Total={t_reward}, Format={f_reward}, Answer={a_reward}")
            print(f"Avg Entropy: {entropy:.4f}")
            print("-" * 30)

    # 3. 计算聚合统计量
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    metrics = {
        "eval/reward_total": safe_mean(total_rewards),
        "eval/reward_format": safe_mean(format_rewards),
        "eval/reward_answer": safe_mean(answer_rewards),
        "eval/avg_token_entropy": safe_mean(token_entropies),
        "eval/avg_response_len": safe_mean(response_lengths),
        "eval/avg_correct_len": safe_mean(correct_lengths),
        "eval/avg_incorrect_len": safe_mean(incorrect_lengths),
    }

    return metrics


def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    计算组内归一化后的奖励 (Advantages)。
    """
    raw_rewards_list = []

    # 1. 计算每个 rollout 的原始奖励
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(resp, gt)
        raw_rewards_list.append(reward_dict.get("reward", 0.0))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)  # shape: (rollout_batch_size,)

    # 2. 变形为组 (batch_size_per_prompt, group_size)
    # rollout_batch_size 必定能被 group_size 整除
    reshaped_rewards = raw_rewards.view(-1, group_size)

    # 3. 计算均值
    group_means = reshaped_rewards.mean(dim=1, keepdim=True)

    # 4. 计算优势 (Advantage)
    if normalize_by_std:
        # std 需要加上 eps 避免除以 0
        # unbiased=False 保持与标准偏差公式一致
        group_stds = reshaped_rewards.std(dim=1, unbiased=False, keepdim=True)
        advantages_reshaped = (reshaped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        advantages_reshaped = reshaped_rewards - group_means

    # 变回一维张量
    advantages = advantages_reshaped.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算未截断的朴素策略梯度损失。
    """
    # raw_rewards_or_advantages 的形状是 (batch_size,)
    # 为了与 policy_log_probs (batch_size, sequence_length) 相乘，我们需要增加一个维度
    adv_expanded = raw_rewards_or_advantages.unsqueeze(-1)

    # 逐 token 损失
    per_token_loss = -adv_expanded * policy_log_probs

    return per_token_loss


def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GRPO-Clip 逐 token 损失。
    """
    # 扩展 advantages 以匹配序列维度
    adv_expanded = advantages.unsqueeze(-1)  # (batch_size, 1)

    # 1. 计算重要性采样比率: ratio = exp(log_p_new - log_p_old)
    # 巧妙利用指数运算法则避免除法带来的不稳定性
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # 2. 计算未截断的项 (Unclipped)
    unclipped_term = ratio * adv_expanded

    # 3. 计算截断的项 (Clipped)
    # 将 ratio 截断在 [1 - cliprange, 1 + cliprange] 之间
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_term = clipped_ratio * adv_expanded

    # 4. 取两者的最小值 (针对最大化目标)，然后取负号转为最小化 Loss
    per_token_loss = -torch.min(unclipped_term, clipped_term)

    # 5. 记录被截断的 token 比例 (用于监控训练稳定性)
    # 如果截断项小于未截断项，说明发生了截断
    is_clipped = (clipped_term < unclipped_term).to(torch.float32)
    clip_fraction = is_clipped.mean()

    metadata = {
        "clip_fraction": clip_fraction.detach()
    }

    return per_token_loss, metadata


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
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"
        loss, clip_meta = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata.update(clip_meta)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:
    """
    计算张量在给定维度上的均值，仅考虑 mask == 1 的元素。
    """
    # 确保 mask 和 tensor 的数据类型一致
    mask = mask.to(tensor.dtype)

    # 将被 mask 掉的元素置为 0
    masked_tensor = tensor * mask

    # 根据 dim 参数进行求和
    if dim is None:
        sum_tensor = torch.sum(masked_tensor)
        count_tensor = torch.sum(mask)
    else:
        sum_tensor = torch.sum(masked_tensor, dim=dim)
        count_tensor = torch.sum(mask, dim=dim)

    # 为防止除以 0，使用 clamp 设定一个极小的下界
    count_tensor = torch.clamp(count_tensor, min=1e-8)

    # 计算均值
    mean_tensor = sum_tensor / count_tensor

    return mean_tensor


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
    """
    执行单次 GRPO 微批次的前向与反向传播。
    """
    # 1. 计算逐 token 的策略梯度损失
    # 返回的 per_token_loss 形状: (batch_size, sequence_length)
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 2. 沿着序列维度 (dim=1) 计算掩码均值，得到每个样本的标量损失
    # 形状变为: (batch_size,)
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    # 3. 在 batch 维度上计算平均损失
    # 形状变为标量
    loss = torch.mean(per_example_loss)

    # 4. 根据梯度累积步数缩放损失
    scaled_loss = loss / gradient_accumulation_steps

    # 5. 反向传播计算梯度
    scaled_loss.backward()

    # 6. 将 loss 记录到 metadata 中以便外部 wandb 记录
    metadata["loss"] = loss.detach()
    metadata["scaled_loss"] = scaled_loss.detach()

    return scaled_loss, metadata







