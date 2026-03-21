import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

# ─────────────────────────────────────────
# run_compute_per_instance_dpo_loss
# Purpose: 对单条 (prompt, chosen, rejected) 样本计算 DPO 损失
# Key concept:
#   DPO 损失公式：
#     L = -log σ( β * ( log π_θ(y_w|x)/π_ref(y_w|x)
#                      - log π_θ(y_l|x)/π_ref(y_l|x) ) )
#   其中 y_w 为 chosen（被偏好）回复，y_l 为 rejected（被拒绝）回复。
#   通过比较"策略相对参考模型的对数比率差"而非直接比较绝对概率，
#   可以天然地规避奖励模型的显式训练，同时惩罚策略偏离参考模型过多。
# ─────────────────────────────────────────
def run_compute_per_instance_dpo_loss(
        lm: torch.nn.Module,
        lm_ref: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        beta: float,
        prompt: str,
        response_chosen: str,
        response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.
    """
    # 1. 使用 Alpaca 指令模板格式化 prompt + response；末尾必须加 EOS token，
    #    以便模型学习何时停止生成
    template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{prompt}\n\n### Response:\n{response}"
    )

    # 将 chosen / rejected 分别填入模板，得到完整文本
    chosen_text = template.format(prompt=prompt, response=response_chosen) + tokenizer.eos_token
    rejected_text = template.format(prompt=prompt, response=response_rejected) + tokenizer.eos_token

    # 2. 编码为 token id（add_special_tokens=False 避免重复插入 BOS）
    chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False, return_tensors="pt")
    rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False, return_tensors="pt")

    # 3. 获取两个模型各自所在的设备（策略模型与参考模型可能不在同一张卡上）
    device_lm = next(lm.parameters()).device
    device_ref = next(lm_ref.parameters()).device

    # ─────────────────────────────────────────
    # get_sequence_log_prob（内部辅助函数）
    # Purpose: 计算给定模型在整段序列上的 log P(sequence) = Σ log P(token_t | token_{<t})
    # Key concept:
    #   - shift logits/labels：位置 i 的 logit 预测位置 i+1 的 token，
    #     因此 logits[:-1] 与 labels[1:] 对齐
    #   - log_softmax + gather：数值稳定地取出真实 token 的对数概率
    # ─────────────────────────────────────────
    def get_sequence_log_prob(model: torch.nn.Module, input_ids: torch.Tensor, device: torch.device):
        input_ids = input_ids.to(device)

        # 参考模型只用于提供 baseline，无需梯度；策略模型需要梯度回传
        with torch.no_grad() if model is lm_ref else torch.enable_grad():
            logits = model(input_ids).logits  # shape: (1, seq_len, vocab_size)

            # ── Shift 对齐：预测"下一个 token" ──
            # logits[0, :-1, :] 对应位置 0..T-2 的预测
            # input_ids[0, 1:]  对应位置 1..T-1 的真实 token
            shift_logits = logits[0, :-1, :]       # (seq_len-1, vocab_size)
            shift_labels = input_ids[0, 1:]        # (seq_len-1,)

            # log_softmax 在 vocab 维度上归一化，数值稳定
            log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (seq_len-1, vocab_size)

            # gather：只取真实 token 对应的 log prob，避免对整个 vocab 求和
            token_log_probs = torch.gather(
                log_probs_all, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # (seq_len-1,)

            # 对所有 token 的 log prob 求和 → log P(sequence | model)
            return token_log_probs.sum()

    # 4. 策略模型 (π_θ) 对 chosen 和 rejected 序列的对数概率
    log_pi_theta_chosen   = get_sequence_log_prob(lm,     torch.tensor(chosen_ids),   device_lm)
    log_pi_theta_rejected = get_sequence_log_prob(lm,     torch.tensor(rejected_ids), device_lm)

    # 5. 参考模型 (π_ref) 的对数概率，计算后移至策略模型所在设备以便后续运算
    log_pi_ref_chosen   = get_sequence_log_prob(lm_ref, torch.tensor(chosen_ids),   device_ref).to(device_lm)
    log_pi_ref_rejected = get_sequence_log_prob(lm_ref, torch.tensor(rejected_ids), device_ref).to(device_lm)

    # 6. 计算"隐式奖励"差值
    #    DPO 将偏好建模等价于：
    #      r(x,y) ∝ log π_θ(y|x) - log π_ref(y|x)
    #    用比率差（log ratio difference）可以消去配分函数，
    #    而不必显式训练奖励模型
    pi_logratios  = log_pi_theta_chosen - log_pi_theta_rejected    # Δ log π_θ
    ref_logratios = log_pi_ref_chosen   - log_pi_ref_rejected      # Δ log π_ref
    logits_diff   = pi_logratios - ref_logratios                   # 隐式奖励之差

    # 7. DPO 最终损失：-log σ(β * diff)
    #    β 超参数控制偏离参考模型的惩罚强度，越大越保守
    loss = -F.logsigmoid(beta * logits_diff)

    return loss
