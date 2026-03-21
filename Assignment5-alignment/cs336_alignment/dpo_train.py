import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


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
    # 1. 使用 Alpaca 模板格式化，并务必在末尾添加 EOS token [cite: 1588]
    template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{prompt}\n\n### Response:\n{response}"
    )

    chosen_text = template.format(prompt=prompt, response=response_chosen) + tokenizer.eos_token
    rejected_text = template.format(prompt=prompt, response=response_rejected) + tokenizer.eos_token

    # 2. 编码文本 (不自动添加特殊 token 以保持纯粹)
    chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False, return_tensors="pt")
    rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False, return_tensors="pt")

    # 3. 获取模型所在的设备 [cite: 1584-1585]
    device_lm = next(lm.parameters()).device
    device_ref = next(lm_ref.parameters()).device

    # 辅助函数：计算一整段序列的 log_prob 之和
    def get_sequence_log_prob(model: torch.nn.Module, input_ids: torch.Tensor, device: torch.device):
        input_ids = input_ids.to(device)

        # 为了节省显存，在 reference model 前向传播时不计算梯度 [cite: 1584]
        with torch.no_grad() if model is lm_ref else torch.enable_grad():
            logits = model(input_ids).logits  # (1, seq_length, vocab_size)

            # 模型预测下一个 Token：错开一位
            shift_logits = logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]

            # 使用 log_softmax 计算所有词的对数概率
            log_probs_all = F.log_softmax(shift_logits, dim=-1)

            # 提取真实 Token 对应的 log_prob
            token_log_probs = torch.gather(
                log_probs_all, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # 全部求和得到整段序列的 log prob
            return token_log_probs.sum()

    # 4. 计算当前策略网络 (theta) 对选中和拒绝序列的概率
    log_pi_theta_chosen = get_sequence_log_prob(lm, torch.tensor(chosen_ids), device_lm)
    log_pi_theta_rejected = get_sequence_log_prob(lm, torch.tensor(rejected_ids), device_lm)

    # 5. 计算参考网络 (ref) 对选中和拒绝序列的概率，并将结果移至当前策略网络的设备上 [cite: 1585]
    log_pi_ref_chosen = get_sequence_log_prob(lm_ref, torch.tensor(chosen_ids), device_ref).to(device_lm)
    log_pi_ref_rejected = get_sequence_log_prob(lm_ref, torch.tensor(rejected_ids), device_ref).to(device_lm)

    # 6. 计算隐式奖励的对数比率差
    # (log_pi_theta_chosen - log_pi_ref_chosen) - (log_pi_theta_rejected - log_pi_ref_rejected)
    # 通过数学等价转换，等同于:
    pi_logratios = log_pi_theta_chosen - log_pi_theta_rejected
    ref_logratios = log_pi_ref_chosen - log_pi_ref_rejected
    logits_diff = pi_logratios - ref_logratios

    # 7. 计算最终的 DPO 损失： -log(sigmoid(beta * diff)) [cite: 1551]
    loss = -F.logsigmoid(beta * logits_diff)

    return loss