import json
import random
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams


# 导入你的底层工具函数
from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    log_generations
)
from drgrpo_grader import r1_zero_reward_fn


# 导入 vLLM 同步工具 (同 SFT)
# from utils import init_vllm, load_policy_into_vllm_instance

def main():
    # ================= 超参数设定 ================= [cite: 821-845]
    n_grpo_steps = 200
    learning_rate = 1e-5
    advantage_eps = 1e-6
    rollout_batch_size = 256
    group_size = 8
    sampling_temperature = 1.0
    sampling_min_tokens = 4
    sampling_max_tokens = 1024

    epochs_per_rollout_batch = 1  # On-policy 为 1; Off-policy 可设为更大
    train_batch_size = 256
    gradient_accumulation_steps = 128

    loss_type = "reinforce_with_baseline"  # 可选: "no_baseline", "reinforce_with_baseline", "grpo_clip"
    use_std_normalization = True
    cliprange = 0.2

    # 校验超参数逻辑 [cite: 847-859]
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size

    wandb.init(project="cs336-alignment-grpo", config=locals())
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # ================= 初始化模型 =================
    device_policy = "cuda:0"
    device_vllm = "cuda:1"
    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device_policy)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
    )

    vllm_engine = init_vllm(model_path, device_vllm, seed=42)

    # 加载数据集
    with open("/data/a5-alignment/MATH/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]
    with open("/data/a5-alignment/MATH/validation.jsonl", "r") as f:
        val_data = [json.loads(line) for line in f]

    with open("prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    sampling_params = SamplingParams(
        temperature=sampling_temperature, min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens, n=group_size,
        stop=["</answer>"], include_stop_str_in_output=True
    )

    global_train_step = 0
    eval_step = 0

    # ================= GRPO 主循环 ================= [cite: 819-820]
    for step in range(n_grpo_steps):
        print(f"\n--- GRPO Step {step + 1}/{n_grpo_steps} ---")

        # 1. 权重同步 & 采样
        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, vllm_engine)

        sampled_batch = random.sample(train_data, n_prompts_per_rollout_batch)
        prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in sampled_batch]
        gts = [ex["solution"] for ex in sampled_batch]

        # 2. 生成 Rollouts
        outputs = vllm_engine.generate(prompts, sampling_params)

        # 展开 responses 和 gts
        flat_prompts, flat_responses, flat_gts = [], [], []
        for i, output in enumerate(outputs):
            for j in range(group_size):
                flat_prompts.append(prompts[i])
                flat_responses.append(output.outputs[j].text)
                flat_gts.append(gts[i])

        # 3. 计算奖励和优势 [cite: 636-668]
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )

        # 4. 计算旧策略对数概率 (仅在 off-policy 的 GRPO-Clip 时需要)
        old_log_probs_list = []
        if loss_type == "grpo_clip":
            with torch.inference_mode():  # 冻结梯度
                # 为了防止 OOM，分 chunk 计算 old_log_probs
                for i in range(0, len(flat_prompts), micro_train_batch_size):
                    chunk_p = flat_prompts[i:i + micro_train_batch_size]
                    chunk_r = flat_responses[i:i + micro_train_batch_size]
                    chunk_toks = tokenize_prompt_and_output(chunk_p, chunk_r, tokenizer)

                    log_res = get_response_log_probs(
                        policy_model,
                        chunk_toks["input_ids"].to(device_policy),
                        chunk_toks["labels"].to(device_policy)
                    )
                    # 必须转移到 CPU 或其他地方存储，防止显存爆炸
                    old_log_probs_list.append(log_res["log_probs"].cpu())

        # 5. 模型更新阶段 (PPO/GRPO 风格的内层循环)
        policy_model.train()

        # 构建扁平化的经验池索引
        indices = list(range(len(flat_prompts)))

        for epoch in range(epochs_per_rollout_batch):
            random.shuffle(indices)  # 打乱样本

            # 按 train_batch_size 遍历
            for i in range(0, len(indices), train_batch_size):
                batch_indices = indices[i:i + train_batch_size]

                optimizer.zero_grad()
                batch_loss = 0.0
                batch_entropies = []
                batch_clip_fracs = []

                # 按微批次遍历 (Gradient Accumulation)
                for j in range(0, len(batch_indices), micro_train_batch_size):
                    micro_indices = batch_indices[j:j + micro_train_batch_size]

                    mb_prompts = [flat_prompts[idx] for idx in micro_indices]
                    mb_responses = [flat_responses[idx] for idx in micro_indices]
                    mb_advs = advantages[micro_indices].to(device_policy)
                    mb_raw_rewards = raw_rewards[micro_indices].to(device_policy)

                    toks = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                    input_ids = toks["input_ids"].to(device_policy)
                    labels = toks["labels"].to(device_policy)
                    response_mask = toks["response_mask"].to(device_policy)

                    # 当前策略的 log probs
                    log_res = get_response_log_probs(
                        policy_model, input_ids, labels, return_token_entropy=True
                    )

                    # 提取对应的旧策略 log probs (如果需要)
                    mb_old_log_probs = None
                    if loss_type == "grpo_clip":
                        # 根据原始索引重组旧的 log probs，注意需要移回 GPU 并对齐形状
                        # (此处实现略有复杂，因为 batch padding 可能因打乱而不同，
                        # 工业界通常将 log_probs 与序列打包在一起打乱，这里为简明做示意)
                        pass

                        # 计算 Loss 和梯度
                    scaled_loss, loss_meta = grpo_microbatch_train_step(
                        policy_log_probs=log_res["log_probs"],
                        response_mask=response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=mb_raw_rewards,
                        advantages=mb_advs,
                        old_log_probs=mb_old_log_probs,
                        cliprange=cliprange
                    )

                    batch_loss += loss_meta["loss"].item()

                    # 记录熵和 clip fraction
                    entropy = log_res["token_entropy"]
                    valid_entropy = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1)
                    batch_entropies.append(valid_entropy.item())
                    if "clip_fraction" in loss_meta:
                        batch_clip_fracs.append(loss_meta["clip_fraction"].item())

                # 梯度裁剪和优化器步进 [cite: 864]
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                global_train_step += 1

                # 日志记录 [cite: 873-879]
                wandb.log({
                    "train_step": global_train_step,
                    "train/loss": batch_loss / gradient_accumulation_steps,
                    "train/grad_norm": grad_norm.item(),
                    "train/token_entropy": sum(batch_entropies) / len(batch_entropies),
                    "train/reward_mean": reward_meta["reward_mean"],
                    "train/clip_fraction": sum(batch_clip_fracs) / len(batch_clip_fracs) if batch_clip_fracs else 0.0,
                })

        # ================= 验证阶段 ================= [cite: 865-866]
        if (step + 1) % 5 == 0:
            policy_model.eval()
            load_policy_into_vllm_instance(policy_model, vllm_engine)

            # 使用至少 1024 个验证样本
            val_prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in val_data[:1024]]
            val_gts = [ex["solution"] for ex in val_data[:1024]]

            val_outputs = vllm_engine.generate(val_prompts, sampling_params)
            val_responses = [out.outputs[0].text for out in val_outputs]

            metrics = log_generations(val_prompts, val_responses, val_gts, r1_zero_reward_fn, step=step)
            metrics["eval_step"] = eval_step
            wandb.log(metrics)
            eval_step += 1


if __name__ == "__main__":
    main()