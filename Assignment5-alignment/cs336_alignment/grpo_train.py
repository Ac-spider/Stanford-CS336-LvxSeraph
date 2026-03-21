import json
import random
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

# 导入 GRPO 所需的工具函数
from cs336_alignment.utils import (
    tokenize_prompt_and_output,       # prompt+output 分词，生成 response_mask
    get_response_log_probs,           # 前向传播，获取真实 token 的 log_prob
    compute_group_normalized_rewards, # 组内奖励归一化 → advantage
    grpo_microbatch_train_step,       # 单微批次策略梯度 loss + backward
    log_generations                   # 评估阶段记录生成样本指标
)
# 数学题评测函数：检查格式（<think>/<answer> 标签）+ 答案正确性
from drgrpo_grader import r1_zero_reward_fn


# ─────────────────────────────────────────
# main
# Purpose: GRPO（Group Relative Policy Optimization）主训练循环
# 整体流程（每个 GRPO step）：
#   1. Rollout：用 vLLM 对每个 prompt 并行采样 group_size 个回复
#   2. 奖励计算 + 组内归一化 → advantage（相对优势信号）
#   3. 可选：记录旧策略 log_probs（用于 grpo_clip 的重要性采样）
#   4. 训练多个 epoch（epochs_per_rollout_batch）：
#      - On-policy（=1）：每次 rollout 只训练一轮，数据新鲜度高
#      - Off-policy（>1）：复用同一批 rollout 数据多轮，减少采样开销
#   5. 每 5 步做一次验证集评估
# Key concept:
#   GRPO 核心思想：
#     对同一 prompt 采样 G 个回复，组内计算相对奖励 advantage，
#     替代传统 PPO 中需要独立 value network 的做法，
#     大幅减少参数量和计算开销，同时保持策略梯度的方差控制效果。
# ─────────────────────────────────────────
def main():
    # ── 超参数设定 ──
    n_grpo_steps = 200                # GRPO 总迭代步数
    learning_rate = 1e-5
    advantage_eps = 1e-6              # 组内标准差的下界，防止除零
    rollout_batch_size = 256          # 每次 rollout 的总样本数 = n_prompts × group_size
    group_size = 8                    # 每个 prompt 采样的回复数 G（组大小）
    sampling_temperature = 1.0        # 采样温度（=1.0 保持多样性）
    sampling_min_tokens = 4           # 生成回复的最小 token 数
    sampling_max_tokens = 1024        # 生成回复的最大 token 数

    # On-policy vs Off-policy：
    #   epochs_per_rollout_batch=1 → 纯 on-policy（每次 rollout 后只更新一次）
    #   epochs_per_rollout_batch>1 → off-policy（复用 rollout 数据多轮，
    #                                           但数据与当前策略的偏差（distribution shift）会增大）
    epochs_per_rollout_batch = 1
    train_batch_size = 256            # 每次 optimizer.step() 使用的总样本数
    gradient_accumulation_steps = 128 # 梯度累积步数（micro_batch_size = train_batch_size / grad_acc）

    # loss_type 三选一：
    #   "no_baseline"            - 纯 REINFORCE，直接用原始奖励
    #   "reinforce_with_baseline"- REINFORCE + 组内基线减法（减均值）
    #   "grpo_clip"              - PPO Clip 目标（需要旧策略 log_probs）
    loss_type = "reinforce_with_baseline"
    use_std_normalization = True      # 是否除以组内标准差（进一步降低梯度方差）
    cliprange = 0.2                   # PPO Clip 的 ε 参数（仅 grpo_clip 使用）

    # ── 参数合法性校验 ──
    assert train_batch_size % gradient_accumulation_steps == 0
    # micro_batch_size：每次前向传播实际处理的样本数
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    # rollout_batch 必须能被 group_size 整除（确保完整的组结构）
    assert rollout_batch_size % group_size == 0
    # 每次 rollout 实际采样的不同 prompt 数量
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    # 训练 batch 至少要包含一个完整的组
    assert train_batch_size >= group_size

    # 初始化 wandb，将所有本地变量（超参数）记录为 config
    wandb.init(project="cs336-alignment-grpo", config=locals())
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    # ── 模型初始化 ──
    device_policy = "cuda:0"   # 策略模型（训练）
    device_vllm   = "cuda:1"   # vLLM 推理引擎
    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # bfloat16 + flash_attention_2：平衡精度与显存/速度
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device_policy)

    # AdamW with betas=(0.9, 0.95)，weight_decay=0（RL 训练通常不需要 L2 正则）
    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
    )

    # 启动 vLLM 推理引擎（独立 GPU，热更新权重）
    vllm_engine = init_vllm(model_path, device_vllm, seed=42)

    # 加载训练集和验证集（JSONL 格式，每行一个 {"problem": ..., "solution": ...}）
    with open("/data/a5-alignment/MATH/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]
    with open("/data/a5-alignment/MATH/validation.jsonl", "r") as f:
        val_data = [json.loads(line) for line in f]

    # 加载 R1-Zero 风格的 prompt 模板（含 {question} 占位符）
    with open("prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    # vLLM 采样参数：
    #   n=group_size 表示每个 prompt 并行采样 G 个回复（GRPO 核心需求）
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        n=group_size,                      # 组采样：每个 prompt 生成 G 个候选
        stop=["</answer>"],                # 遇到 </answer> 停止生成
        include_stop_str_in_output=True    # 保留 stop token 以便奖励函数判断格式
    )

    global_train_step = 0  # optimizer.step() 次数
    eval_step = 0

    # ════════════════════════════════════════
    # GRPO 主循环
    # ════════════════════════════════════════
    for step in range(n_grpo_steps):
        print(f"\n--- GRPO Step {step + 1}/{n_grpo_steps} ---")

        # ── Phase 1：Rollout（数据收集）──
        # 切换到 eval 模式（关闭 dropout），热更新 vLLM 权重
        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, vllm_engine)

        # 随机采样本轮要用的 prompt（不重放完整数据集，随机采样保持多样性）
        sampled_batch = random.sample(train_data, n_prompts_per_rollout_batch)
        prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in sampled_batch]
        gts     = [ex["solution"] for ex in sampled_batch]

        # vLLM 批量生成：每个 prompt 生成 group_size 个回复
        outputs = vllm_engine.generate(prompts, sampling_params)

        # 展平：将 (n_prompts, group_size) 结构展开为长度 rollout_batch_size 的平铺列表
        # flat_prompts[i*G + j] 对应第 i 个 prompt 的第 j 个回复
        flat_prompts, flat_responses, flat_gts = [], [], []
        for i, output in enumerate(outputs):
            for j in range(group_size):
                flat_prompts.append(prompts[i])
                flat_responses.append(output.outputs[j].text)
                flat_gts.append(gts[i])

        # ── Phase 2：奖励计算 + 组内归一化 ──
        # compute_group_normalized_rewards 内部：
        #   1. 逐条调用 r1_zero_reward_fn 得到原始标量奖励
        #   2. 按 group_size 分组，计算组内 (reward - mean) / std → advantage
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )

        # ── Phase 3（可选）：记录旧策略 log_probs（grpo_clip 需要）──
        # 在 inference_mode 下前向传播，不计算梯度，节省显存
        old_log_probs_list = []
        if loss_type == "grpo_clip":
            with torch.inference_mode():  # 冻结梯度，仅用于记录 π_old
                for i in range(0, len(flat_prompts), micro_train_batch_size):
                    chunk_p = flat_prompts[i:i + micro_train_batch_size]
                    chunk_r = flat_responses[i:i + micro_train_batch_size]
                    chunk_toks = tokenize_prompt_and_output(chunk_p, chunk_r, tokenizer)

                    log_res = get_response_log_probs(
                        policy_model,
                        chunk_toks["input_ids"].to(device_policy),
                        chunk_toks["labels"].to(device_policy)
                    )
                    # 保存到 CPU 避免 GPU 显存占用堆积
                    old_log_probs_list.append(log_res["log_probs"].cpu())

        # ── Phase 4：训练（支持多 epoch 复用 rollout 数据）──
        policy_model.train()

        # 所有展平样本的索引
        indices = list(range(len(flat_prompts)))

        # epochs_per_rollout_batch：
        #   =1  → On-policy：每次 rollout 后只训练一轮（数据与当前策略严格对齐）
        #   >1  → Off-policy：复用同一批 rollout 数据多轮（节省采样开销，但引入分布偏移）
        for epoch in range(epochs_per_rollout_batch):
            # 每个 epoch 内打乱顺序，避免相同组的样本总是相邻（影响 batch norm 等）
            random.shuffle(indices)

            for i in range(0, len(indices), train_batch_size):
                batch_indices = indices[i:i + train_batch_size]

                # 清零梯度（每个 train_batch 开始时）
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_entropies   = []
                batch_clip_fracs  = []

                # ── 梯度累积内循环（Gradient Accumulation）──
                # 将 train_batch 切分为多个 micro_batch，累积梯度后统一更新
                for j in range(0, len(batch_indices), micro_train_batch_size):
                    micro_indices = batch_indices[j:j + micro_train_batch_size]

                    # 取出当前微批次的 prompts / responses / advantages / raw_rewards
                    mb_prompts     = [flat_prompts[idx]   for idx in micro_indices]
                    mb_responses   = [flat_responses[idx] for idx in micro_indices]
                    mb_advs        = advantages[micro_indices].to(device_policy)
                    mb_raw_rewards = raw_rewards[micro_indices].to(device_policy)

                    # 分词
                    toks          = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                    input_ids     = toks["input_ids"].to(device_policy)
                    labels        = toks["labels"].to(device_policy)
                    response_mask = toks["response_mask"].to(device_policy)

                    # 前向传播：获取当前策略 log_probs 和 token 熵
                    log_res = get_response_log_probs(
                        policy_model, input_ids, labels, return_token_entropy=True
                    )

                    # grpo_clip 需要旧策略的 log_probs（从 Phase 3 缓存中取）
                    mb_old_log_probs = None
                    if loss_type == "grpo_clip":
                        pass  # 此处预留：从 old_log_probs_list 中按索引取对应片段

                    # 计算策略梯度损失并 backward（scaled_loss = loss / gradient_accumulation_steps）
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

                    # 累积本次微批次的 loss（用于日志）
                    batch_loss += loss_meta["loss"].item()

                    # 计算有效 token 的平均熵（仅统计 response 部分）
                    entropy       = log_res["token_entropy"]
                    valid_entropy = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1)
                    batch_entropies.append(valid_entropy.item())

                    # 如果使用 grpo_clip，记录 clip 比例
                    if "clip_fraction" in loss_meta:
                        batch_clip_fracs.append(loss_meta["clip_fraction"].item())

                # ── 梯度裁剪 + 参数更新 ──
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                global_train_step += 1

                # 记录训练指标到 wandb
                wandb.log({
                    "train_step":          global_train_step,
                    "train/loss":          batch_loss / gradient_accumulation_steps,
                    "train/grad_norm":     grad_norm.item(),
                    "train/token_entropy": sum(batch_entropies) / len(batch_entropies),
                    "train/reward_mean":   reward_meta["reward_mean"],
                    # clip_fraction 仅 grpo_clip 有意义，其他 loss_type 为 0
                    "train/clip_fraction": sum(batch_clip_fracs) / len(batch_clip_fracs) if batch_clip_fracs else 0.0,
                })

        # ── Phase 5：定期评估（每 5 个 GRPO step）──
        if (step + 1) % 5 == 0:
            policy_model.eval()
            # 热更新 vLLM 权重为最新策略
            load_policy_into_vllm_instance(policy_model, vllm_engine)

            # 取验证集前 1024 条（至少 1024 条保证评估统计量可靠）
            val_prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in val_data[:1024]]
            val_gts     = [ex["solution"] for ex in val_data[:1024]]

            # 每个 prompt 只生成 1 个回复（greedy/sampling 均可）用于评估
            val_outputs   = vllm_engine.generate(val_prompts, sampling_params)
            val_responses = [out.outputs[0].text for out in val_outputs]

            # 计算 reward / 格式正确率 / 平均长度等指标
            metrics = log_generations(val_prompts, val_responses, val_gts, r1_zero_reward_fn, step=step)
            metrics["eval_step"] = eval_step
            wandb.log(metrics)
            eval_step += 1

if __name__ == "__main__":
    main()
