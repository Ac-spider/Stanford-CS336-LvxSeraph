import json
import torch
import random
import wandb
from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations
)
from drgrpo_grader import r1_zero_reward_fn


def main():
    n_ei_steps = 5
    G = 8  # 每个问题的 rollout 数量
    sft_epochs = 1  # 每次 EI 步中 SFT 的 epoch 数
    db_size = 1024  # 每步采样的题目数量 (D_b)
    sft_batch_size = 16
    gradient_accumulation_steps = 4
    learning_rate = 1e-5

    wandb.init(project="cs336-alignment-ei", config={
        "G": G, "sft_epochs": sft_epochs, "db_size": db_size
    })

    device_policy = "cuda:0"
    device_vllm = "cuda:1"
    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_policy)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    vllm_engine = init_vllm(model_path, device_vllm, seed=42)
    load_policy_into_vllm_instance(policy_model, vllm_engine)

    with open("/data/a5-alignment/MATH/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]
    with open("/data/a5-alignment/MATH/validation.jsonl", "r") as f:
        val_data = [json.loads(line) for line in f][:1024]  # 验证集采样

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        min_tokens=4,  # 确保不会生成空字符串引发 NaN
        n=G,  # 让 vLLM 为每个 prompt 生成 G 个输出
        stop=["</answer>"],  # 在第二个 answer 标签处停止
        include_stop_str_in_output=True
    )

    global_step = 0

    for ei_step in range(n_ei_steps):
        print('START')

    sampled_prompts_data = random.sample(train_data, db_size)
    prompts = [ex["prompt"] for ex in sampled_prompts_data]  # 假设已经使用了 r1_zero 格式化
    gts = [ex["ground_truth"] for ex in sampled_prompts_data]

    print("Generating rollouts...")
    outputs = vllm_engine.generate(prompts, sampling_params)

    d_sft = []
    for i, output in enumerate(outputs):
        gt = gts[i]
        prompt = prompts[i]
        for j in range(G):
            response_text = output.outputs[j].text
            reward_dict = r1_zero_reward_fn(response_text, gt)

            if reward_dict.get("reward", 0.0) > 0:
                d_sft.append({"prompt": prompt, "response": response_text})

    print(f"Generated {len(outputs) * G} total responses. Kept {len(d_sft)} correct responses for SFT.")


    policy_model.train()
    for epoch in range(sft_epochs):
        random.shuffle(d_sft)
        for i in range(0, len(d_sft), sft_batch_size):
            batch = d_sft[i:i + sft_batch_size]
            batch_prompts = [ex["prompt"] for ex in batch]
            batch_responses = [ex["response"] for ex in batch]

            tokenized = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
            input_ids = tokenized["input_ids"].to(device_policy)
            labels = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)

            # 获取 log probs 和 熵 [cite: 263-264, 454]
            log_prob_res = get_response_log_probs(
                policy_model, input_ids, labels, return_token_entropy=True
            )

            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=log_prob_res["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            if (i // sft_batch_size + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # [cite: 449]
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                entropy = log_prob_res["token_entropy"]
                valid_entropy = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1)

                wandb.log({
                    "train/sft_loss": loss.item() * gradient_accumulation_steps,
                    "train/token_entropy": valid_entropy.item(),
                    "train_step": global_step
                })


    policy_model.eval()
    load_policy_into_vllm_instance(policy_model, vllm_engine)

if __name__ == "__main__":
    main()



















