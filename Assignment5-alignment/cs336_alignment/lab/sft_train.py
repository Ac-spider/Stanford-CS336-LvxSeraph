import json
import torch
import wandb
import random
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

# 从你自己的 utils 文件导入我们之前写的组件
from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations
)
from drgrpo_grader import r1_zero_reward_fn

# ======= 复制文档中提供的 vLLM 辅助函数 =======
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# ===============================================

def load_sft_data(path: str, max_examples: int = None) -> List[dict]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            if max_examples and len(data) >= max_examples:
                break
    return data


def main():
    # 超参数设置 (你可以使用 argparse 或 typer 提取出来以便 sweep)
    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    sft_data_path = "/data/a5-alignment/MATH/sft.jsonl"
    val_data_path = "/data/a5-alignment/MATH/validation.jsonl"

    num_train_examples = 1024  # 可选值: 128, 256, 512, 1024, None(全量)
    batch_size = 16
    gradient_accumulation_steps = 4
    learning_rate = 1e-5
    epochs = 3
    eval_every_n_steps = 50

    # 1. 初始化 Wandb
    wandb.init(project="cs336-alignment-sft", config={
        "lr": learning_rate, "batch_size": batch_size, "num_examples": num_train_examples
    })
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # 2. 加载模型与 Tokenizer
    print("Loading model and tokenizer...")
    device_policy = "cuda:0"
    device_vllm = "cuda:1"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 根据文档要求使用 bfloat16 和 flash_attention_2
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_policy)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    # 3. 初始化 vLLM (放在第二张卡上)
    print("Initializing vLLM on GPU 1...")
    vllm_engine = init_vllm(model_path, device_vllm, seed=42)

    # 4. 加载数据
    train_data = load_sft_data(sft_data_path, max_examples=num_train_examples)
    val_data = load_sft_data(val_data_path, max_examples=1024)  # 验证集取 1024 条即可

    global_step = 0
    eval_step = 0

    # 5. 开始训练循环
    for epoch in range(epochs):
        random.shuffle(train_data)

        # 简单手写 batch 生成器
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            prompt_strs = [ex["prompt"] for ex in batch]
            output_strs = [ex["response"] for ex in batch]

            # 分词并移至 GPU
            tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            input_ids = tokenized["input_ids"].to(device_policy)
            labels = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)

            # 前向传播：获取 log probs
            log_prob_res = get_response_log_probs(policy_model, input_ids, labels)
            policy_log_probs = log_prob_res["log_probs"]

            # 计算 Loss 并反向传播
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            # 梯度累积逻辑
            if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪 (clip_value = 1.0)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train_step": global_step
                })

                # ====== 验证阶段 ======
                if global_step % eval_every_n_steps == 0:
                    policy_model.eval()
                    # 同步权重到 vLLM
                    load_policy_into_vllm_instance(policy_model, vllm_engine)

                    val_prompts = [ex["prompt"] for ex in val_data]
                    val_gts = [ex["ground_truth"] for ex in val_data]  # 注意：具体键名视验证集而定

                    sampling_params = SamplingParams(
                        temperature=1.0, top_p=1.0, max_tokens=1024,
                        stop=["</answer>"], include_stop_str_in_output=True
                    )

                    # 使用 vLLM 生成
                    outputs = vllm_engine.generate(val_prompts, sampling_params)
                    val_responses = [out.outputs[0].text for out in outputs]

                    # 记录和日志
                    metrics = log_generations(
                        val_prompts, val_responses, val_gts, r1_zero_reward_fn, step=global_step
                    )
                    metrics["eval_step"] = eval_step
                    wandb.log(metrics)

                    eval_step += 1
                    policy_model.train()


if __name__ == "__main__":
    main()