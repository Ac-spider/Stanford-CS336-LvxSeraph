import json
import torch
import wandb
import random
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

# 从自定义 utils 模块导入分词、log_prob 计算、SFT 训练步、生成日志等工具函数
from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations
)
# 自定义数学题评测奖励函数（检查格式 + 答案正确性）
from drgrpo_grader import r1_zero_reward_fn

# ─────────────────────────────────────────
# vLLM 推理引擎辅助函数
# Purpose: 在独立 GPU 上启动 vLLM 推理引擎，
#          使训练（PyTorch）与推理（vLLM）可以并行使用不同 GPU，
#          避免显存争抢。
# ─────────────────────────────────────────
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy."""
    # 固定 vLLM 随机种子，保证生成可复现
    vllm_set_random_seed(seed)
    # patch 掉分布式相关检查（单机单卡推理时不需要多进程通信）
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
            enable_prefix_caching=True,         # 开启 prefix KV cache 加速重复 prompt
            gpu_memory_utilization=gpu_memory_utilization,
        )


# ─────────────────────────────────────────
# load_policy_into_vllm_instance
# Purpose: 将 PyTorch 策略模型的最新权重热加载（hot-swap）到 vLLM 引擎，
#          无需重启引擎即可用最新策略生成新样本。
# Key concept:
#   直接访问 vLLM 内部的 model_runner，调用其 load_weights 接口，
#   将 state_dict 的 (name, tensor) 对传入完成原地权重替换。
# ─────────────────────────────────────────
def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    # 获取策略模型的完整参数字典
    state_dict = policy.state_dict()
    # 深入 vLLM 引擎内部获取模型实例
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # 原地更新 vLLM 模型权重（热更新，无需重新分配显存）
    llm_model.load_weights(state_dict.items())


def load_sft_data(path: str, max_examples: int = None) -> List[dict]:
    """从 JSONL 文件按行读取数据，max_examples 控制最大加载条数（调试时可用）。"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            if max_examples and len(data) >= max_examples:
                break
    return data


# ─────────────────────────────────────────
# main
# Purpose: SFT（监督微调）主训练循环
# 整体流程：
#   1. 初始化 wandb / 模型 / tokenizer / vLLM 引擎
#   2. 加载训练集和验证集
#   3. 外层循环 epoch，内层循环 batch：
#      a. 分词 → 前向传播 → 计算 SFT loss → backward（梯度累积）
#      b. 每 gradient_accumulation_steps 步执行一次 optimizer.step()
#      c. 每 eval_every_n_steps 步：热更新 vLLM 权重 → 生成验证集回复 → 记录指标
# ─────────────────────────────────────────
def main():
    # ── 超参数 ──
    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    sft_data_path = "/data/a5-alignment/MATH/sft.jsonl"
    val_data_path = "/data/a5-alignment/MATH/validation.jsonl"

    num_train_examples = 1024       # 训练样本数，可选 128/256/512/1024/None(全量)
    batch_size = 16                 # 每次前向的样本数
    gradient_accumulation_steps = 4 # 累积多少步后更新参数
    learning_rate = 1e-5
    epochs = 3
    eval_every_n_steps = 50         # 每隔多少个 optimizer step 做一次评估

    # 1. 初始化 Wandb，定义 train/eval 两个独立的 step 轴
    wandb.init(project="cs336-alignment-sft", config={
        "lr": learning_rate, "batch_size": batch_size, "num_examples": num_train_examples
    })
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # 2. 加载模型与 Tokenizer
    print("Loading model and tokenizer...")
    device_policy = "cuda:0"   # 策略模型（训练）放在 GPU 0
    device_vllm   = "cuda:1"   # vLLM 推理引擎放在 GPU 1（避免训练/推理显存争抢）

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # bfloat16 节省显存；flash_attention_2 加速注意力计算
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_policy)

    # AdamW 优化器（标准 LLM 微调配置）
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    # 3. 初始化 vLLM（放在第二张卡，用于评估阶段快速生成）
    print("Initializing vLLM on GPU 1...")
    vllm_engine = init_vllm(model_path, device_vllm, seed=42)

    # 4. 加载数据
    train_data = load_sft_data(sft_data_path, max_examples=num_train_examples)
    val_data   = load_sft_data(val_data_path,  max_examples=1024)

    global_step = 0   # optimizer.step() 计数（梯度更新次数）
    eval_step   = 0   # 评估计数

    # 5. 训练主循环
    for epoch in range(epochs):
        # 每个 epoch 随机打乱训练数据
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            prompt_strs = [ex["prompt"]   for ex in batch]
            output_strs = [ex["response"] for ex in batch]

            # ── 分词并移至 GPU ──
            tokenized    = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            input_ids    = tokenized["input_ids"].to(device_policy)
            labels       = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)

            # ── 前向传播：获取 response 部分每个 token 的 log_prob ──
            log_prob_res     = get_response_log_probs(policy_model, input_ids, labels)
            policy_log_probs = log_prob_res["log_probs"]

            # ── 计算 SFT loss 并 backward（梯度累积：scaled_loss = loss / grad_acc_steps）──
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            # ── 梯度累积逻辑：每累积 gradient_accumulation_steps 个 microbatch 后更新参数 ──
            if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪（max_norm=1.0）防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,  # 还原未缩放的 loss
                    "train_step": global_step
                })

                # ── 定期评估 ──
                if global_step % eval_every_n_steps == 0:
                    policy_model.eval()
                    # 将最新训练权重同步到 vLLM（热更新，无需重启引擎）
                    load_policy_into_vllm_instance(policy_model, vllm_engine)

                    val_prompts = [ex["prompt"]       for ex in val_data]
                    val_gts     = [ex["ground_truth"] for ex in val_data]  # 键名视数据集格式而定

                    # vLLM 采样参数：确定性解码（temperature=1.0）
                    sampling_params = SamplingParams(
                        temperature=1.0, top_p=1.0, max_tokens=1024,
                        stop=["</answer>"], include_stop_str_in_output=True
                    )

                    # 批量推理生成验证集回复
                    outputs      = vllm_engine.generate(val_prompts, sampling_params)
                    val_responses = [out.outputs[0].text for out in outputs]

                    # 计算评估指标并记录到 wandb
                    metrics = log_generations(
                        val_prompts, val_responses, val_gts, r1_zero_reward_fn, step=global_step
                    )
                    metrics["eval_step"] = eval_step
                    wandb.log(metrics)

                    eval_step += 1
                    policy_model.train()  # 切回训练模式


if __name__ == "__main__":
    main()
