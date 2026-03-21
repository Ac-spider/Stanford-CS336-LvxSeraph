import json
import random
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    log_generations
)
from drgrpo_grader import r1_zero_reward_fn

from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id, device=device, dtype=torch.bfloat16,
            enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def collate_experiences(batch_experiences: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(ex["input_ids"]) for ex in batch_experiences)

    padded_input_ids, padded_labels, padded_masks, padded_old_log_probs, advs = [], [], [], [], []

    for ex in batch_experiences:
        pad_len = max_len - len(ex["input_ids"])

        padded_input_ids.append(torch.cat([ex["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([ex["labels"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))

        padded_masks.append(torch.cat([ex["response_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        padded_old_log_probs.append(torch.cat([ex["old_log_prob"], torch.zeros(pad_len, dtype=torch.float32)]))

        advs.append(ex["advantage"])

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "response_mask": torch.stack(padded_masks),
        "old_log_probs": torch.stack(padded_old_log_probs),
        "advantages": torch.tensor(advs, dtype=torch.float32).unsqueeze(-1)
    }


def main():
    n_grpo_steps = 200
    rollout_batch_size = 256
    group_size = 8

    epochs_per_rollout_batch = 2
    train_batch_size = 256
    gradient_accumulation_steps = 128
    learning_rate = 5e-6

    loss_type = "grpo_clip"
    cliprange = 0.2

    advantage_eps = 1e-6
    sampling_temperature = 1.0
    sampling_min_tokens = 4
    sampling_max_tokens = 1024
    use_std_normalization = True

    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    wandb.init(project="cs336-alignment-grpo-offpolicy", config=locals())
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    device_policy = "cuda:0"
    device_vllm = "cuda:1"
    model_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device_policy)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    vllm_engine = init_vllm(model_path, device_vllm, seed=42)

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

    for step in range(n_grpo_steps):
        print(f"\n--- Off-Policy GRPO Step {step + 1}/{n_grpo_steps} ---")

        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, vllm_engine)

        sampled_batch = random.sample(train_data, n_prompts_per_rollout_batch)
        prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in sampled_batch]
        gts = [ex["solution"] for ex in sampled_batch]

        outputs = vllm_engine.generate(prompts, sampling_params)
        flat_prompts, flat_responses, flat_gts = [], [], []
        for i, output in enumerate(outputs):
            for j in range(group_size):
                flat_prompts.append(prompts[i])
                flat_responses.append(output.outputs[j].text)
                flat_gts.append(gts[i])

        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn, rollout_responses=flat_responses,
            repeated_ground_truths=flat_gts, group_size=group_size,
            advantage_eps=advantage_eps, normalize_by_std=use_std_normalization
        )

        print("Freezing old_log_probs...")
        experience_buffer = []

        with torch.inference_mode():
            for i in range(0, len(flat_prompts), micro_train_batch_size):
                chunk_p = flat_prompts[i:i + micro_train_batch_size]
                chunk_r = flat_responses[i:i + micro_train_batch_size]
                chunk_advs = advantages[i:i + micro_train_batch_size]

                toks = tokenize_prompt_and_output(chunk_p, chunk_r, tokenizer)
                input_ids = toks["input_ids"].to(device_policy)
                labels = toks["labels"].to(device_policy)
                response_mask = toks["response_mask"]

                log_res = get_response_log_probs(policy_model, input_ids, labels)
                old_log_probs = log_res["log_probs"].cpu()
                input_ids_cpu = input_ids.cpu()
                labels_cpu = labels.cpu()

                for j in range(len(chunk_p)):
                    real_len = torch.sum(input_ids_cpu[j] != pad_token_id).item()
                    if real_len == 0: real_len = len(input_ids_cpu[j])

                    experience_buffer.append({
                        "input_ids": input_ids_cpu[j][:real_len],
                        "labels": labels_cpu[j][:real_len],
                        "response_mask": response_mask[j][:real_len],
                        "old_log_prob": old_log_probs[j][:real_len],
                        "advantage": chunk_advs[j].item()
                    })

        policy_model.train()
        print(f"Starting Inner Optimization Loop ({epochs_per_rollout_batch} Epochs)...")

        for epoch in range(epochs_per_rollout_batch):
            random.shuffle(experience_buffer)

            for i in range(0, len(experience_buffer), train_batch_size):
                batch_exps = experience_buffer[i:i + train_batch_size]
                optimizer.zero_grad()

                batch_loss = 0.0
                batch_entropies, batch_clip_fracs = [], []

                for j in range(0, len(batch_exps), micro_train_batch_size):
                    micro_exps = batch_exps[j:j + micro_train_batch_size]

                    collated = collate_experiences(micro_exps, pad_token_id)
                    input_ids = collated["input_ids"].to(device_policy)
                    labels = collated["labels"].to(device_policy)
                    response_mask = collated["response_mask"].to(device_policy)
                    mb_old_log_probs = collated["old_log_probs"].to(device_policy)
                    mb_advs = collated["advantages"].to(device_policy)

                    log_res = get_response_log_probs(policy_model, input_ids, labels, return_token_entropy=True)

                    scaled_loss, loss_meta = grpo_microbatch_train_step(
                        policy_log_probs=log_res["log_probs"],
                        response_mask=response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        advantages=mb_advs,
                        old_log_probs=mb_old_log_probs,
                        cliprange=cliprange
                    )

                    batch_loss += loss_meta["loss"].item()

                    entropy = log_res["token_entropy"]
                    valid_entropy = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1)
                    batch_entropies.append(valid_entropy.item())
                    if "clip_fraction" in loss_meta:
                        batch_clip_fracs.append(loss_meta["clip_fraction"].item())

                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                global_train_step += 1

                wandb.log({
                    "train_step": global_train_step,
                    "train/loss": batch_loss / gradient_accumulation_steps,
                    "train/token_entropy": sum(batch_entropies) / len(batch_entropies),
                    "train/reward_mean": reward_meta["reward_mean"],
                    "train/clip_fraction": sum(batch_clip_fracs) / len(batch_clip_fracs) if batch_clip_fracs else 0.0,
                })

        if (step + 1) % 5 == 0:
            policy_model.eval()
            load_policy_into_vllm_instance(policy_model, vllm_engine)

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