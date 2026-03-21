import json
from typing import Callable, List
from vllm import LLM, SamplingParams
# 导入提供的奖励函数
from drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sampling_params: SamplingParams
) -> List[dict]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and return results.
    """
    # 1. 使用 vLLM 批量生成回复 [cite: 117]
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for output, prompt, gt in zip(outputs, prompts, ground_truths):
        generated_text = output.outputs[0].text

        # 2. 计算奖励 [cite: 135]
        # 注意：r1_zero_reward_fn 会返回一个包含 format_reward, answer_reward, reward 的字典
        reward_dict = reward_fn(generated_text, gt)

        results.append({
            "prompt": prompt,
            "ground_truth": gt,
            "generated_text": generated_text,
            "format_reward": reward_dict.get("format_reward", 0.0),
            "answer_reward": reward_dict.get("answer_reward", 0.0),
            "total_reward": reward_dict.get("reward", 0.0)
        })
    return results


def main():
    # 1. 读取 r1_zero prompt 模板 [cite: 83]
    with open("prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    # 2. 加载 MATH 验证集 [cite: 147]
    val_data_path = "/data/a5-alignment/MATH/validation.jsonl"
    prompts = []
    ground_truths = []
    original_data = []

    with open(val_data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            original_data.append(data)
            # 根据具体的数据集格式提取问题和答案，这里假设键为 'problem' 和 'solution' / 'answer'
            question = data.get("problem", data.get("question", ""))
            answer = data.get("solution", data.get("answer", ""))

            # 格式化 prompt [cite: 84]
            formatted_prompt = prompt_template.replace("{question}", question)
            prompts.append(formatted_prompt)
            ground_truths.append(answer)

    # 3. 初始化 vLLM SamplingParams [cite: 141, 144]
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 4. 初始化 vLLM 模型 (使用预下载的路径) [cite: 115, 127]
    llm = LLM(model="/data/a5-alignment/models/Qwen2.5-Math-1.5B")

    # 5. 运行评估 [cite: 149]
    print("开始生成和评估...")
    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)

    # 6. 计算总体指标并序列化到磁盘 [cite: 148]
    total_format_correct = sum(1 for r in results if r["format_reward"] == 1.0)
    total_answer_correct = sum(1 for r in results if r["answer_reward"] == 1.0)

    print(f"总计 Format 正确率: {total_format_correct / len(results):.2%}")
    print(f"总计 Answer 正确率: {total_answer_correct / len(results):.2%}")

    with open("baseline_results.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()