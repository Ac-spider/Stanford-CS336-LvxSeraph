import json
from typing import Callable, List
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sampling_params: SamplingParams
) -> List[dict]:

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for output, prompt, gt in zip(outputs, prompts, ground_truths):
        generated_text = output.outputs[0].text

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
    with open("prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    val_data_path = "/data/a5-alignment/MATH/validation.jsonl"
    prompts = []
    ground_truths = []
    original_data = []

    with open(val_data_path,'r') as f:
        for line in f:
            data = json.loads(line)
            original_data.append(data)

            question = data.get("problem",data.get("question",""))
            answer = data.get("solution",data.get("answer",""))

            formatted_prompt = prompt_template.replace("{question}",question)
            prompts.append(formatted_prompt)
            ground_truths.append(answer)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    llm = LLM(model="/data/a5-alignment/models/Qwen2.5-Math-1.5B")

    print("Start")
    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)

    total_format_correct = sum(1 for r in results if r["format_reward"] == 1.0)
    total_answer_correct = sum(1 for r in results if r["answer_reward"] == 1.0)

    print(f"Format Accuracy: {total_format_correct / len(results):.2%}")
    print(f"Answer Accuracy: {total_answer_correct / len(results):.2%}")

    with open("baseline_results.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()













