# CS336 Assignment 5: Alignment

## Assignment Description

> 实现大语言模型对齐训练的三种方法：SFT（监督微调）、DPO（直接偏好优化）和 GRPO（Group Relative Policy Optimization，组相对策略优化）。
> 任务模型为 Qwen2.5-Math-1.5B，目标任务为 MATH 数学题解答，评测指标为回复格式合规性与答案正确性。

---

## Tech Stack

| 类别 | 详情 |
|------|------|
| Language | Python 3.10+ |
| Model | Qwen2.5-Math-1.5B |
| 推理加速 | vLLM（独立 GPU，热更新权重） |
| 训练框架 | PyTorch + HuggingFace Transformers |
| 注意力实现 | Flash Attention 2（bfloat16） |
| 实验追踪 | Weights & Biases (wandb) |
| 数学评测 | 自定义 `r1_zero_reward_fn`（格式奖励 + 答案奖励） |
| 优化器 | AdamW（SFT: lr=1e-5；GRPO: lr=1e-5, betas=(0.9,0.95)） |

---

## Core Knowledge Points

1. **SFT 监督微调**：在专家演示数据上做有监督训练，response_mask 只对 output 部分计算 cross-entropy loss，避免惩罚 prompt token。

2. **DPO 直接偏好优化**：无需显式奖励模型，通过比较 chosen/rejected 回复相对参考模型的对数比率差来学习偏好。损失公式：
   ```
   L = -log σ( β * ( log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) ) )
   ```

3. **GRPO 组相对策略优化**：对每个 prompt 采样 G 个回复（group_size），组内计算相对奖励 advantage，替代传统 PPO 中的 value network，降低参数量和计算开销。

4. **REINFORCE 策略梯度**：基础策略梯度算法，损失为 `L = -R * log π_θ(a|s)`；no_baseline 模式直接用原始奖励，reinforce_with_baseline 模式用组归一化后的 advantage 降低方差。

5. **PPO Clip 目标**：限制新旧策略的偏离程度，提升训练稳定性：
   ```
   L_clip = -E[ min( ratio * A,  clip(ratio, 1-ε, 1+ε) * A ) ]
   ```
   其中 ratio = π_θ(a|s) / π_θ_old(a|s)，ε=0.2（cliprange）。

6. **组内奖励归一化**：将 rollout_batch_size 个奖励按 group_size 分组，组内减均值、除标准差，得到 advantage，消除不同 prompt 绝对难度差异的干扰。

7. **梯度累积（Gradient Accumulation）**：将大 batch 拆分为多个 micro_batch 依次前向/反向，累积梯度后统一更新，等效于更大 batch_size 但节省显存。每次 backward 前将 loss 除以累积步数保证梯度尺度不变。

8. **vLLM 推理加速**：将推理引擎部署在独立 GPU，通过 `load_policy_into_vllm_instance` 热更新（hot-swap）权重，无需重启引擎即可让 vLLM 使用最新策略生成样本。开启 prefix KV cache 加速含公共前缀的 prompt 批量推理。

---

## Code Structure

```
Assignment5-alignment/
├── cs336_alignment/
│   ├── dpo_train.py        # DPO 损失计算（per-instance）
│   ├── utils.py            # 共用工具函数（分词、log_prob、各类 loss、奖励归一化）
│   ├── sft_train.py        # SFT 主训练循环（vLLM 评估 + AdamW 训练）
│   └── grpo_train.py       # GRPO 主训练循环（rollout → 归一化 → 多 epoch 训练）
├── prompts/
│   └── r1_zero.prompt      # R1-Zero 风格 prompt 模板（含 {question} 占位符）
└── README.md
```

### 核心函数说明

| 文件 | 函数 | 说明 |
|------|------|------|
| `utils.py` | `tokenize_prompt_and_output` | 拼接分词，生成 response_mask |
| `utils.py` | `compute_entropy` | log-sum-exp 数值稳定的熵计算 |
| `utils.py` | `get_response_log_probs` | 模型前向，gather 真实 token 的 log_prob |
| `utils.py` | `sft_microbatch_train_step` | SFT masked 求和 + 梯度累积 + backward |
| `utils.py` | `compute_group_normalized_rewards` | GRPO 组内奖励归一化 → advantage |
| `utils.py` | `compute_naive_policy_gradient_loss` | REINFORCE: `-advantage * log_prob` |
| `utils.py` | `compute_grpo_clip_loss` | PPO clip 损失 + clip_fraction 统计 |
| `utils.py` | `compute_policy_gradient_loss` | 三种 loss_type 的分发器 |
| `utils.py` | `grpo_microbatch_train_step` | GRPO masked_mean + 梯度累积 + backward |
| `dpo_train.py` | `run_compute_per_instance_dpo_loss` | 单样本 DPO 损失（含 shift & gather） |
| `sft_train.py` | `load_policy_into_vllm_instance` | 热更新 PyTorch 权重到 vLLM 引擎 |
| `sft_train.py` | `main` | SFT 完整训练 + 验证循环 |
| `grpo_train.py` | `main` | GRPO rollout → 奖励 → 训练 → 评估循环 |

---

## How to Run

### 环境准备

```bash
pip install torch transformers vllm wandb
# 安装作业提供的 drgrpo_grader（包含 r1_zero_reward_fn）
```

### SFT 训练

```bash
cd Assignment5-alignment
python -m cs336_alignment.sft_train
# 训练数据：/data/a5-alignment/MATH/sft.jsonl
# 验证数据：/data/a5-alignment/MATH/validation.jsonl
# 需要双卡 GPU（cuda:0 训练，cuda:1 推理）
```

### DPO 训练

```python
# 在代码中直接调用
from cs336_alignment.dpo_train import run_compute_per_instance_dpo_loss
loss = run_compute_per_instance_dpo_loss(
    lm, lm_ref, tokenizer, beta=0.1,
    prompt=..., response_chosen=..., response_rejected=...
)
```

### GRPO 训练

```bash
python -m cs336_alignment.grpo_train
# 超参数在 main() 开头直接修改：
#   loss_type: "no_baseline" / "reinforce_with_baseline" / "grpo_clip"
#   group_size: 每个 prompt 的采样数（默认 8）
#   epochs_per_rollout_batch: On-policy=1，Off-policy>1
```

---

## Key Results / Observations

- **SFT** 在少量（128-1024 条）专家数据上微调后，模型能快速学会 `<think>/<answer>` 的输出格式，format_reward 显著提升。
- **GRPO + reinforce_with_baseline** 训练曲线比 no_baseline 更稳定，reward 方差更小；token_entropy 随训练逐步下降，说明策略逐渐收敛。
- **group_size=8** 在奖励信号稀疏（数学题答案对/错二值奖励）时尤为重要：一个 prompt 生成多个回复，总能包含至少一个正确/错误样本，保证 advantage 非零。
- **vLLM 热更新**是整个在线训练流程的关键工程细节，每次策略更新后无需重启推理引擎，节省大量初始化时间。

---

## Notes

- 所有模型使用 `bfloat16` 精度和 `flash_attention_2`，在 A100/H100 上可获得最佳性能。
- `gradient_accumulation_steps` 在 GRPO 中设为 128，等效 batch_size=256，但实际每次前向只处理 2 条样本，适合显存受限的环境。
- DPO 的 `beta` 超参数控制偏离参考模型的惩罚强度，建议范围 0.05-0.5。
- GRPO 的 `advantage_eps` 仅在不使用 std 归一化时作为分母下界；`normalize_by_std=True` 时分母由 `group_stds` 决定。
