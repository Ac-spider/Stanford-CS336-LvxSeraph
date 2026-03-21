import torch
from model import TransformerLM, softmax
from tokenizer import Tokenizer

def load_model_and_tokenizer(checkpoint_path, vocab_path, merges_path):
    """
    从检查点加载模型和分词器。
    Purpose: 恢复训练好的模型用于文本生成推理。
    Key concept: 推理模式（model.eval()）——关闭 Dropout/BatchNorm 的随机性，节省计算图内存。
    """
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    # 与训练时相同的模型超参数（必须完全一致才能正确加载权重）
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    num_layers = 4
    num_heads = 16

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, device=device)

    print(f"正在加载检查点: {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 切换到推理模式

    return model, tokenizer, device, context_length


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, context_length: int,
                  temperature: float = 1.0, top_p: float = 0.9, device: str = "cpu"):
    """
    自回归文本生成（支持 Temperature + Top-p 采样）。
    Purpose: 给定提示词，逐步生成新 token 直到达到最大长度或遇到结束符。
    Key concept:
      - 自回归生成：每次只生成一个 token，将其追加到输入序列后继续生成
      - Temperature 采样：logit / temperature，temperature 越低分布越尖锐（更确定），越高越随机
      - Top-p（Nucleus）采样：只保留累积概率超过 p 的最小 token 集合再采样，平衡多样性和质量
    """
    # 将 prompt 编码为 token ID 并转为张量
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    print(f"\n[Prompt]: {prompt}")
    print("-" * 40)

    # 获取结束符 <|endoftext|> 的 ID，生成时遇到则停止
    end_token_bytes = b"<|endoftext|>"
    end_id = tokenizer.inverse_vocab.get(end_token_bytes)

    with torch.no_grad():  # 推理阶段不需要梯度，节省显存
        for _ in range(max_new_tokens):
            # 只取最近 context_length 个 token 作为输入（滑动窗口）
            logits = model(x[:, -context_length:])[0, -1, :]  # 取最后一个位置的 logit

            if temperature == 0:
                # 贪婪解码：直接取概率最高的 token
                next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            else:
                # Temperature 缩放：调整分布的尖锐程度
                probs = softmax(logits / temperature, -1)

                if top_p < 1:
                    # ── Top-p（Nucleus）采样 ──
                    # 按概率从大到小排序
                    sorted_probs, sorted_indices = torch.sort(probs, -1, descending=True)
                    # 计算累积概率
                    cumsum_probs = torch.cumsum(sorted_probs, -1)
                    # 找出累积概率超过 top_p 的位置（需保留第一个超过的 token）
                    sorted_probs_to_remove = cumsum_probs > top_p
                    # 右移一位：确保至少保留一个 token（防止 nucleus 为空）
                    sorted_probs_to_remove[1:] = sorted_probs_to_remove[:-1].clone()
                    sorted_probs_to_remove[0] = False

                    # 将待移除位置的概率置为 0
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                        -1, sorted_indices, sorted_probs_to_remove
                    )
                    probs[indices_to_remove] = 0
                    # 重新归一化（使概率之和为 1）
                    probs = probs / torch.sum(probs, -1, keepdim=True)

                # 按概率分布随机采样下一个 token
                next_token = torch.multinomial(probs, 1).unsqueeze(0)

            # 将新 token 追加到序列末尾（自回归）
            x = torch.cat((x, next_token), -1)

            # 遇到结束符则停止生成
            if next_token.item() == end_id:
                break

        generated = tokenizer.decode(x[0].tolist())

    return generated


if __name__ == '__main__':
    import os

    vocab_file = "outputs/TinyStories_vocab.pkl"
    merges_file = "outputs/TinyStories_merges.pkl"

    # 依次加载不同训练步数的检查点，观察模型生成质量随训练进度的变化
    for step in range(1000, 10001, 1500):
        checkpoint_file = f"checkpoints/model_step_{step}.pt"

        if not os.path.exists(checkpoint_file):
            print(f"找不到检查点 {checkpoint_file}。请先运行训练或修改路径。")
            continue

        model, tokenizer, device, context_length = load_model_and_tokenizer(
            checkpoint_file, vocab_file, merges_file
        )

        prompt_text = "Once upon a time, there was an evil dragon who"

        generated_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=100,
            context_length=context_length,
            temperature=0.8,
            top_p=0.9,
            device=device
        )

        print(f'{step}/10000,\n{generated_output}')
