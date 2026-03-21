import os
import json
import gzip
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any
from transformers import PreTrainedTokenizerBase


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool
    ):
        self.seq_length = seq_length
        self.examples = []

        # 1. 兼容读取 .gz 或普通 .jsonl 文件
        lines = []
        open_func = gzip.open if str(dataset_path).endswith('.gz') else open
        with open_func(dataset_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    lines.append(line)

        # 2. 如果要求打乱，在拼接前打乱样本顺序
        if shuffle:
            random.shuffle(lines)

        # 3. Alpaca 的指令模板
        template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n{response}"
        )

        all_tokens = []
        # 获取分词器的 EOS Token ID，用于分隔不同的样本
        eos_token_id = tokenizer.eos_token_id

        # 4. 遍历并拼接所有 Token
        for line in lines:
            data = json.loads(line)
            # 兼容可能不同的键名 (prompt/instruction, response/output)
            prompt = data.get("prompt", data.get("instruction", ""))
            response = data.get("response", data.get("output", ""))

            text = template.format(prompt=prompt, response=response)

            # 编码时不自动添加特殊 token，我们手动添加 eos_token_id 来作为边界
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(eos_token_id)

        # 5. 切割成固定长度 seq_length 的数据块
        for i in range(0, len(all_tokens) - seq_length + 1, seq_length):
            chunk = all_tokens[i:i + seq_length]
            self.examples.append({
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                # 对于标准的 Causal LM，labels 通常与 input_ids 相同
                # (HuggingFace 的模型内部会自动执行 shift 操作: logits[:, :-1], labels[:, 1:])
                "labels": torch.tensor(chunk, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def get_packed_sft_dataset(
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
) -> Dataset:
    """实例化并返回 PackedSFTDataset"""
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)


def run_iterate_batches(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
):
    """
    返回一个批量迭代器。直接使用 PyTorch 的 DataLoader 即可完美满足需求。
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)