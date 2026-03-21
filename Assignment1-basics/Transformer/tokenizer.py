import regex as re
import pickle
from typing import Iterable, Iterator

# ─────────────────────────────────────────
# GPT-2 预分词正则表达式（与 bpe.py 保持一致）
# Purpose: 将文本切分为若干「词」，再对每个词独立做 BPE 编码
# Key concept: 预分词确保 BPE 不跨越自然词边界（如缩略词、数字、标点各自独立）
# ─────────────────────────────────────────
GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    """
    基于 BPE 的文本分词器。
    Purpose: 将原始文本编码为 token ID 序列，或将 token ID 序列解码回文本。
    Key concept: BPE 分词——利用训练阶段学到的合并规则，贪心地将字节序列合并为更长 token。
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """
        Args:
            vocab: {token_id -> bytes} 映射（由 train_bpe 生成）
            merges: 按优先级排列的合并规则列表，索引越小优先级越高
            special_tokens: 不参与 BPE 合并、直接整体匹配的特殊 token
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 若词表中尚未包含特殊 token，则动态追加
        if self.special_tokens:
            existing_bytes = set(self.vocab.values())
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for st in self.special_tokens:
                st_bytes = st.encode('utf-8')
                if st_bytes not in existing_bytes:
                    self.vocab[next_id] = st_bytes
                    existing_bytes.add(st_bytes)
                    next_id += 1

        # 构建反向词表：bytes -> token_id，用于编码时快速查找
        self.inverse_vocab = {v: i for i, v in self.vocab.items()}

        # 构建合并规则排名表：pair -> rank，rank 越小优先级越高
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}

        # 编译特殊 token 的正则表达式，用于在 encode 时先切出特殊 token
        if self.special_tokens:
            escaped = [re.escape(st) for st in self.special_tokens]
            self.special_pat = re.compile('(' + '|'.join(escaped) + ')')
        else:
            self.special_pat = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """从 pickle 文件加载词表和合并规则，快速恢复已训练的分词器。"""
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text: str) -> list[int]:
        """
        对不含特殊 token 的文本段进行 BPE 编码。
        Purpose: 用预分词正则切词后，对每个词贪心应用 BPE 合并规则。
        Key concept: 贪心 BPE 解码——每次选 rank 最小（优先级最高）的相邻 pair 合并。
        """
        ids = []

        # 用 GPT-2 正则将文本切为词，逐词编码
        for match in re.finditer(GPT2_PAT, text):
            # 将词转为字节序列，每个字节初始为独立元素
            token_bytes = match.group().encode('utf-8')
            b_list = [bytes([b]) for b in token_bytes]

            # 反复合并 rank 最低的 pair，直到无可合并为止
            while len(b_list) >= 2:
                best_pair = None
                min_rank = float('inf')

                # 遍历所有相邻对，找到 rank 最低（优先级最高）的 pair
                for i in range(len(b_list) - 1):
                    pair = (b_list[i], b_list[i + 1])
                    rank = self.merges_rank.get(pair, float('inf'))
                    if rank < min_rank:
                        best_pair = pair
                        min_rank = rank

                # 没有可合并的 pair，退出循环
                if not best_pair:
                    break

                # 将所有出现的 best_pair 替换为合并后的新 token
                i = 0
                new_b_list = []
                while i < len(b_list):
                    if (i < len(b_list) - 1
                            and b_list[i] == best_pair[0]
                            and b_list[i + 1] == best_pair[1]):
                        new_b_list.append(b_list[i] + b_list[i + 1])
                        i += 2
                    else:
                        new_b_list.append(b_list[i])
                        i += 1
                b_list = new_b_list

            # 将最终的字节序列转为 token ID
            for b in b_list:
                ids.append(self.inverse_vocab[b])

        return ids

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 列表。
        Purpose: 先用特殊 token 正则切分文本，再对普通段落调用 _encode_chunk。
        Key concept: 特殊 token 不参与 BPE，直接整体映射为对应 ID。
        """
        if not self.special_tokens:
            return self._encode_chunk(text)

        ids = []
        # 按特殊 token 切分文本（保留分隔符）
        parts = self.special_pat.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                # 特殊 token 直接查反向词表获取 ID
                part_bytes = part.encode('utf-8')
                ids.append(self.inverse_vocab[part_bytes])
            else:
                # 普通文本段做 BPE 编码
                ids.extend(self._encode_chunk(part))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对可迭代文本流进行流式编码，逐个 yield token ID。
        Purpose: 处理超大文本时节省内存，避免一次性加载全文。
        Key concept: 生成器（Generator）——惰性求值，按需产出 token。
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码回原始文本。
        Purpose: 将模型输出的 ID 序列转换为人类可读文本。
        Key concept: 反向词表查找 + 字节拼接 + UTF-8 解码（errors='replace' 容错）。
        """
        b_list = []
        for token_id in ids:
            if token_id in self.vocab:
                b_list.append(self.vocab[token_id])
            else:
                raise ValueError(f'Token_id:{token_id} Not Found')

        # 拼接所有字节后统一解码为 UTF-8 字符串
        b_text = b''.join(b_list)
        return b_text.decode(encoding='utf-8', errors='replace')
