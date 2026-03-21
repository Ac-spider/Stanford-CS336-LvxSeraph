import os
import regex as re
from collections import Counter, defaultdict
import multiprocessing as mp
from typing import BinaryIO

# ─────────────────────────────────────────
# GPT-2 预分词正则表达式
# Purpose: 将原始文本按照 GPT-2 的规则切分为若干「词」，再对每个词做 BPE
# Key concept: 预分词（pre-tokenization）——防止跨词边界合并，保证缩略词、数字、标点独立
# ─────────────────────────────────────────
GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将大文件分成若干字节范围，边界对齐到特殊 token（如 <|endoftext|>）处。
    Purpose: 支持多进程并行读取，每个进程处理一个 chunk，避免截断特殊 token。
    Key concept: 文件分块 + 边界对齐，保证每个 chunk 是完整的文档集合。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 先跳到文件末尾获取文件总大小，再跳回开头
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 均匀切分初始边界
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # 最后一个边界固定为文件末尾

    mini_chunk_size = 4096  # 每次向前读取 4 KB 寻找特殊 token

    # 对每个中间边界，向前扫描直到找到特殊 token，使边界对齐
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # 读到文件末尾，将边界设为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在当前小块中查找特殊 token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    # 去重并排序，防止多个边界落在同一位置
    return sorted(set(chunk_boundaries))


def _process_chunk(args) -> Counter:
    """
    处理文件的一个字节区间，统计预分词后每个「词」（字节元组）的出现频次。
    Purpose: 多进程 worker，将文本按特殊 token 切段后用 GPT-2 正则匹配词语。
    Key concept: 多进程 + Counter 词频统计，为 BPE 合并提供原始数据。
    """
    input_path, start, end, special_tokens = args

    # 按字节范围读取文本块
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    # 解码为 UTF-8 文本（忽略无效字节）
    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    # 先按特殊 token 切段，避免跨文档合并
    if special_tokens:
        escaped_specials = [re.escape(st) for st in special_tokens]
        split_pat = "|".join(escaped_specials)
        parts = re.split(split_pat, chunk_text)
    else:
        parts = [chunk_text]

    # 对每段文本用 GPT-2 正则提取词，记录每个词的字节元组表示及频次
    word_counts = Counter()
    for part in parts:
        for match in re.finditer(GPT2_PAT, part):
            # 将匹配到的词转为字节序列，每个字节独立为一个元素
            token_bytes = match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            word_counts[token_tuple] += 1

    return word_counts


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE（Byte Pair Encoding）分词器。
    Purpose: 从原始文本中学习合并规则，构建大小为 vocab_size 的词表。
    Key concept: BPE 算法——反复找频率最高的相邻字节对并合并，直到词表达到目标大小。

    Args:
        input_path: 训练语料文件路径
        vocab_size: 目标词表大小（含基础 256 字节 + 特殊 token）
        special_tokens: 不参与合并的特殊 token 列表（如 <|endoftext|>）
    Returns:
        vocab: {token_id -> bytes} 映射
        merges: 按合并顺序排列的 (bytes_a, bytes_b) 列表
    """
    # 初始化词表：前 256 个 ID 对应单字节（所有可能的 UTF-8 基础单元）
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # 将特殊 token 加入词表（不参与合并，直接整体映射）
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    # 利用所有 CPU 核心并行统计词频
    num_processes = mp.cpu_count()
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"

    # 找到文件分块边界，确保边界对齐到特殊 token
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    # 构造每个进程的参数
    chunk_args = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
    ]

    # 多进程并行统计词频，汇总各进程结果
    word_counts = Counter()
    with mp.Pool(num_processes) as pool:
        for chunk_counts in pool.imap_unordered(_process_chunk, chunk_args):
            word_counts.update(chunk_counts)

    merges = []
    current_vocab_size = len(vocab)

    # ─────────────────────────────────────────
    # 初始化增量缓存结构
    # Purpose: 避免每次合并后重新遍历所有词重新统计 pair 频次（O(V*L) -> O(受影响词数)）
    # Key concept: 增量更新——只对包含被合并 pair 的词做更新，大幅降低时间复杂度
    # ─────────────────────────────────────────
    pair_counts = defaultdict(int)    # pair -> 全局频次
    pair_to_words = defaultdict(set)  # pair -> 包含该 pair 的所有词集合（反向索引）

    # 初始化：遍历所有词，统计相邻字节对的出现频次
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    # ─────────────────────────────────────────
    # 迭代合并主循环
    # Purpose: 每轮选出频率最高的 pair 合并，直到词表达到目标大小
    # Key concept: 贪心策略——局部最优的频率合并近似全局最优的压缩率
    # ─────────────────────────────────────────
    while current_vocab_size < vocab_size:
        if not pair_counts:
            break

        # 获取频率最高的 pair；频率相同时按字典序取最大值（保证确定性）
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)

        # 新 token = 两个字节序列拼接，分配下一个可用 ID
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token_bytes
        next_id += 1
        current_vocab_size += 1

        # 取出所有含有 best_pair 的词，准备更新
        words_to_process = list(pair_to_words[best_pair])

        # 清理 best_pair 对应的索引（该 pair 已合并，不会再单独出现）
        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word in words_to_process:
            count = word_counts[word]
            if count == 0:
                continue

            # 步骤1：从缓存中移除旧词中各 pair 的贡献
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p == best_pair:
                    continue  # best_pair 已在外层清理，跳过防止重复操作

                pair_counts[p] -= count
                if pair_counts[p] <= 0:
                    del pair_counts[p]

                if word in pair_to_words.get(p, set()):
                    pair_to_words[p].remove(word)
                    if not pair_to_words[p]:
                        del pair_to_words[p]

            # 步骤2：生成合并后的新词（将所有 best_pair 替换为 new_token_bytes）
            new_word = []
            i = 0
            while i < len(word):
                # 若当前位置匹配 best_pair，则合并为新 token
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            # 步骤3：将新词的各 pair 加入缓存
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += count
                pair_to_words[p].add(new_word)

            # 步骤4：用新词替换旧词的频次记录
            del word_counts[word]
            word_counts[new_word] += count

    return vocab, merges
