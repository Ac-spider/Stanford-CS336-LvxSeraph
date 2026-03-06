import os
import regex as re
from collections import Counter, defaultdict
import multiprocessing as mp
from typing import BinaryIO

GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _process_chunk(args) -> Counter:
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    if special_tokens:
        escaped_specials = [re.escape(st) for st in special_tokens]
        split_pat = "|".join(escaped_specials)
        parts = re.split(split_pat, chunk_text)
    else:
        parts = [chunk_text]

    word_counts = Counter()
    for part in parts:
        for match in re.finditer(GPT2_PAT, part):
            token_bytes = match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            word_counts[token_tuple] += 1

    return word_counts


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    num_processes = mp.cpu_count()

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    chunk_args = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
    ]

    word_counts = Counter()
    with mp.Pool(num_processes) as pool:
        for chunk_counts in pool.imap_unordered(_process_chunk, chunk_args):
            word_counts.update(chunk_counts)

    merges = []
    current_vocab_size = len(vocab)

    # --- 初始化增量缓存结构 ---
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    # --- 迭代合并 ---
    while current_vocab_size < vocab_size:
        if not pair_counts:
            break

        # 获取频率最高（且字典序最大）的 pair
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)

        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token_bytes
        next_id += 1
        current_vocab_size += 1

        # 取出包含 best_pair 的所有 word
        words_to_process = list(pair_to_words[best_pair])

        # 清理 best_pair 对应的索引（该 pair 已经合并，不会再独立出现）
        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word in words_to_process:
            count = word_counts[word]
            if count == 0:
                continue

            # 1. 移除旧 word 中 pairs 的计数和反向索引
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p == best_pair:
                    continue  # best_pair 已经在循环外被彻底清理，直接跳过

                pair_counts[p] -= count
                if pair_counts[p] <= 0:
                    del pair_counts[p]

                if word in pair_to_words.get(p, set()):
                    pair_to_words[p].remove(word)
                    if not pair_to_words[p]:
                        del pair_to_words[p]

            # 2. 生成替换后的新 word
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            # 3. 将新 word 中的 pairs 添加到计数和反向索引中
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += count
                pair_to_words[p].add(new_word)

            # 4. 更新 word_counts
            del word_counts[word]
            word_counts[new_word] += count

    return vocab, merges




