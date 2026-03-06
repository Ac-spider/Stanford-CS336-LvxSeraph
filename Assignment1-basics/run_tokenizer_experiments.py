import os
import time
import numpy as np
from tokenizer import Tokenizer

def run_experiments():
    dataset_path = "TinyStories.txt"
    vocab_path = "outputs/TinyStories_vocab.pkl"
    merges_path = "outputs/TinyStories_merges.pkl"
    output_npy_path = "outputs/TinyStories_tokens.npy"

    print("加载分词器...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    print("\n--- 1. 计算压缩率 ---")

    sample_docs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        buffer = ""
        while len(sample_docs) < 10:
            line = f.readline()
            if not line:
                break
            buffer += line
            if "<|endoftext|>" in line:
                docs = buffer.split("<|endoftext|>")
                for doc in docs[:-1]:
                    if len(sample_docs) < 10:
                        sample_docs.append(doc + "<|endoftext|>")
                buffer = docs[-1]

    total_bytes = 0
    total_tokens = 0
    for doc in sample_docs:
        doc_bytes = len(doc.encode("utf-8"))
        doc_tokens = len(tokenizer.encode(doc))
        total_bytes += doc_bytes
        total_tokens += doc_tokens

    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0

    print(f"采样了 {len(sample_docs)} 个文档。")
    print(f"总字节数: {total_bytes} bytes, 总 Token 数: {total_tokens}")
    print(f"压缩率: {compression_ratio:.2f} bytes/token")

    print("\n--- 2. 估计吞吐量 ---")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_text = f.read(5 * 1024 * 1024)

    test_bytes = len(test_text.encode("utf-8"))

    start_time = time.time()
    _ = tokenizer.encode(test_text)
    end_time = time.time()

    elapsed_time = end_time - start_time
    throughput_bps = test_bytes / elapsed_time
    throughput_mbps = throughput_bps / (1024 * 1024)

    print(f"处理了 {test_bytes} 字节，耗时 {elapsed_time:.2f} 秒。")
    print(f"吞吐量: {throughput_mbps:.2f} MB/s ({throughput_bps:.2f} bytes/s)")

    pile_size_bytes = 825 * 1024 * 1024 * 1024
    pile_time_seconds = pile_size_bytes / throughput_bps
    print(f"估算处理 The Pile (825GB) 所需时间: {pile_time_seconds / 3600:.2f} 小时")

    print("\n--- 3. 全量数据集编码 ---")
    print(f"开始对 {dataset_path} 进行全量编码，这可能需要一些时间...")

    all_tokens = []

    def file_line_generator(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    start_encode = time.time()

    for token_id in tokenizer.encode_iterable(file_line_generator(dataset_path)):
        all_tokens.append(token_id)

    end_encode = time.time()

    print(f"全量编码完成，耗时 {end_encode - start_encode:.2f} 秒。")
    print(f"共生成 {len(all_tokens)} 个 token。")

    tokens_np = np.array(all_tokens, dtype=np.uint16)
    np.save(output_npy_path, tokens_np)

    print(f"Token 数据已保存至 {output_npy_path}，文件大小: {os.path.getsize(output_npy_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    run_experiments()