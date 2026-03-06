import time
import tracemalloc
import pickle
import os
import cProfile
import pstats
from bpe import train_bpe
from datasets import load_dataset

def download_dataset(dataset_name, hf_path, output_txt_path, text_field="text", split="train"):
    if os.path.exists(output_txt_path):
        print(f"[{dataset_name}] 本地文件已存在，跳过下载: {output_txt_path}")
        return

    print(f"[{dataset_name}] 开始从 Hugging Face 下载: {hf_path} ...")
    dataset = load_dataset(hf_path, split=split)

    print(f"[{dataset_name}] 下载完成，正在写入 {output_txt_path} ...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item[text_field])
            f.write("\n<|endoftext|>\n")

    print(f"[{dataset_name}] 文件写入完成: {output_txt_path}")


def run_experiment(dataset_name, input_path, vocab_size, special_tokens):
    print(f"\n========== 开始 {dataset_name} 的 BPE 训练 ==========")

    tracemalloc.start()
    start_time = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_hours = (end_time - start_time) / 3600
    peak_memory_gb = peak / (1024 ** 3)
    print(f"训练耗时: {elapsed_hours:.4f} 小时")
    print(f"峰值内存使用: {peak_memory_gb:.4f} GB")

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{dataset_name}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"outputs/{dataset_name}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print(f"词表和合并规则已保存至 outputs/ 目录。")

    longest_token_id = max(vocab.keys(), key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    print(f"最长的 Token (ID: {longest_token_id}), 长度: {len(longest_token_bytes)} bytes")
    print(f"字节内容: {longest_token_bytes}")
    try:
        print(f"解码后文本: {longest_token_bytes.decode('utf-8')}")
    except UnicodeDecodeError:
        print("注意: 该 token 包含无法直接解码为 UTF-8 的字节。")


def profile_code(input_path, vocab_size, special_tokens):
    print("\n========== 开始性能分析 (Profiling) ==========")
    profiler = cProfile.Profile()
    profiler.enable()
    train_bpe(input_path, vocab_size, special_tokens)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('tottime')
    stats.print_stats(15)


if __name__ == "__main__":
    tinystories_path = "TinyStories.txt"


    download_dataset(
        dataset_name="TinyStories",
        hf_path="roneneldan/TinyStories",
        output_txt_path=tinystories_path,
        text_field="text",
        split="train"
    )


    if os.path.exists(tinystories_path):
        run_experiment("TinyStories", tinystories_path, 10000, ["<|endoftext|>"])
        profile_code(tinystories_path, 1000, ["<|endoftext|>"])
    else:
        print(f"未找到数据集: {tinystories_path}")
