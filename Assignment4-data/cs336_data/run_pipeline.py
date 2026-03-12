import os
import glob
import pathlib
import concurrent.futures
import multiprocessing

from tqdm import tqdm
from fastwarc.warc import ArchiveIterator
import fasttext

from filters import (
    identify_language, classify_nsfw, classify_toxic_speech,
    gopher_quality_filter, classify_quality,exact_line_deduplication,minhash_deduplication
)

def init_worker():
    """每个工作进程启动时初始化的函数，负责加载模型到内存"""
    global lang_model, nsfw_model, toxic_model, quality_model
    # 加载集群上提供的预训练模型
    lang_model = fasttext.load_model("lid.176.bin")
    nsfw_model = fasttext.load_model("jigsaw_fasttext_bigrams_nsfw_final.bin")
    toxic_model = fasttext.load_model("jigsaw_fasttext_bigrams_hatespeech_final.bin")
    quality_model = fasttext.load_model("quality_classifier.bin")


def process_single_wet_file(input_path: str, output_path: str) -> dict:
    """处理单个 WET 文件并返回统计数据"""
    stats = {
        "total_records": 0, "lang_passed": 0, "gopher_passed": 0,
        "nsfw_passed": 0, "toxic_passed": 0, "quality_passed": 0, "final_kept": 0
    }

    with open(input_path, 'rb') as stream, open(output_path, 'w', encoding='utf-8') as f_out:
        for record in ArchiveIterator(stream):
            if record.record_type.name == 'conversion':
                text = record.reader.read().decode('utf-8', errors='replace')
                stats["total_records"] += 1

                # 1. 语言过滤 (仅保留英文)
                lang, score = identify_language(text, lang_model)
                if lang != 'en' or score < 0.65: continue
                stats["lang_passed"] += 1

                # 2. Gopher 启发式质量过滤
                if not gopher_quality_filter(text): continue
                stats["gopher_passed"] += 1

                # 3. NSFW 过滤
                nsfw_label, nsfw_score = classify_nsfw(text, nsfw_model)
                if nsfw_label == 'nsfw' and nsfw_score > 0.8: continue
                stats["nsfw_passed"] += 1

                # 4. 毒性言论过滤
                toxic_label, toxic_score = classify_toxic_speech(text, toxic_model)
                if toxic_label == 'toxic' and toxic_score > 0.8: continue
                stats["toxic_passed"] += 1

                # 5. 维基百科质量分类器
                q_label, q_score = classify_quality(text, quality_model)
                if q_label == 'low' and q_score > 0.7: continue
                stats["quality_passed"] += 1

                # 顺利通过所有检查，保留文本 (使用双换行符分隔文档)
                stats["final_kept"] += 1
                f_out.write(text.strip() + "\n\n")

    return stats


def main():
    wet_filepaths = glob.glob("./CC-MAIN-*.warc.wet.gz")
    output_directory_path = "./filtered_wet_outputs/"
    os.makedirs(output_directory_path, exist_ok=True)

    num_cpus = multiprocessing.cpu_count()

    total_stats = {
        "total_records": 0, "lang_passed": 0, "gopher_passed": 0,
        "nsfw_passed": 0, "toxic_passed": 0, "quality_passed": 0, "final_kept": 0
    }

    print(f"开始使用 {num_cpus} 个进程处理 {len(wet_filepaths)} 个 WET 文件...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus,initializer=init_worker) as executor:
        futures = []
        for wet_filepath in wet_filepaths:
            wet_filename = str(pathlib.Path(wet_filepath).name).replace('.gz','.txt')
            output_path = os.path.join(output_directory_path,wet_filename)
            futures.append(executor.submit(process_single_wet_file,wet_filepath,output_path))

        for future in tqdm(concurrent.futures.as_completed(futures),total=len(wet_filepaths)):
            file_stats = future.result()
            for k,v in file_stats.items():
                total_stats[k] += v

    print("\n流水线处理完成！数据过滤留存率：")
    for k, v in total_stats.items():
        print(f"{k}: {v}")


if __name__ == '__main__':

    print("阶段 1：开始过滤清洗")
    main()

    filtered_files = glob.glob("filtered_wet_outputs/*.txt")
    dedup_step1_dir = "dedup_exact_outputs/"
    dedup_step2_dir = "dedup_final_outputs/"

    print("阶段 2：exact_line_deduplication")
    exact_line_deduplication(filtered_files, dedup_step1_dir)

    print("阶段 3：开始 MinHash 模糊文档去重")
    step1_files = glob.glob(f"{dedup_step1_dir}/*.txt")

    minhash_deduplication(
        input_paths=step1_files,
        num_hashes=128,
        num_bands=16,
        ngram_size=5,
        output_dir=dedup_step2_dir
    )

    print("全部数据处理管线执行完毕！最终的训练数据在:", dedup_step2_dir)













