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


# ─────────────────────────────────────────
# init_worker
# Purpose: 多进程池中每个工作进程的初始化函数，在进程启动时各自独立加载全部模型
# Key concept: ProcessPoolExecutor 使用 fork/spawn 创建子进程，进程间不共享内存；
#              必须在每个子进程内单独加载模型，否则子进程无法访问父进程的模型对象；
#              使用 global 将模型绑定到进程级全局命名空间，避免每次任务调用重复加载
# ─────────────────────────────────────────
def init_worker():
    """每个工作进程启动时初始化的函数，负责加载模型到内存"""
    global lang_model, nsfw_model, toxic_model, quality_model
    # 加载集群上提供的预训练模型
    lang_model = fasttext.load_model("lid.176.bin")                                        # 语言识别
    nsfw_model = fasttext.load_model("jigsaw_fasttext_bigrams_nsfw_final.bin")             # NSFW 分类
    toxic_model = fasttext.load_model("jigsaw_fasttext_bigrams_hatespeech_final.bin")      # 毒性内容分类
    quality_model = fasttext.load_model("quality_classifier.bin")                          # 质量分类器


# ─────────────────────────────────────────
# process_single_wet_file
# Purpose: 处理单个 WET（Web Extracted Text）文件，对每条记录依次执行 5 级过滤，
#          将通过全部过滤的文本写入输出文件，并返回各阶段的留存统计
# Key concept:
#   WET 文件是 Common Crawl 预提取的纯文本格式（WARC conversion 类型记录）；
#   5 级串联过滤流水线（早期过滤先执行，减少后续昂贵模型推理次数）：
#     1. 语言过滤：只保留英文（lang=en，置信度>0.65）
#     2. Gopher 启发式过滤：词数/平均词长/省略号比例/字母词占比
#     3. NSFW 过滤：置信度>0.8 的色情内容丢弃
#     4. 毒性过滤：置信度>0.8 的仇恨言论丢弃
#     5. 质量分类器：质量标签为 low 且置信度>0.7 的文档丢弃
# ─────────────────────────────────────────
def process_single_wet_file(input_path: str, output_path: str) -> dict:
    """处理单个 WET 文件并返回统计数据"""
    # 各阶段留存计数器，用于分析过滤漏斗（funnel）
    stats = {
        "total_records": 0, "lang_passed": 0, "gopher_passed": 0,
        "nsfw_passed": 0, "toxic_passed": 0, "quality_passed": 0, "final_kept": 0
    }

    with open(input_path, 'rb') as stream, open(output_path, 'w', encoding='utf-8') as f_out:
        for record in ArchiveIterator(stream):
            # WET 文件中正文记录类型为 'conversion'（区别于 'warcinfo'/'request'/'response'）
            if record.record_type.name == 'conversion':
                text = record.reader.read().decode('utf-8', errors='replace')
                stats["total_records"] += 1

                # 1. 语言过滤 (仅保留英文)
                # 置信度阈值 0.65 平衡精准率与召回率
                lang, score = identify_language(text, lang_model)
                if lang != 'en' or score < 0.65: continue
                stats["lang_passed"] += 1

                # 2. Gopher 启发式质量过滤
                # 无需模型推理，纯统计规则，速度极快
                if not gopher_quality_filter(text): continue
                stats["gopher_passed"] += 1

                # 3. NSFW 过滤
                # 阈值 0.8：只丢弃高置信度 NSFW，减少误杀
                nsfw_label, nsfw_score = classify_nsfw(text, nsfw_model)
                if nsfw_label == 'nsfw' and nsfw_score > 0.8: continue
                stats["nsfw_passed"] += 1

                # 4. 毒性言论过滤
                # 与 NSFW 过滤逻辑对称，阈值同为 0.8
                toxic_label, toxic_score = classify_toxic_speech(text, toxic_model)
                if toxic_label == 'toxic' and toxic_score > 0.8: continue
                stats["toxic_passed"] += 1

                # 5. 维基百科质量分类器
                # 阈值 0.7：只丢弃高置信度低质量文档
                q_label, q_score = classify_quality(text, quality_model)
                if q_label == 'low' and q_score > 0.7: continue
                stats["quality_passed"] += 1

                # 顺利通过所有检查，保留文本 (使用双换行符分隔文档)
                stats["final_kept"] += 1
                f_out.write(text.strip() + "\n\n")

    return stats


# ─────────────────────────────────────────
# main
# Purpose: 协调整个数据处理管线的三个阶段：
#          阶段1 - 多进程并行过滤（5级流水线）
#          阶段2 - 精确行级去重（MD5哈希）
#          阶段3 - MinHash LSH 近似文档去重
# Key concept:
#   - ProcessPoolExecutor：进程级并行（绕过 GIL），充分利用多核 CPU；
#     initializer=init_worker 确保每个工作进程都有独立的模型副本
#   - concurrent.futures.as_completed：异步收集结果，支持实时进度条
#   - 三阶段去重递进策略：精确行去重去除完全相同行，MinHash去重去除近似重复文档
# ─────────────────────────────────────────
def main():
    # 扫描当前目录下所有 Common Crawl WET 压缩文件
    wet_filepaths = glob.glob("./CC-MAIN-*.warc.wet.gz")
    output_directory_path = "./filtered_wet_outputs/"
    os.makedirs(output_directory_path, exist_ok=True)

    # 获取 CPU 核心数，用于设置进程池大小（IO密集+模型推理，充分利用多核）
    num_cpus = multiprocessing.cpu_count()

    # 汇总所有工作进程的统计数据
    total_stats = {
        "total_records": 0, "lang_passed": 0, "gopher_passed": 0,
        "nsfw_passed": 0, "toxic_passed": 0, "quality_passed": 0, "final_kept": 0
    }

    print(f"开始使用 {num_cpus} 个进程处理 {len(wet_filepaths)} 个 WET 文件...")

    # 创建进程池，initializer 在每个子进程启动时调用一次，加载模型到进程内存
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus,initializer=init_worker) as executor:
        futures = []
        for wet_filepath in wet_filepaths:
            # 去掉 .gz 后缀作为输出文件名（输出为解压后的文本）
            wet_filename = str(pathlib.Path(wet_filepath).name).replace('.gz','.txt')
            output_path = os.path.join(output_directory_path,wet_filename)
            # 提交任务到进程池，立即返回 Future 对象
            futures.append(executor.submit(process_single_wet_file,wet_filepath,output_path))

        # as_completed 按完成顺序迭代，tqdm 显示实时进度条
        for future in tqdm(concurrent.futures.as_completed(futures),total=len(wet_filepaths)):
            file_stats = future.result()
            for k,v in file_stats.items():
                total_stats[k] += v   # 累加各进程的统计数据

    print("\n流水线处理完成！数据过滤留存率：")
    for k, v in total_stats.items():
        print(f"{k}: {v}")


if __name__ == '__main__':

    # ── 阶段 1：多进程并行过滤（语言 + Gopher + NSFW + 毒性 + 质量）──
    print("阶段 1：开始过滤清洗")
    main()

    filtered_files = glob.glob("filtered_wet_outputs/*.txt")
    dedup_step1_dir = "dedup_exact_outputs/"
    dedup_step2_dir = "dedup_final_outputs/"

    # ── 阶段 2：精确行级去重（MD5 哈希，去除完全相同的行）──
    print("阶段 2：exact_line_deduplication")
    exact_line_deduplication(filtered_files, dedup_step1_dir)

    # ── 阶段 3：MinHash LSH 近似文档去重（Jaccard 阈值 0.8）──
    print("阶段 3：开始 MinHash 模糊文档去重")
    step1_files = glob.glob(f"{dedup_step1_dir}/*.txt")

    minhash_deduplication(
        input_paths=step1_files,
        num_hashes=128,          # 签名向量长度，越长估计越准确
        num_bands=16,            # band 数量（128/16=8 rows/band），控制 LSH 灵敏度
        ngram_size=5,            # 5-gram 词级特征，平衡粒度与计算量
        output_dir=dedup_step2_dir
    )

    print("全部数据处理管线执行完毕！最终的训练数据在:", dedup_step2_dir)
