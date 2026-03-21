import random
import gzip
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from filters import (
    extract_text_from_html_bytes,
    identify_language,
    gopher_quality_filter,
)

# ─────────────────────────────────────────
# 模块级语言模型加载
# Purpose: 在脚本启动时加载语言识别模型，供正负样本提取时复用
# Key concept: fastText 语言模型仅加载一次，避免循环内重复 IO 开销
# ─────────────────────────────────────────
lang_model = fasttext.load_model("./lid.176.bin")


# ─────────────────────────────────────────
# clean_text_for_fasttext
# Purpose: 清理文本换行符，将多行文本压缩为单行，满足 fastText 输入格式要求
# Key concept: fastText 训练/推理要求每条样本占一行；
#              换行符会导致文本被截断，产生不完整的训练样本
# ─────────────────────────────────────────
def clean_text_for_fasttext(text: str) -> str:
    """清理文本中的换行符，fastText 要求单行输入"""
    return text.replace('\n', ' ').replace('\r', ' ').strip()


# ─────────────────────────────────────────
# build_dataset
# Purpose: 从 WARC 文件构建 fastText 质量分类器的有监督训练数据集
#          正样本（__label__high）来自 Wikipedia 页面 WARC（高质量百科内容）
#          负样本（__label__low）来自普通 Common Crawl WET 文件（随机爬取内容）
# Key concept:
#   - 正负样本平衡：max_samples_per_class 控制两类样本数量上限，避免类别不平衡
#   - 正样本处理路径：WARC response 记录 -> HTML bytes -> resiliparse 提取纯文本
#     -> 语言过滤(en, >0.5) + Gopher 过滤 -> fastText 格式
#   - 负样本处理路径：WET conversion 记录（已提取纯文本）-> 语言过滤(en, >0.5)
#     -> fastText 格式（不做 Gopher 过滤，保留低质量样本的多样性）
#   - 训练格式：每行 "__label__<class><空格><文本>"，正类为 "high"，负类为 "low"
#   - random.shuffle 打乱样本顺序，防止训练时正负样本成块出现导致梯度偏差
# ─────────────────────────────────────────
def build_dataset(
        positive_warc_path: str,          # Wikipedia 爬取的 WARC 文件路径（正样本来源）
        negative_wet_path: str,           # Common Crawl WET 文件路径（负样本来源）
        output_path: str,                 # 输出训练文件路径
        max_samples_per_class: int = 5000 # 每类最多采集的样本数量
):
    dataset = []   # 存储所有格式化训练样本的列表

    # ── 处理正样本：Wikipedia WARC 文件 ──
    #处理 WIKI 上的 URL
    print("正在处理正样本...")
    pos_count = 0          # 已成功提取的正样本数
    total_scanned_pos = 0  # 已扫描的记录总数（含过滤掉的）
    with open(positive_warc_path, 'rb') as f:
        # WarcRecordType.response：HTTP 响应记录，包含原始 HTML 内容
        # 正样本从 HTTP 响应中提取 HTML
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if pos_count >= max_samples_per_class:
                break   # 达到正样本上限，停止扫描

            total_scanned_pos += 1
            if total_scanned_pos % 100 == 0:
                print(f"已扫描 {total_scanned_pos} 条网页，当前成功提取正样本: {pos_count} 条")

            try:
                html_bytes = record.reader.read()  #record.reader

                # 跳过超大页面（>30MB），防止内存溢出
                if len(html_bytes) > 30 * 1024 * 1024:
                    print(f"跳过过大网页，第 {total_scanned_pos} 条，大小: {len(html_bytes)} bytes")
                    continue

                # resiliparse 从 HTML 字节流提取纯文本（自动处理编码）
                text = extract_text_from_html_bytes(html_bytes)

                # 过滤空文本和极短文本（少于50词）
                if not text or len(text.split()) < 50:
                    continue

                # 正样本额外要求：英文 + Gopher 质量过滤，确保正样本质量可靠
                lang, score = identify_language(text, lang_model)
                if lang == 'en' and score > 0.5 and gopher_quality_filter(text):
                    clean_text = clean_text_for_fasttext(text)
                    if clean_text:
                        # fastText 训练格式：标签与文本直接拼接（无空格分隔符是刻意设计）
                        dataset.append(f"__label__high{clean_text}\n")
                        pos_count += 1
            except Exception as e:
                print(f"警告：处理WIKI第 {total_scanned_pos} 条数据时发生内部错误 -> {type(e).__name__}: {e}")
                continue

    # ── 处理负样本：Common Crawl WET 文件 ──
    #处理 OPENCLAW 上的 URL
    print("正在处理负样本...")
    neg_count = 0          # 已成功提取的负样本数
    total_scanned_pos = 0  # 复用变量名，此处表示负样本扫描计数
    with open(negative_wet_path, 'rb') as f:
        # WarcRecordType.conversion：WET 文件中的纯文本记录，已由 Common Crawl 预提取
        for record in ArchiveIterator(f, record_types=WarcRecordType.conversion):
            if neg_count >= max_samples_per_class:
                break   # 达到负样本上限，停止扫描

            total_scanned_pos += 1
            if total_scanned_pos % 100 == 0:
                print(f"已扫描 {total_scanned_pos} 条网页，当前成功提取负样本: {neg_count} 条")

            try:
                # WET 文件已是纯文本，直接 decode，无需 HTML 解析
                text = record.reader.read().decode('utf-8')
                if not text:
                    continue

                # 同样限制为英文（不做 Gopher 过滤，保留低质量样本作为负例）
                lang, score = identify_language(text, lang_model)
                if lang == 'en' and score > 0.5:
                    clean_text = clean_text_for_fasttext(text)
                    if clean_text:
                        dataset.append(f"__label__low{clean_text}\n")
                        neg_count += 1
            except Exception as e:
                print(f"警告：处理OPENCLAW第 {total_scanned_pos} 条数据时发生内部错误 -> {type(e).__name__}: {e}")
                continue

    print(f"提取完成: {pos_count} 个正样本，{neg_count} 个负样本。")

    # 打乱样本顺序，防止同类样本连续排列影响 SGD 训练效果
    random.shuffle(dataset)

    # 写入训练文件，每行一条样本（fastText 标准训练格式）
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.writelines(dataset)

    print(f"训练数据已成功保存至 {output_path}")


if __name__ == "__main__":
    # 替换为你实际的负样本 WET 文件路径
    POSITIVE_WARC = "subsampled_positive_urls.warc.gz"             # Wikipedia 正样本 WARC
    NEGATIVE_WET = "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"  # CC 负样本 WET
    OUTPUT_FILE = "train.txt"                                       # fastText 训练数据输出路径

    build_dataset(POSITIVE_WARC, NEGATIVE_WET, OUTPUT_FILE, max_samples_per_class=5000)
