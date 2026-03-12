import random
import gzip
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from filters import (
    extract_text_from_html_bytes,
    identify_language,
    gopher_quality_filter,
)

lang_model = fasttext.load_model("./lid.176.bin")

def clean_text_for_fasttext(text: str) -> str:
    """清理文本中的换行符，fastText 要求单行输入"""
    return text.replace('\n', ' ').replace('\r', ' ').strip()


def build_dataset(
        positive_warc_path: str,
        negative_wet_path: str,
        output_path: str,
        max_samples_per_class: int = 5000
):
    dataset = []

    #处理 WIKI 上的 URL
    print("正在处理正样本...")
    pos_count = 0
    total_scanned_pos = 0
    with open(positive_warc_path, 'rb') as f:
        # 正样本从 HTTP 响应中提取 HTML
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if pos_count >= max_samples_per_class:
                break

            total_scanned_pos += 1
            if total_scanned_pos % 100 == 0:
                print(f"已扫描 {total_scanned_pos} 条网页，当前成功提取正样本: {pos_count} 条")

            try:
                html_bytes = record.reader.read()  #record.reader

                if len(html_bytes) > 30 * 1024 * 1024:
                    print(f"跳过过大网页，第 {total_scanned_pos} 条，大小: {len(html_bytes)} bytes")
                    continue

                text = extract_text_from_html_bytes(html_bytes)

                if not text or len(text.split()) < 50:
                    continue

                lang, score = identify_language(text, lang_model)
                if lang == 'en' and score > 0.5 and gopher_quality_filter(text):
                    clean_text = clean_text_for_fasttext(text)
                    if clean_text:
                        dataset.append(f"__label__high{clean_text}\n")
                        pos_count += 1
            except Exception as e:
                print(f"警告：处理WIKI第 {total_scanned_pos} 条数据时发生内部错误 -> {type(e).__name__}: {e}")
                continue

    #处理 OPENCLAW 上的 URL
    print("正在处理负样本...")
    neg_count = 0
    total_scanned_pos = 0
    with open(negative_wet_path, 'rb') as f:

        for record in ArchiveIterator(f, record_types=WarcRecordType.conversion):
            if neg_count >= max_samples_per_class:
                break

            total_scanned_pos += 1
            if total_scanned_pos % 100 == 0:
                print(f"已扫描 {total_scanned_pos} 条网页，当前成功提取负样本: {neg_count} 条")

            try:
                text = record.reader.read().decode('utf-8')
                if not text:
                    continue

                # 同样限制为英文
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
    random.shuffle(dataset)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.writelines(dataset)

    print(f"训练数据已成功保存至 {output_path}")

if __name__ == "__main__":
    # 替换为你实际的负样本 WET 文件路径
    POSITIVE_WARC = "subsampled_positive_urls.warc.gz"
    NEGATIVE_WET = "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
    OUTPUT_FILE = "train.txt"

    build_dataset(POSITIVE_WARC, NEGATIVE_WET, OUTPUT_FILE, max_samples_per_class=5000)













