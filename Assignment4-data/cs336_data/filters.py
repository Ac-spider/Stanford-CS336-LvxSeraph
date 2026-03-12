import random

import unicodedata
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re
import mmh3

import os
import hashlib
from collections import defaultdict
from typing import List,Set, Tuple, Dict

lang_model = fasttext.load_model("./lid.176.bin")
nsfw_model = fasttext.load_model("./jigsaw_fasttext_bigrams_nsfw_final.bin")
toxic_model = fasttext.load_model("./jigsaw_fasttext_bigrams_hatespeech_final.bin")

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)

    if not encoding:
        encoding = 'utf-8'

    try:
        decoded_html = html_bytes.decode(encoding=encoding)
    except UnicodeDecodeError:
        decoded_html = html_bytes.decode(encoding='utf-8',errors='replace')

    return extract_plain_text(decoded_html)



def identify_language(text: str, model: fasttext.FastText._FastText) -> tuple[str, float]:
    text_clean = text.replace('\n','').replace('\r','')

    predictions = model.predict(text_clean,k=1)

    raw_label = predictions[0][0]
    score = float(predictions[1][0])  #原格式为numpy_array

    lang_id = raw_label.replace('__label__','')

    return lang_id,score


def mask_emails(text: str) -> tuple[str, int]:
    # 匹配常见电子邮箱格式
    # 例如: user@example.com, first.last+tag@sub.domain.org
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    # 匹配常见的美国电话号码格式，包含各种分隔符和可选的国家代码
    # 例如: 123-456-7890, (123) 456-7890, +1 123 456 7890, 123.456.7890
    phone_pattern = r'\b(?:\+?1[\s.-]?)?(?:\([0-9]{3}\)|[0-9]{3})[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}\b'
    return re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)

def mask_ips(text: str) -> tuple[str, int]:
    # 严格匹配 IPv4 地址 (0.0.0.0 到 255.255.255.255)
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    return re.subn(ip_pattern, "|||IP_ADDRESS|||", text)


def classify_nsfw(text: str, model: fasttext.FastText._FastText) -> tuple[str, float]:
    """
    使用 jigsaw_fasttext_nsfw_jigsaw_model.bin 进行预测
    """

    text_clean = text.replace('\n', ' ').replace('\r', ' ')
    predictions = model.predict(text_clean, k=1)

    raw_label = predictions[0][0]
    score = float(predictions[1][0])

    label = raw_label.replace('__label__', '')

    return label, score


def classify_toxic_speech(text: str, model: fasttext.FastText._FastText) -> tuple[str, float]:
    """
    使用 jigsaw_fasttext_hatespeech_jigsaw_model.bin 进行预测
    """
    text_clean = text.replace('\n', ' ').replace('\r', ' ')
    predictions = model.predict(text_clean, k=1)

    raw_label = predictions[0][0]
    score = float(predictions[1][0])

    label = raw_label.replace('__label__', '')

    return label, score

def gopher_quality_filter(text: str) -> bool:
    """
    实现 Gopher 的启发式质量过滤规则。
    如果文本符合所有高质量特征则返回 True，否则返回 False 以将其丢弃。
    """
    words = text.split()
    num_words = len(words)

    # 规则 1：包含少于 50 或多于 100,000 个单词则移除
    if num_words < 50 or num_words > 100000:
        return False

    # 规则 2：平均单词长度在 3 到 10 个字符范围之外则移除
    total_chars = sum(len(w) for w in words)
    mean_word_length = total_chars / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # 规则 3：超过 30% 的行以省略号 ("...") 结尾则移除
    lines = text.splitlines()
    if len(lines)>0:
        ellipsis_lines = sum(1 for i in lines if i.strip().endswith('...'))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    # 规则 4：包含少于 80% 带有至少一个字母字符的单词则移除
    alpha_words = sum(1 for word in words if any(w.isalpha() for w in word))
    if (alpha_words / num_words) < 0.8:
        return False

    return True


def train_quality_classifier(train_file_path: str, output_model_path: str):

    # 训练 fastText 监督分类模型
    model = fasttext.train_supervised(
        input=train_file_path,
        epoch=5,
        lr=0.1,
        wordNgrams=2,  # 使用二元语法
        bucket=200000,
        dim=50
    )

    model.save_model(output_model_path)
    print(f"分类器模型已成功保存至: {output_model_path}")

#train_quality_classifier("train.txt", "quality_classifier.bin")


def classify_quality(text: str, model: fasttext.FastText._FastText) -> tuple[str, float]:

    text_clean = text.replace('\n', ' ').replace('\r', ' ')

    predictions = model.predict(text_clean, k=1)

    raw_label = predictions[0][0]
    score = float(predictions[1][0])

    label = raw_label.replace('__label__', '')

    return label, score


def exact_line_deduplication(input_paths: List[str], output_dir: str) -> None:
    line_counts = defaultdict(int)

    for path in input_paths:
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                h = hashlib.md5(line.encode()).digest()
                line_counts[h] += 1

    os.makedirs(output_dir, exist_ok=True)

    for file_path in input_paths:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir,file_name)
        with open(file_path, 'r',encoding='utf-8') as f:
            with open(output_path, 'w', encoding='utf-8') as w:
                for line in f:
                    h = hashlib.md5(line.encode()).digest()
                    if line_counts[h] == 1:
                        w.write(line)


def normalize_text(text: str) -> str:
    """
    对文本进行规范化：转小写、去除标点符号、规范化空白字符、
    移除重音并应用 NFD Unicode 规范化 。
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    # 移除重音符号
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 规范化空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_ngrams(text: str, n: int) -> Set[str]:
    """将规范化后的文本转换为 n-gram 集合。"""
    words = text.split()
    if len(words) < n:
        return set([" ".join(words)]) if words else set()
    return set(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def compute_minhash(ngrams: Set[str], num_hashes: int) -> List[float]:
    """
    计算文档 n-gram 集合的 MinHash 签名
    使用 mmh3 (MurmurHash3) 附加不同的种子来模拟 k 个独立的随机哈希函数。
    """
    signature = [float('inf')] * num_hashes

    for ngram in ngrams:
        for i in range(num_hashes):
            val = mmh3.hash(ngram,i,signed=False)
            if val < signature[i]:
                signature[i] = val

    return signature

def compute_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """计算两个集合之间的真实 Jaccard 相似度 [cite: 279]。"""
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def minhash_deduplication(
        input_paths: List[str],
        num_hashes: int,
        num_bands: int,
        ngram_size: int,
        output_dir: str,
        jaccard_threshold: float = 0.8
) -> None:
    r = num_hashes // num_bands

    doc_ngrams = {}
    signatures = {}

    for path in input_paths:
        with open(path,'r',encoding='utf-8') as f:
            raw_text = f.read()
        norm_text = normalize_text(raw_text)
        ngrams = get_ngrams(norm_text,ngram_size)
        signature = compute_minhash(ngrams,num_hashes)
        doc_ngrams[path] = ngrams
        signatures[path] = signature

    buckets = defaultdict(list)
    for path,sig in signatures.items():
        for b in range(num_bands):
            band_sig = tuple(sig[b*r:(b+1)*r])
            buckets_key = (b,hash(band_sig))
            buckets[buckets_key].append(path)

    candidates = set()
    for doc_list in buckets.values():
        if len(doc_list) > 1:
            for i in range(len(doc_list)):
                for j in range(i+1,len(doc_list)):
                    u,v = doc_list[i],doc_list[j]
                    if u>v:u,v=v,u
                    candidates.add((u,v))

    parents = {path:path for path in input_paths}

    def find(i):
        if parents[i] == i:
            return i
        parents[i] = find(parents[i])
        return parents[i]

    def union(u,v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parents[root_u] = root_v

    for u,v in candidates:
        sim = compute_jaccard(doc_ngrams[u],doc_ngrams[v])
        if sim > jaccard_threshold:
            union(u,v)

    clusters = defaultdict(list)
    for path in input_paths:
        clusters[find(path)].append(path)

    keepdoc=set()
    for docs in clusters.values():
        keepdoc.add(random.choice(docs))

    os.makedirs(output_dir,exist_ok=True)
    for path in keepdoc:
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir,filename)
        with open(path,'r',encoding='utf-8') as f_in:
            with open(output_path,'w',encoding='utf-8') as f_out:
                f_out.write(f_in.read())














