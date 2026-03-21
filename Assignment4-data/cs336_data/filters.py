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

# ─────────────────────────────────────────
# 全局模型加载
# Purpose: 在模块导入时一次性将三个 fastText 模型加载到内存，避免每次调用重复 IO
# Key concept: fastText 二进制模型（.bin）加载后驻留内存，推理极快（纯 C++ 后端）
# ─────────────────────────────────────────
lang_model = fasttext.load_model("./lid.176.bin")                                          # 176 种语言检测模型
nsfw_model = fasttext.load_model("./jigsaw_fasttext_bigrams_nsfw_final.bin")               # NSFW（色情/不雅）内容分类
toxic_model = fasttext.load_model("./jigsaw_fasttext_bigrams_hatespeech_final.bin")        # 仇恨言论/毒性内容分类


# ─────────────────────────────────────────
# extract_text_from_html_bytes
# Purpose: 将原始 HTML 字节流解码并提取纯文本，剥离所有 HTML 标签、脚本、样式
# Key concept: resiliparse 库的 extract_plain_text 采用启发式算法识别正文区域；
#              detect_encoding 自动探测字节流编码（BOM / meta charset / 统计推断）
# ─────────────────────────────────────────
def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    # 自动探测编码，fallback 到 UTF-8
    encoding = detect_encoding(html_bytes)

    if not encoding:
        encoding = 'utf-8'

    try:
        # 用探测到的编码解码原始字节
        decoded_html = html_bytes.decode(encoding=encoding)
    except UnicodeDecodeError:
        # 解码失败时强制 UTF-8 并用替换字符填充无效字节
        decoded_html = html_bytes.decode(encoding='utf-8',errors='replace')

    # resiliparse 提取正文纯文本，自动去除导航栏、广告等噪声块
    return extract_plain_text(decoded_html)


# ─────────────────────────────────────────
# identify_language
# Purpose: 使用 fastText 语言识别模型预测文本所属语言及置信度
# Key concept: fastText 输出标签格式为 "__label__<lang_code>"，如 "__label__en"；
#              k=1 只返回置信度最高的一个预测结果
# ─────────────────────────────────────────
def identify_language(text: str, model: fasttext.FastText._FastText) -> tuple[str, float]:
    # fastText 要求单行输入，去除换行符防止截断
    text_clean = text.replace('\n','').replace('\r','')

    # k=1 返回 top-1 预测；predictions[0] 是标签列表，predictions[1] 是概率列表
    predictions = model.predict(text_clean,k=1)

    raw_label = predictions[0][0]          # 原始标签，如 "__label__en"
    score = float(predictions[1][0])       # 原格式为numpy_array，转 float

    # 去掉前缀，得到 ISO 639-1 语言代码，如 "en"、"zh"
    lang_id = raw_label.replace('__label__','')

    return lang_id,score


# ─────────────────────────────────────────
# mask_emails / mask_phone_numbers / mask_ips
# Purpose: 用占位符替换文本中的个人隐私信息（邮箱、电话、IP），实现数据脱敏
# Key concept: re.subn(pattern, repl, string) 返回 (替换后的文本, 替换次数) 二元组；
#              在数据清洗流水线中，替换次数可作为隐私风险评分的依据
# ─────────────────────────────────────────
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
    # 三段 "0-255." + 最后一段 "0-255"，用词边界 \b 防止匹配更长数字串
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    return re.subn(ip_pattern, "|||IP_ADDRESS|||", text)


# ─────────────────────────────────────────
# classify_nsfw
# Purpose: 判断文本是否包含 NSFW（Not Safe For Work）内容，返回标签与置信度
# Key concept: 使用 Jigsaw 数据集微调的 fastText bigram 分类器；
#              标签为 "nsfw" 或 "not_nsfw"
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# classify_toxic_speech
# Purpose: 判断文本是否包含仇恨言论/毒性内容，返回标签与置信度
# Key concept: 同样基于 Jigsaw 数据集训练；标签为 "toxic" 或 "not_toxic"；
#              与 NSFW 模型并列使用，构成内容安全双重过滤
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# gopher_quality_filter
# Purpose: 复现 DeepMind Gopher 论文中的启发式数据质量过滤规则，
#          快速剔除低质量/垃圾网页文本，无需模型推理
# Key concept: Gopher 规则基于统计特征（词数、平均词长、省略号比例、字母词占比），
#              是大规模数据清洗的第一道高效屏障；
#              所有规则均为"不满足则丢弃"的否定过滤逻辑
# ─────────────────────────────────────────
def gopher_quality_filter(text: str) -> bool:
    """
    实现 Gopher 的启发式质量过滤规则。
    如果文本符合所有高质量特征则返回 True，否则返回 False 以将其丢弃。
    """
    words = text.split()
    num_words = len(words)  # 总词数，用于后续比例计算的分母

    # 规则 1：包含少于 50 或多于 100,000 个单词则移除
    # 太短可能是导航栏/广告，太长可能是拼接的垃圾文本
    if num_words < 50 or num_words > 100000:
        return False

    # 规则 2：平均单词长度在 3 到 10 个字符范围之外则移除
    # 过短 -> 大量单字母符号/缩写；过长 -> 无空格乱码或 URL 堆砌
    total_chars = sum(len(w) for w in words)  # 所有词的字符总数
    mean_word_length = total_chars / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # 规则 3：超过 30% 的行以省略号 ("...") 结尾则移除
    # 省略号密集通常意味着内容被截断或是列表式低质量摘要
    lines = text.splitlines()
    if len(lines)>0:
        ellipsis_lines = sum(1 for i in lines if i.strip().endswith('...'))  # 以省略号结尾的行数
        if ellipsis_lines / len(lines) > 0.3:
            return False

    # 规则 4：包含少于 80% 带有至少一个字母字符的单词则移除
    # 字母词占比低意味着文本充斥数字、符号或乱码
    alpha_words = sum(1 for word in words if any(w.isalpha() for w in word))  # 含字母的词数
    if (alpha_words / num_words) < 0.8:
        return False

    return True


# ─────────────────────────────────────────
# train_quality_classifier
# Purpose: 用 fastText 有监督学习训练文本质量分类器（高质量 vs 低质量）
# Key concept: fastText 监督分类使用词袋 + n-gram 特征；
#              wordNgrams=2 启用 bigram，捕捉短语级语义；
#              训练数据格式为 "__label__<class> <text>" 每行一条
# ─────────────────────────────────────────
def train_quality_classifier(train_file_path: str, output_model_path: str):

    # 训练 fastText 监督分类模型
    model = fasttext.train_supervised(
        input=train_file_path,
        epoch=5,          # 训练轮数，较少轮数防止小数据集过拟合
        lr=0.1,           # 学习率
        wordNgrams=2,     # 使用二元语法（bigram），增强短语语义捕捉
        bucket=200000,    # 哈希桶数量，控制 n-gram 碰撞率
        dim=50            # 词向量维度
    )

    model.save_model(output_model_path)
    print(f"分类器模型已成功保存至: {output_model_path}")

#train_quality_classifier("train.txt", "quality_classifier.bin")


# ─────────────────────────────────────────
# classify_quality
# Purpose: 使用训练好的质量分类器判断文本是高质量（high）还是低质量（low）
# Key concept: 与 gopher_quality_filter 互补——Gopher 规则快速粗筛，
#              质量分类器（基于 Wikipedia 正样本训练）进行细粒度语义评分
# ─────────────────────────────────────────
def classify_quality(text: str, model: fasttext.FastText._FastText) -> tuple[str, float]:

    text_clean = text.replace('\n', ' ').replace('\r', ' ')

    predictions = model.predict(text_clean, k=1)

    raw_label = predictions[0][0]
    score = float(predictions[1][0])

    label = raw_label.replace('__label__', '')

    return label, score


# ─────────────────────────────────────────
# exact_line_deduplication
# Purpose: 跨多个文件的精确行级去重，只保留在全局语料中仅出现一次的行
# Key concept: 使用 MD5 哈希将每行映射到固定长度摘要（16 bytes），
#              以 defaultdict 统计全局出现次数；
#              两遍扫描：第一遍建立计数表，第二遍过滤唯一行
# ─────────────────────────────────────────
def exact_line_deduplication(input_paths: List[str], output_dir: str) -> None:
    line_counts = defaultdict(int)  # MD5摘要 -> 全局出现次数

    # 第一遍：遍历所有文件，统计每行的 MD5 哈希出现次数
    for path in input_paths:
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                h = hashlib.md5(line.encode()).digest()  # 计算行的 MD5 摘要（bytes 类型）
                line_counts[h] += 1

    os.makedirs(output_dir, exist_ok=True)

    # 第二遍：只写出全局出现次数恰好为 1 的行（严格去重）
    for file_path in input_paths:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir,file_name)
        with open(file_path, 'r',encoding='utf-8') as f:
            with open(output_path, 'w', encoding='utf-8') as w:
                for line in f:
                    h = hashlib.md5(line.encode()).digest()
                    if line_counts[h] == 1:    # 仅保留全局唯一行
                        w.write(line)


# ─────────────────────────────────────────
# normalize_text
# Purpose: 文本规范化预处理，消除大小写/重音/标点/多余空白等表面差异，
#          使得语义相同但格式不同的文本能被 MinHash 识别为相似
# Key concept: NFD Unicode 规范化将组合字符拆分（如 é -> e + ́），
#              再通过 unicodedata.combining 过滤掉重音组合符
# ─────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    对文本进行规范化：转小写、去除标点符号、规范化空白字符、
    移除重音并应用 NFD Unicode 规范化 。
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)   # 分解组合字符（如 é -> e + 重音符）
    # 移除重音符号（combining 类别的字符均为附加符号）
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # 移除标点符号（保留字母数字和空白）
    text = re.sub(r'[^\w\s]', '', text)
    # 规范化空白字符（将多个连续空白压缩为单个空格）
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────
# get_ngrams
# Purpose: 将规范化文本切分为词级 n-gram 集合，作为文档指纹的基础特征
# Key concept: n-gram 集合是 Jaccard 相似度和 MinHash 的核心输入；
#              词级（word-level）n-gram 比字符级对词序变化更鲁棒
# ─────────────────────────────────────────
def get_ngrams(text: str, n: int) -> Set[str]:
    """将规范化后的文本转换为 n-gram 集合。"""
    words = text.split()
    # 词数不足 n 时退化为整段文本作为单一 gram，避免空集合
    if len(words) < n:
        return set([" ".join(words)]) if words else set()
    # 滑动窗口生成所有大小为 n 的词组
    return set(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


# ─────────────────────────────────────────
# compute_minhash
# Purpose: 为文档的 n-gram 集合计算 MinHash 签名向量，用于 LSH 近似相似度搜索
# Key concept: MinHash 利用"最小哈希值的期望等于 Jaccard 相似度"这一性质；
#              使用 MurmurHash3（mmh3）的不同种子模拟 num_hashes 个独立随机哈希函数；
#              签名向量长度 = num_hashes，每位存储对应哈希函数作用于所有 n-gram 的最小值
# ─────────────────────────────────────────
def compute_minhash(ngrams: Set[str], num_hashes: int) -> List[float]:
    """
    计算文档 n-gram 集合的 MinHash 签名
    使用 mmh3 (MurmurHash3) 附加不同的种子来模拟 k 个独立的随机哈希函数。
    """
    signature = [float('inf')] * num_hashes  # 初始化为无穷大，逐步取最小值

    for ngram in ngrams:
        for i in range(num_hashes):
            # seed=i 使每个位置使用不同的哈希函数；signed=False 保证无符号整数
            val = mmh3.hash(ngram,i,signed=False)
            if val < signature[i]:         # MinHash：保留每个哈希函数的最小值
                signature[i] = val

    return signature


# ─────────────────────────────────────────
# compute_jaccard
# Purpose: 计算两个 n-gram 集合之间的精确 Jaccard 相似度，用于验证候选对
# Key concept: Jaccard(A,B) = |A∩B| / |A∪B|，取值 [0,1]；
#              1.0 表示完全相同，0.0 表示完全不同；
#              在 MinHash LSH 中作为最终验证，过滤 LSH 产生的假阳性候选对
# ─────────────────────────────────────────
def compute_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """计算两个集合之间的真实 Jaccard 相似度 [cite: 279]。"""
    if not set1 and not set2:
        return 1.0   # 两个空集合视为完全相同
    return len(set1.intersection(set2)) / len(set1.union(set2))


# ─────────────────────────────────────────
# minhash_deduplication
# Purpose: 使用 MinHash + LSH（局部敏感哈希）对文档集合进行近似去重，
#          去除 Jaccard 相似度超过阈值的近重复文档
# Key concept:
#   - LSH Band 技巧：将 num_hashes 个签名值分成 num_bands 个 band，
#     同一 band 内所有值相同的文档对被视为候选对（高概率相似）
#     碰撞概率 ≈ 1-(1-s^r)^b，其中 s=Jaccard，r=每band行数，b=band数
#   - Union-Find（并查集）：高效地将相似文档聚合成连通分量（簇），
#     路径压缩使 find() 近似 O(1)
#   - 每个簇随机保留一个代表文档，其余丢弃
# ─────────────────────────────────────────
def minhash_deduplication(
        input_paths: List[str],
        num_hashes: int,       # 签名向量总长度（哈希函数数量）
        num_bands: int,        # LSH band 数量，越多召回率越高但精度降低
        ngram_size: int,       # n-gram 的 n，控制文档指纹粒度
        output_dir: str,
        jaccard_threshold: float = 0.8   # Jaccard 相似度阈值，超过则视为重复
) -> None:
    r = num_hashes // num_bands   # 每个 band 包含的行数（rows per band）

    doc_ngrams = {}    # path -> n-gram 集合（用于精确 Jaccard 验证）
    signatures = {}    # path -> MinHash 签名向量

    # 第一步：为每个文档计算规范化文本、n-gram 集合和 MinHash 签名
    for path in input_paths:
        with open(path,'r',encoding='utf-8') as f:
            raw_text = f.read()
        norm_text = normalize_text(raw_text)
        ngrams = get_ngrams(norm_text,ngram_size)
        signature = compute_minhash(ngrams,num_hashes)
        doc_ngrams[path] = ngrams
        signatures[path] = signature

    # 第二步：LSH 分 band，将签名投影到 bucket 中，收集候选相似对
    buckets = defaultdict(list)   # (band_id, band_hash) -> 文档路径列表
    for path,sig in signatures.items():
        for b in range(num_bands):
            band_sig = tuple(sig[b*r:(b+1)*r])          # 取第 b 个 band 的签名片段
            buckets_key = (b,hash(band_sig))             # band 编号 + 片段哈希 作为 bucket 键
            buckets[buckets_key].append(path)

    # 从 bucket 中提取所有候选对（同 bucket 内两两配对），去重后存入 set
    candidates = set()
    for doc_list in buckets.values():
        if len(doc_list) > 1:
            for i in range(len(doc_list)):
                for j in range(i+1,len(doc_list)):
                    u,v = doc_list[i],doc_list[j]
                    if u>v:u,v=v,u          # 规范化顺序，避免 (a,b) 和 (b,a) 重复
                    candidates.add((u,v))

    # 第三步：Union-Find（并查集）初始化，每个文档是自己的根节点
    parents = {path:path for path in input_paths}

    def find(i):
        """路径压缩的查找操作，返回 i 所在集合的根节点"""
        if parents[i] == i:
            return i
        parents[i] = find(parents[i])   # 路径压缩：将沿途节点直接指向根
        return parents[i]

    def union(u,v):
        """合并 u 和 v 所在的集合"""
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parents[root_u] = root_v    # 将 u 的根指向 v 的根，完成合并

    # 第四步：对每个候选对计算精确 Jaccard 相似度，超过阈值则合并到同一集合
    for u,v in candidates:
        sim = compute_jaccard(doc_ngrams[u],doc_ngrams[v])
        if sim > jaccard_threshold:
            union(u,v)   # 合并相似文档到同一簇

    # 第五步：按根节点分组，得到所有去重簇
    clusters = defaultdict(list)
    for path in input_paths:
        clusters[find(path)].append(path)

    # 每个簇随机保留一个代表文档（random.choice 保证无偏选择）
    keepdoc=set()
    for docs in clusters.values():
        keepdoc.add(random.choice(docs))

    # 第六步：将保留的文档写入输出目录
    os.makedirs(output_dir,exist_ok=True)
    for path in keepdoc:
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir,filename)
        with open(path,'r',encoding='utf-8') as f_in:
            with open(output_path,'w',encoding='utf-8') as f_out:
                f_out.write(f_in.read())
