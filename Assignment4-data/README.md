# CS336 Assignment 4: Data Engineering

## Assignment Description

> 构建面向大规模语言模型预训练的完整网络爬取数据清洗流水线。
> 原始数据来自 Common Crawl（WARC/WET 格式），经过 HTML 解析、语言过滤、
> Gopher 启发式质量过滤、NSFW/毒性内容过滤、fastText 质量分类器过滤，
> 最终通过精确行去重与 MinHash LSH 近似文档去重，产出可用于 LLM 训练的高质量语料。
> 此外还包含训练质量分类器所需的正负样本数据集构建流程（Wikipedia vs. 随机爬取页面）。

原始作业说明文档：[cs336_spring2025_assignment4_data.pdf](./cs336_spring2025_assignment4_data.pdf)

---

## Tech Stack

| 类别 | 详情 |
|------|------|
| Language | Python 3.10+ |
| HTML 解析 | resiliparse (`extract_plain_text` + `detect_encoding`) |
| 语言检测 | fastText (`lid.176.bin`，176种语言) |
| 内容分类 | fastText NSFW 模型 / 毒性模型 / 自训练质量分类器 |
| 数据格式 | WARC / WET (Common Crawl 标准格式) |
| WARC 解析 | fastwarc (`ArchiveIterator`) |
| 哈希 | hashlib MD5（精确去重）/ mmh3 MurmurHash3（MinHash） |
| 并行计算 | `concurrent.futures.ProcessPoolExecutor` |
| 进度显示 | tqdm |

---

## Core Knowledge Points

1. **Gopher 质量过滤（Gopher Heuristic Filters）**
   DeepMind 在训练 Gopher 模型时提出的启发式规则集：
   词数范围（50~100000）、平均词长（3~10字符）、省略号行比例（<30%）、
   字母词占比（>80%）。无需模型推理，计算速度极快，是数据清洗的第一道屏障。

2. **fastText 分类器（fastText Supervised Classifier）**
   基于词袋 + n-gram 特征的轻量级文本分类器，支持语言识别、NSFW检测、
   毒性检测、质量评估等多任务。训练格式为 `__label__<class> <text>` 每行一条。
   推理速度极快（纯 C++ 后端），适合大规模数据流水线。

3. **MinHash（Minimum Hash Signature）**
   用于估算两个集合 Jaccard 相似度的概率算法。
   核心原理：对于哈希函数 h，Pr[min_h(A) = min_h(B)] = Jaccard(A, B)。
   使用 k 个独立哈希函数（通过不同 MurmurHash3 种子模拟）生成长度为 k 的签名向量，
   签名向量中相同位置相等的比例是 Jaccard 的无偏估计。

4. **LSH Band 技巧（Locality Sensitive Hashing）**
   将 MinHash 签名分为 b 个 band（每 band r 行），同一 band 内签名完全相同的文档对
   被投影到同一 bucket，视为候选相似对。碰撞概率曲线为
   P(碰撞) ≈ 1 - (1 - s^r)^b，通过调整 b 和 r 控制检测阈值的陡峭程度。

5. **Jaccard 相似度（Jaccard Similarity）**
   两集合相似度度量：J(A,B) = |A∩B| / |A∪B|，取值 [0, 1]。
   在去重中作为精确验证步骤，过滤 LSH 产生的假阳性候选对。
   文档用词级 n-gram 集合表示，n=5 时能有效捕捉段落级重复。

6. **Union-Find 并查集（Disjoint Set Union）**
   高效处理动态连通性问题的数据结构。在 MinHash 去重中将相似文档聚合为连通分量（簇），
   路径压缩优化使 find() 近似 O(1)，适合处理大规模候选对合并。
   每个簇随机保留一个代表文档，其余丢弃。

7. **WARC / WET 格式（Web ARChive）**
   Common Crawl 的标准数据格式。WARC 存储原始 HTTP 请求/响应（含 HTML）；
   WET 是 WARC 的衍生格式，已将 HTML 转换为纯文本（conversion 记录类型）。
   本作业同时处理两种格式：WET 用于主清洗流水线，WARC 用于构建正样本训练集。

8. **多进程并行处理（ProcessPoolExecutor）**
   Python `concurrent.futures.ProcessPoolExecutor` 绕过 GIL 实现真正的多核并行。
   `initializer` 参数确保每个子进程在启动时独立加载模型副本，
   `as_completed` 异步收集结果，配合 tqdm 实现实时进度追踪。

---

## Code Structure

```
Assignment4-data/
├── cs336_data/
│   ├── filters.py                  # 核心过滤函数库
│   │   ├── extract_text_from_html_bytes()   # HTML -> 纯文本
│   │   ├── identify_language()              # fastText 语言检测
│   │   ├── mask_emails/phone/ips()          # 正则隐私脱敏
│   │   ├── classify_nsfw()                  # NSFW 内容分类
│   │   ├── classify_toxic_speech()          # 毒性言论分类
│   │   ├── gopher_quality_filter()          # Gopher 启发式质量过滤
│   │   ├── train_quality_classifier()       # 训练 fastText 质量分类器
│   │   ├── classify_quality()               # 质量分类推理
│   │   ├── exact_line_deduplication()       # MD5 精确行去重
│   │   ├── normalize_text()                 # 文本规范化预处理
│   │   ├── get_ngrams()                     # 词级 n-gram 提取
│   │   ├── compute_minhash()                # MinHash 签名计算
│   │   ├── compute_jaccard()                # Jaccard 相似度计算
│   │   └── minhash_deduplication()          # MinHash LSH 近似文档去重
│   │
│   ├── run_pipeline.py             # 主处理流水线
│   │   ├── init_worker()           # 多进程工作进程初始化
│   │   ├── process_single_wet_file()  # 单文件 5 级过滤
│   │   └── main()                  # 并行调度 + 三阶段去重
│   │
│   └── build_quality_dataset.py    # 构建质量分类器训练集
│       ├── clean_text_for_fasttext()   # fastText 格式预处理
│       └── build_dataset()             # 正负样本构建
│
├── cs336-basics/                   # 作业提供的 LM 训练代码（不修改）
└── README.md
```

---

## How to Run

### 环境准备

```bash
pip install resiliparse fasttext fastwarc mmh3 tqdm
```

下载所需模型文件到工作目录：
- `lid.176.bin` — fastText 语言识别模型（官方：https://fasttext.cc/docs/en/language-identification.html）
- `jigsaw_fasttext_bigrams_nsfw_final.bin` — NSFW 分类模型
- `jigsaw_fasttext_bigrams_hatespeech_final.bin` — 毒性内容分类模型
- `quality_classifier.bin` — 质量分类模型（需先运行步骤1训练）

### 步骤 1：构建质量分类器训练集并训练模型

```bash
# 准备正样本 WARC（Wikipedia 页面）和负样本 WET（Common Crawl 随机页面）
python cs336_data/build_quality_dataset.py
# 输出：train.txt（fastText 训练格式）

# 在 Python 中运行（或取消注释 filters.py 底部的调用）：
# from filters import train_quality_classifier
# train_quality_classifier("train.txt", "quality_classifier.bin")
```

### 步骤 2：运行主清洗流水线

```bash
# 将 CC-MAIN-*.warc.wet.gz 文件放到工作目录
cd cs336_data
python run_pipeline.py
```

流水线自动执行三阶段处理：
1. 多进程并行过滤 -> `filtered_wet_outputs/`
2. 精确行去重 -> `dedup_exact_outputs/`
3. MinHash 文档去重 -> `dedup_final_outputs/`（最终训练数据）

---

## Key Results / Observations

- **过滤漏斗**：原始 Common Crawl 数据经过完整流水线后，保留率通常在 5%~15%，
  其中语言过滤（仅保留英文）是最大的数据缩减步骤。
- **Gopher 过滤效果**：平均词长规则对剔除 URL 堆砌、乱码页面效果显著；
  字母词占比规则有效过滤数字/符号密集的表格类低质量页面。
- **MinHash 参数选择**：`num_hashes=128, num_bands=16`（每 band 8 行）
  在 Jaccard=0.8 附近产生陡峭的概率阶跃，误报率和漏报率均控制在合理范围。
- **多进程加速**：相比单进程，`ProcessPoolExecutor` 在多核机器上可获得接近线性的速度提升，
  瓶颈主要在磁盘 IO 和 fastText 模型推理。
- **正负样本质量**：Wikipedia 正样本经过 Gopher 过滤后质量较稳定；
  负样本刻意不做 Gopher 过滤，保留更多低质量样本的多样性，提高分类器的泛化能力。

---

## Notes

- 所有模型文件（`.bin`）需与脚本在同一工作目录，或修改代码中的路径。
- `build_quality_dataset.py` 中 `total_scanned_pos` 在负样本循环中被复用，
  语义为"已扫描负样本记录数"，变量名存在命名歧义（已在注释中说明）。
- MinHash 去重在文档级别操作（每个文件视为一个文档），
  精确行去重在行级别操作，两者互补而非重复。
- 隐私脱敏函数（`mask_emails/phone/ips`）目前未集成到主流水线，
  可按需在 `process_single_wet_file` 中调用。
