import multiprocessing

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

input_path = "dedup_final_outputs/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.txt"
output_path = "my_tokenized_data.bin"

tokenizer = None

def init_worker():
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_line_and_add_eos(line):
    """对单行文本进行分词，并在末尾追加一个 EOS (End of Sequence) token"""
    if not line.strip():  # 如果是空行，直接返回空列表
        return []
    return tokenizer.encode(line) + [tokenizer.eos_token_id]


def main():
    with open(input_path,'r',encoding='utf-8') as f:
        lines = f.readlines()

    #num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=8, initializer=init_worker)

    results = []
    #chunksize = max(1, len(lines) // (num_cpus * 4))
    chunksize = 10000

    for result in tqdm(
        pool.imap(tokenize_line_and_add_eos,lines,chunksize),
        total = len(lines),
        desc='Tokenizing lines'
    ):
        results.append(result)

    pool.close()
    pool.join()

    all_id = [id for sublist in results for id in sublist]
    print(f'all ids: {len(all_id)}')

    id_array = np.array(all_id,dtype=np.uint16)
    id_array.tofile(output_path)
    print('OK')

if __name__ == '__main__':
    main()











