import regex as re
import pickle
from typing import Iterable, Iterator

GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],
                 special_tokens:list[str]|None = None):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        if self.special_tokens:
            existing_bytes = set(self.vocab.values())
            next_id = max(self.vocab.keys())+1 if self.vocab else 0
            for st in self.special_tokens:
                st_bytes = st.encode('utf-8')
                if st_bytes not in existing_bytes:
                    self.vocab[next_id] = st_bytes
                    existing_bytes.add(st_bytes)
                    next_id += 1

        self.inverse_vocab = {k:i for i,k in self.vocab.items()}
        self.merges_rank = {pairs:i for i,pairs in enumerate(self.merges)}

        if self.special_tokens:
            escaped = [re.escape(st) for st in self.special_tokens]
            self.special_pat = re.compile('('+'|'.join(escaped)+')')
        else:
            self.special_pat = None


    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        with open(vocab_filepath,'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges = pickle.load(f)
        return cls(vocab,merges,special_tokens)

    def _encode_chunk(self,text):
        ids = []

        for match in re.finditer(GPT2_PAT,text):
            token_bytes = match.group().encode('utf-8')
            b_list = [bytes([b]) for b in token_bytes]

            while len(b_list) >= 2:
                best_pair = None
                min_rank = float('inf')
                for i in range(len(b_list)-1):
                    pair = (b_list[i],b_list[i+1])
                    rank = self.merges_rank.get(pair,float('inf'))
                    if rank < min_rank:
                        best_pair = pair
                        min_rank = rank

                if not best_pair:
                    break

                i=0
                new_b_list = []
                while i < len(b_list):
                    if i<len(b_list)-1 and b_list[i] == best_pair[0] and b_list[i+1] == best_pair[1]:
                        new_token_bytes = b_list[i]+b_list[i+1]
                        new_b_list.append(new_token_bytes)
                        i+=2
                    else:
                        new_b_list.append(b_list[i])
                        i+=1
                b_list = new_b_list

            for b in b_list:
                ids.append(self.inverse_vocab[b])

        return ids

    def encode(self,text):
        if not self.special_tokens:
            return self._encode_chunk(text)

        ids = []
        parts = self.special_pat.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                part_bytes = part.encode('utf-8')
                ids.append(self.inverse_vocab[part_bytes])
            else:
                ids.extend(self._encode_chunk(part))

        return ids

    def encode_iterable(self,iterable:Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self,ids:list[int]) -> str:
        b_list = []
        for token_id in ids:
            if token_id in self.vocab:
                b_list.append(self.vocab[token_id])
            else:
                raise ValueError(f'Token_id:{token_id}Not Found')

        b_text = b''.join(b_list)
        return b_text.decode(encoding='utf-8',errors='replace')






















