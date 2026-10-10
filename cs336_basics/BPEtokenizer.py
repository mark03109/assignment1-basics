import pickle
import regex as re
from collections.abc import Iterable, Iterator
import os

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens
        self.vocab_reverse: dict[bytes, int] = {v: k for k, v in vocab.items()}
        # self.quick_merge: dict[bytes, list[tuple[bytes, bytes]]] = {}
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath, "rb") as f1:
            vocab = pickle.load(f1)
        with open(merges_filepath, "rb") as f2:
            merges = pickle.load(f2)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        string_iter = re.finditer(PAT, text)
        word_int_dict: dict[str, list[int]] = {}
        indices: list[int] = []
        while True:
            try: 
                temp_word = next(string_iter).group(0)
            except StopIteration:
                break
            if temp_word in word_int_dict:
                indices.extend(word_int_dict[temp_word])
            else:
                if temp_word in self.special_tokens:
                    word_int_dict[temp_word] = [self.vocab[temp_word.encode("utf-8")]]
                    indices.extend(word_int_dict[temp_word])
                else:
                    temp_word_bytes_list: list[bytes] = list(bytes([b]) for b in temp_word.encode("utf-8"))
                    new_bytes_list: list[bytes] = []
                    for merge in self.merges:
                        i = 0
                        while i < len(temp_word_bytes_list):
                            if i + 1 < len(temp_word_bytes_list) and temp_word_bytes_list[i] == merge[0] and temp_word_bytes_list[i+1] == merge[1]:
                                new_bytes_list.append(merge[0] + merge[1])
                                i += 2
                            else:
                                new_bytes_list.append(temp_word_bytes_list[i])
                                i += 1
                        temp_word_bytes_list = new_bytes_list
                        new_bytes_list = []
                    word_indices: list[int] = []
                    for temp_word_bytes in temp_word_bytes_list:
                        word_indices.append(self.vocab_reverse[temp_word_bytes])
                    word_int_dict[temp_word] = word_indices
                    indices.extend(word_int_dict[temp_word])
        return indices
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, ids))
        return b"".join(bytes_list).decode("utf-8")

if __name__ == "__main__":
    # vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
    #           6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    # special_tokens = ["<|endoftext|>"]
    # tokenizer = Tokenizer(vocab, merges, special_tokens)
    # indices = tokenizer.encode("the cat ate")
    # print(indices)
    # tokens = tokenizer.encode_iterable(["the", " cat", " ate"])
    # while True:
    #     try:
    #         print(next(tokens))
    #     except StopIteration:
    #         break
    # print(tokenizer.decode(indices))
    # indices1 = [2, 3, 10, 9, 1, 4, 5, 5, 8, 9, 1, 10]
    # print(tokenizer.decode(indices1))
    tokenizer = Tokenizer.from_files(vocab_filepath="data/TinyStoriesV2-GPT4-train-vocab.pkl", 
                                     merges_filepath="data/TinyStoriesV2-GPT4-train-merges.pkl", 
                                     special_tokens=["<|endoftext|>"])
    fd = open("test.txt")
    text = fd.read(-1)
    tokens = tokenizer.encode(text)
    text_new = tokenizer.decode(tokens)
    print(tokens)
    print(text_new)
    for token in tokens:
        print(f"{token}: {tokenizer.vocab[token]}") 
