import pickle
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens
        self.vocab_reverse: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.quick_merge: dict[bytes, list[tuple[bytes, bytes]]] = {}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath, "rb") as f1:
            vocab = pickle.load()
        with open(merges_filepath, "rb") as f2:
            merges = pickle.load()
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        string_iter = re.finditer(PAT, text)
        word_int_dict: dict[str, list[int]] = {}
        ret: list[int] = []
        while True:
            try: 
                temp_word = next(string_iter).group(0)
            except StopIteration:
                break
            if temp_word in word_int_dict:
                ret.extend(iter(word_int_dict[temp_word]))
            else:
                temp_word_byte = temp_word.encode("utf-8")
                temp_list: list[int] = list(temp_word_byte)
                for merge in self.merges:
