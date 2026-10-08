import pickle

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath, "rb") as f1:
            vocab = pickle.load()
        with open(merges_filepath, "rb") as f2:
            merges = pickle.load()
        return cls(vocab, merges, special_tokens)

    def encode(self, test: str) -> list[int]:
        