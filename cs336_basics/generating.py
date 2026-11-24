from cs336_basics.training import ArgumentsLoader
from cs336_basics.module import DecodeStrategy, TransformerLM, RotaryPositionalEmbedding
import torch
import json

class Generator(ArgumentsLoader):
    def __init__(self, arguments_path, type: str | None = None):
        super().__init__(arguments_path)
        if type == "greedy":
            self.func = DecodeStrategy.greedy
        elif type == "sample_p":
            self.func = DecodeStrategy.sample_p
        else:
            self.func = DecodeStrategy.sample_p
        self.load_tokenizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self, pt_path, arguments_path):
        ropes = []
        num_layers = 1
        with open(arguments_path, "r", encoding="utf-8") as f:
            args = json.load(f)
            num_layers = args["transformer"]["num_layers"]
        for i in range(num_layers):
            ropes.append(RotaryPositionalEmbedding.from_files(arguments_path))
        transformer_lm = TransformerLM.from_files(arguments_path, ropes=ropes)
        obj = torch.load(pt_path, map_location="cpu", weights_only=True)
        transformer_lm.load_state_dict(obj["obj_model"])
        return transformer_lm.to(self.device).eval()

    @torch.inference_mode()
    def decode_loop(self, model: TransformerLM, prompt: str):
        print("start generating")
        input_tokens = self.tokenizer.encode(prompt)
        input_tokens_len = len(input_tokens)
        if not input_tokens:
            raise ValueError("The prompt must encode to at least one token.")
        device = model.embedding.weight.device
        gntr_args = self.args["generator"]
        while len(input_tokens) < model.context_length:
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
            logits = model(input_ids)[0]
            if self.func is DecodeStrategy.greedy:
                next_token = self.func(logits).item()
            else:
                next_token = self.func(logits, gntr_args["temperature"], gntr_args["probability"]).item()
            if next_token in self.tokenizer.special_tokens:
                break
            input_tokens.append(next_token)
        return self.tokenizer.decode(input_tokens[input_tokens_len:])

if __name__ == "__main__":
    generator = Generator("cs336_basics/arguments.json")
    transformer_lm = generator.get_model("data/run_002.pt", "cs336_basics/arguments.json")
    prompt = "Tom said"
    print(generator.decode_loop(transformer_lm, prompt))
