from cs336_basics.module import MyFunctions, Checkpoint, RotaryPositionalEmbedding, TransformerLM, AdamW, DecodeStrategy
from cs336_basics.BPEtokenizer import Tokenizer
from tests.adapters import run_train_bpe
import json
import pickle
import numpy as np
import torch
import os
import csv
from time import perf_counter

class ArgumentsLoader():
    def __init__(self, arguments_path: str | os.PathLike):
        with open(arguments_path, "r", encoding="utf-8") as f:
            self.args_path = arguments_path
            self.args: dict = json.load(f)

    def load_tokenizer(self):
        print("loading tokenizer")
        tknz_args = self.args.get("tokenizer")

        if tknz_args.get("if_trained") is False or None:
            vocab, merges = run_train_bpe(tknz_args.get("train_input_path"), tknz_args.get("vocab_size"),
                            tknz_args.get("special_tokens"))
            pickle.dump(vocab, tknz_args.get("vocab_file_path"))
            pickle.dump(merges, tknz_args.get("merges_file_pth"))

            self.args["tokenizer"]["if_trained"] = True

            with open(self.args_path, "w", encoding=" utf-8") as f:
                json.dump(self.args, f, ensure_ascii= False, indent=2)
        
        self.tokenizer = Tokenizer.from_files(tknz_args.get("vocab_file_path"), tknz_args.get("merges_file_path"), tknz_args.get("special_tokens"))

class Train(ArgumentsLoader):
    def __init__(self, arguments_path):
        super().__init__(arguments_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tokenize_data(self):
        print("tokenizing data")
        mp = np.memmap(self.args.get("data_path"), dtype= np.uint8, mode = 'r')
        text = str(mp, encoding = "utf-8")
        indicies = self.tokenizer.encode(text)
        self.input = torch.Tensor(indicies)
        return indicies

    def train_loop(self):
        self.load_tokenizer()
        self.tokenize_data()
        print("start loop")

        trn_args = self.args.get("train")
        context_length = trn_args.get("context_length")
        batch_size = trn_args.get("batch_size")

        embd_args = self.args.get("embedding_module")
        trnsfmr_args = self.args.get("transformer")

        ropes: list[RotaryPositionalEmbedding] = []
        d_model = embd_args.get("embedding_dim")
        num_layers = trnsfmr_args.get("num_layers")
        rope_theta = trnsfmr_args.get("rope_theta")
        num_heads = trnsfmr_args.get("num_heads")
        vocab_size = trnsfmr_args.get("vocab_size")
        d_ff = trnsfmr_args.get("d_ff")
        adamW_args = self.args.get("adamW")
        
        assert d_model % num_heads == 0
        d_head = d_model // num_heads
        for i in range(num_layers):
            ropes.append(RotaryPositionalEmbedding(theta = rope_theta, d_k = d_head,
                                                    max_seq_len= context_length, device = self.device))

        transformer_lm = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length,
                                        num_layers, self.device, ropes = ropes)
        transformer_lm.to(self.device)
        params = transformer_lm.parameters(recurse= True)
        
        adamW = AdamW(params, adamW_args.get("learning_rate"), adamW_args.get("weight_decay"), 
                      (adamW_args.get("beta1"), adamW_args.get("beta2")), adamW_args.get("eps"))

        with open("log/run_002.csv", 'w', newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["step", "elapsed_seconds", "split", "loss"])
            start = perf_counter()

            for step in range(trn_args.get("max_steps")):
                input_ids, target_ids = MyFunctions.data_loader(self.input, 
                                batch_size, context_length, self.device)
                adamW.zero_grad(set_to_none=True)
                logits: torch.Tensor = transformer_lm(input_ids)
                loss = MyFunctions.cross_entropy(logits, target_ids.long())
                loss.backward()
                adamW.step()

                Checkpoint.save_checkpoint(transformer_lm, adamW, step, "data/run_002.pt")
                train_loss = loss.item()
                writer.writerow([step, perf_counter() - start, "train", train_loss])

        return transformer_lm
            
class Generator(ArgumentsLoader):
    def __init__(self, arguments_path, type: str):
        super().__init__(arguments_path)
        if type == "greedy":
            self.func = DecodeStrategy.greedy
        elif type == "sample_p":
            self.func = DecodeStrategy.sample_p
        else:
            self.func = DecodeStrategy.sample_p
        self.load_tokenizer()
        
    def decode_loop(self, model: TransformerLM, prompt: str):
        input_tokens = self.tokenizer.encode(prompt)
        input_tokens_len = len(input_tokens)
        while True:
            logits = model(torch.Tensor(input_tokens))
            gntr_args = self.args["generator"]
            next_token = self.func(logits, gntr_args["temperature"], gntr_args["probability"])
            if next_token in self.tokenizer.special_tokens:
                break
            input_tokens.append(next_token)
        return self.tokenizer.decode(input_tokens[input_tokens_len:])

if __name__ == "__main__":
    my_train = Train("cs336_basics/arguments.json")
    model = my_train.train_loop()
    
