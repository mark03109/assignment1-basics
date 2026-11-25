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

class GetModel(ArgumentsLoader):
    def __init__(self, arguments_path):
        super().__init__(arguments_path)
        self.trn_args = self.args.get("train")
        self.context_length = self.trn_args.get("context_length")
        self.batch_size = self.trn_args.get("batch_size")
        self.embd_args = self.args.get("embedding_module")
        self.trnsfmr_args = self.args.get("transformer")
        self.d_model = self.embd_args.get("embedding_dim")
        self.num_layers = self.trnsfmr_args.get("num_layers")
        self.rope_theta = self.trnsfmr_args.get("rope_theta")
        self.num_heads = self.trnsfmr_args.get("num_heads")
        self.vocab_size = self.trnsfmr_args.get("vocab_size")
        self.d_ff = self.trnsfmr_args.get("d_ff")
        self.adamW_args = self.args.get("adamW")
        self.lr = self.adamW_args.get("learning_rate")
        self.weight_decay = self.adamW_args.get("weight_decay")
        self.beta1 = self.adamW_args.get("beta1")
        self.beta2 = self.adamW_args.get("beta2")
        self.eps = self.adamW_args.get("eps")

    def get_ropes_from_files(self, device = None, dtype = None) -> list[RotaryPositionalEmbedding]:
        ropes: list[RotaryPositionalEmbedding] = []
        assert self.d_model % self.num_heads == 0
        d_head = self.d_model // self.num_heads
        for i in range(self.num_layers):
            ropes.append(RotaryPositionalEmbedding(theta = self.rope_theta, d_k = d_head,
                                                    max_seq_len= self.context_length, device = device, dtype = dtype))
        return ropes
    
    def get_tansformerlm_from_files(self, device = None, dtype = None) -> TransformerLM:
        ropes = self.get_ropes_from_files(device, dtype)
        return TransformerLM(self.d_model, self.num_heads, self.d_ff, self.vocab_size, 
                             self.context_length, self.num_layers, device, dtype, ropes)

    def get_get_model_optimizer_from_files(self, device = None, dtype = None):
        transformer_lm = self.get_tansformerlm_from_files(device, dtype)
        adamW = AdamW(transformer_lm.parameters(recurse=True),self.lr, self.weight_decay, (self.beta1, self.beta2), self.eps)
        return transformer_lm, adamW

    def get_transformerlm_from_pt(self, pt_path, device = None, dtype = None) -> TransformerLM:
        obj = torch.load(pt_path, map_location= "cpu", weights_only= True)
        transformer_lm = self.get_tansformerlm_from_files(device, dtype, None)
        transformer_lm.load_state_dict(obj["obj_model"])
        return transformer_lm.to(self.device).eval()

    def get_model_optimizer_from_pt(self, pt_path, device = None, dtype = None):
        transformer_lm = self.get_transformerlm_from_pt(pt_path, device, dtype)
        obj = torch.load(pt_path, map_location="cpu", weights_only=True)
        adamW = AdamW(transformer_lm.parameters(recurse=True),self.lr, self.weight_decay, (self.beta1, self.beta2), self.eps)
        return transformer_lm, adamW

class Train(GetModel):
    def __init__(self, arguments_path):
        super().__init__(arguments_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_steps = self.args["train"]["max_steps"]

    def tokenize_data(self):
        print("tokenizing data")
        mp = np.memmap(self.args.get("data_path"), dtype= np.uint8, mode = 'r')
        text = str(mp, encoding = "utf-8")
        indicies = self.tokenizer.encode(text)
        self.input = torch.Tensor(indicies)
        return indicies

    def train_loop(self, log_file, pt_file):
        self.load_tokenizer()
        self.tokenize_data()
        print("start loop")

        transformer_lm, adamW = self.get_model_optimizer_from_files(self.device)

        with open(log_file, 'w', newline="", encoding="utf-8") as lg:
            writer = csv.writer(lg)
            writer.writerow(["step", "elapsed_seconds", "split", "loss"])
            start = perf_counter()

            for step in range(self.max_steps):
                input_ids, target_ids = MyFunctions.data_loader(self.input, 
                                self.batch_size, self.context_length, self.device)
                adamW.zero_grad(set_to_none=True)
                logits: torch.Tensor = transformer_lm(input_ids)
                loss = MyFunctions.cross_entropy(logits, target_ids.long())
                loss.backward()
                adamW.step()

                Checkpoint.save_checkpoint(transformer_lm, adamW, step, pt_file)
                train_loss = loss.item()
                writer.writerow([step, perf_counter() - start, "train", train_loss])

        return transformer_lm
            
    def continue_training_loop(self, log_file, pt_file):
        self.load_tokenizer()
        self.tokenize_data()
        transformer_lm, adamW = self.get_transformerlm_from_pt("data/run")
        obj = torch.load(pt_file)
        initial_step = obj["step"]

        with open(log_file, "w", encoding= "utf-8") as lg:
            writer = csv.writer()
            start = perf_counter()

            for step in range(initial_step+1, self.max_steps):
                input_ids, target_ids = MyFunctions.data_loader(self.input, 
                self.batch_size, self.context_length, self.device)
                adamW.zero_grad(set_to_none=True)
                logits: torch.Tensor = transformer_lm(input_ids)
                loss = MyFunctions.cross_entropy(logits, target_ids.long())
                loss.backward()
                adamW.step()

                Checkpoint.save_checkpoint(transformer_lm, adamW, step, pt_file)
                train_loss = loss.item()
                writer.writerow([step, perf_counter() - start, "train", train_loss])

if __name__ == "__main__":
    my_train = Train("cs336_basics/arguments.json")
    model = my_train.train_loop()
    
