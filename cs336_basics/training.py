from cs336_basics.module import MyFunctions, Checkpoint, RotaryPositionalEmbedding, TransformerLM, AdamW
from cs336_basics.BPEtokenizer import Tokenizer
from tests.adapters import run_train_bpe
import json
import pickle
import numpy as np
import torch

class TrainingLoop():
    def __init__(self, adr_train_arguments: str):
        with open(adr_train_arguments, "r", encoding="utf-8") as f:
            self.args: dict = json.load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_tokenizer(self):
        tknz_args = self.args.get("tokenizer")
        if tknz_args.get("if_trained") is False or None:
            vocab, merges = run_train_bpe(tknz_args.get("train_input_path"), tknz_args.get("vocab_size"),
                            tknz_args.get("special_tokens"))
            pickle.dump(vocab, tknz_args.get("vocab_file_path"))
            pickle.dump(merges, tknz_args.get("merges_file_pth"))
        self.tokenizer = Tokenizer.from_files(tknz_args.get("vocab_file_path"), tknz_args.get("merges_file_path"))

    def tokenize(self):
        mp = np.memmap(self.args.get("data_path"), dtype= np.uint8, mode = 'r')
        text = str(mp, encoding = "utf-8")
        indicies = self.tokenizer.encode(text)
        self.input = torch.Tensor(indicies)

    def train(self):
        self.load_tokenizer()
        self.tokenize()
        trn_args = self.args.get("train")
        context_length = trn_args.get("context_length")
        batch_size = trn_args.get("batch_size")
        self.chunked_input, self.chunked_real = MyFunctions.data_loader(self.input, 
                                batch_size, context_length, self.device)

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
        d_head = d_model // d_head
        for i in range(num_layers):
            ropes.append(RotaryPositionalEmbedding(theta = rope_theta, d_k = d_head, max_seq_len= context_length))

        transformer_lm = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length,
                                        num_layers, self.device, ropes = ropes)
        params = transformer_lm.parameters(recurse= True)
        
        adamW = AdamW(params, adamW_args.get("learning_rate"), adamW_args.get("weight_decay"), 
                      (adamW_args.get("beta1"), adamW_args.get("beta2")), adamW_args.get("eps"))

        for epoch in range(trn_args.get("epochs")):
            logits: torch.Tensor = transformer_lm(self.chunked_input)
            vocab_probs: torch.Tensor = MyFunctions.softmax(logits)
            """
            vocab_probs: (batch_size, context_len, vocab_size)
            self.chunked_real: (batch_size, context_length) -> (batch_size, context_length, vocab_size)
            """ 
            target_ids = self.chunked_real.long()  # [batch_size, context_length]
            out = target_ids.new_zeros((*target_ids.shape, vocab_size))
            out.scatter_(dim=-1, index=target_ids.unsqueeze(-1), value=1)

            loss = (out - vocab_probs).square().mean()
            loss.backward()
            adamW.step()
            # Checkpoint.save_checkpoint()
            
            


            
        
        
        
