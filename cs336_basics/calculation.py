class Calculation:
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

    def calculate_flops(self):
        self.input_seqence_length = self.context_length
        self.norm =  2 * self.input_seqence_length * self.d_model
        self.rope = 2 * 4 * self.context_length * self.d_model / 2 
        self.multihead_self_attention = 2 * 4 * self.input_seqence_length * self.d_model * self.d_model +  self.input_seqence_length * self.d_model * self.d_model + 2 * 2 * self.input_seqence_length * self.input_seqence_length * self.d_model
        fnn = 2 * 3 * self.input_seqence_length * self.d_model * self.d_ff
        linear_embedding = 2 * self.input_seqence_length * self.vocab_size * self.d_model
        FLOPs = self.num_layers * (2 * self.norm + self.rope + self.multihead_self_attention + fnn) + self.norm + linear_embedding

        return FLOPs

    def calculate_percentage_flops(self) -> dict:
        self.input_seqence_length = self.context_length
        self.norm =  2 * self.input_seqence_length * self.d_model
        self.rope = 2 * 4 * self.context_length * self.d_model / 2 
        self.multihead_self_attention = 2 * 4 * self.input_seqence_length * self.d_model * self.d_model +  self.input_seqence_length * self.d_model * self.d_model + 2 * 2 * self.input_seqence_length * self.input_seqence_length * self.d_model
        fnn = 2 * 3 * self.input_seqence_length * self.d_model * self.d_ff
        linear_embedding = 2 * self.input_seqence_length * self.vocab_size * self.d_model
        FLOPs = self.num_layers * (2 * self.norm + self.rope + self.multihead_self_attention + fnn) + self.norm + linear_embedding

        percentage_norm = (self.num_layers*2*self.norm +self.norm)/FLOPs*100
        percentage_rope = (self.num_layers*self.rope)/FLOPs*100
        percentage_multihead_self_attention = (self.num_layers*self.multihead_self_attention)/FLOPs*100
        percentage_fnn = (self.num_layers*fnn)/FLOPs*100
        percentage_linear_embedding = (linear_embedding)/FLOPs*100

        return {
            "norm": percentage_norm,
            "rope": percentage_rope,
            "multihead_self_attention": percentage_multihead_self_attention,
            "fnn": percentage_fnn,
            "linear_embedding": percentage_linear_embedding
        }

if __name__ == "__main__":
    gpt2_small = Calculation(vocab_size=50257, context_length=1024, num_layers=12, d_model=768, num_heads=12, d_ff=4288)
    print("FLOPs for GPT-2 small:",f"{gpt2_small.calculate_flops():,}")
    print("Percentage FLOPs for GPT-2 small:", gpt2_small.calculate_percentage_flops())

    gpt2_medium = Calculation(vocab_size=50257, context_length=1024, num_layers=24, d_model=1024, num_heads=16, d_ff=4288)
    print("FLOPs for GPT-2 medium:", f"{gpt2_medium.calculate_flops():,}")
    print("Percentage FLOPs for GPT-2 medium:", gpt2_medium.calculate_percentage_flops())

    gpt2_large = Calculation(vocab_size=50257, context_length=1024, num_layers=36, d_model=1280, num_heads=20, d_ff=4288)
    print("FLOPs for GPT-2 large:", f"{gpt2_large.calculate_flops():,}")
    print("Percentage FLOPs for GPT-2 large:", gpt2_large.calculate_percentage_flops())

    gpt2_xl_1 = Calculation(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288)
    print("FLOPs for GPT-2 XL-1:", f"{gpt2_xl_1.calculate_flops():,}")
    print("Percentage FLOPs for GPT-2 XL-1:", gpt2_xl_1.calculate_percentage_flops())

    gpt2_xl_2 = Calculation(vocab_size=50257, context_length=16384, num_layers=48, d_model=1600, num_heads=25, d_ff=4288)
    print("FLOPs for GPT-2 XL-2 with context length 16384:", f"{gpt2_xl_2.calculate_flops():,}")
    print("Percentage FLOPs for GPT-2 XL-2 with context length 16384:", gpt2_xl_2.calculate_percentage_flops())