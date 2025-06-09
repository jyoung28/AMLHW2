import torch
from train import trainGPT
from dataclasses import dataclass
from transformers import AutoTokenizer



@dataclass
class GrokConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 1 # can manually change to 2
    n_head: int = 4
    n_embd: int = 128 # the 512 is taken care of in MLP since it is 4 * n_embed
    dropout: float = 0.0
    bias: bool = True 

config = GrokConfig()
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainGPT(config, tokenizer, ["I love machine learning"], device=device)
p_values = [97, 113]

for num_layers in range(2):
    config.n_layer  = num_layers + 1
    for p in p_values:
        for i in range(2):
            # addition or sutraction
            for t in range(3):
                # random restart

    # need to mask the input so only on the output of the equation

