import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT
from transformers import AutoTokenizer
import transformers
from dataclasses import dataclass


@dataclass
class SanityConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  


"""
Train function: allows for lots of hyperparams, tokenize before 
"""
def trainGPT(config, tokenizer, X, batch_size=128, epochs=500, lr=1e-3, weight_decay=1e-2, betas=[0.9, 0.999], device='cpu'):
    model = GPT(config)
    model.to(device)
    X = tokenizer(X, padding=True, truncation=True,return_tensors="pt")
    Xtrain = X['input_ids'].to(device)
    Xmasks = X['attention_mask'].to(device)
    optimizer = model.configure_optimizers(weight_decay, lr, betas, device)
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    
    for e in range(epochs):
        avg_loss = 0
        for i in range(num_batches):
            xb = Xtrain[i*batch_size:(i+1)*batch_size]
            xmb = Xmasks[i*batch_size:(i+1)*batch_size]

            labels = xb.clone()
            labels[:, :-1] = xb[:, 1:]  # Shift left
            labels[:, -1] = -100        # Ignore last token
            labels[xmb == tokenizer.pad_token] = -100 
            # labels = labels[:,1:]
            # pad = torch.ones((labels.size(0),1), dtype=labels.dtype, device=labels.device) * -100
            # labels = torch.cat((labels, pad), dim=1)
            # labels[labels==tokenizer.pad_token] = -100

            logits = model(xb, xmb)
            B, T, V = logits.shape
            
            loss = F.cross_entropy(
                logits.view(-1, V), 
                labels.view(-1), 
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        print(f"Epoch {e}: Average loss: {avg_loss/ num_batches}")

    torch.save(model.state_dict(), 'GPT_weights.pth')


def inference(model, input_str):
    model.eval()
    all_predictions=[]





config = SanityConfig()
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# config.vocab_size = tokenizer.vocab_size
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token 
print("Starting Training... ")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainGPT(config, tokenizer, ["I love machine learning"], device=device)
