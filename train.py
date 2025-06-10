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
def trainGPT(config, tokenizer, X, batch_size=128, epochs=50, lr=1e-3, weight_decay=1e-2, betas=[0.9, 0.999], device='cpu'):
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
            xb = Xtrain[i*batch_size:(i+1)*batch_size].to(device)
            xmb = Xmasks[i*batch_size:(i+1)*batch_size].to(device)

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
    return model


def inference(model, tokenizer, input_str, max_new_tokens=50, device='cpu'):
    with torch.no_grad():
        model.to(device)
        model.eval()

        # Tokenize input
        inputs = tokenizer(input_str, return_tensors='pt', add_special_tokens=False)
        print(inputs['input_ids'])
        print(tokenizer.encode(tokenizer.eos_token))
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        generated = input_ids

        for _ in range(max_new_tokens):
            # Trim to block size if needed
            if generated.size(1) > model.config.block_size:
                generated = generated[:, -model.config.block_size:]
                attention_mask = attention_mask[:, -model.config.block_size:]

            # Get logits from model
            logits = model(generated, attention_mask)
            logits = logits[:, -1, :]  # Only take the last time step

            # Get predicted next token (greedy)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # print(tokenizer.decode([next_token.item()], skip_special_tokens=True))

            # Append to generated sequence
            generated = torch.cat((generated, next_token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)

            # Optionally stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode output
    output_str = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_str






config = SanityConfig()
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# config.vocab_size = tokenizer.vocab_size
tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')

tokenizer.pad_token = tokenizer.eos_token 
# special_tokens_dict = {'additional_special_tokens': ['<|startoftext|>']}
# tokenizer.add_special_tokens(special_tokens_dict)
print("Starting Training... ")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = trainGPT(config, tokenizer, ["I love machine learning"], device=device)
print(inference(model, tokenizer, ["I "]))