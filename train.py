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
def trainGPT(config, tokenizer, X, Xtest=None, batch_size=128, epochs=50, lr=1e-3, weight_decay=1e-2, betas=[0.9, 0.999], device='cpu', grokking=False, max_steps=None):
    model = GPT(config)
    model.to(device)
    mask_idxs = []
    test_mask_idxs= []
    
    if (grokking):
        for i in range(len(X)):
            mask_idxs.append(X[i].index('=') + 1) # space after the equal

    X = tokenizer(X, padding=True, truncation=True,return_tensors="pt")
    Xtrain = X['input_ids'].to(device)
    Xmasks = X['attention_mask'].to(device)
    
    train_labels = Xtrain.clone()
    train_labels[:, :-1] = Xtrain[:, 1:]  # Shift left
    train_labels[:, -1] = -100 
    
    if (grokking):
        for idx in range(len(mask_idxs)):
            train_labels[idx, :mask_idxs[idx] + 1] = -100 # mask everything but the output

    train_labels[Xtrain == tokenizer.pad_token] = -100 

    test_labels = None
    if (Xtest is not None):
        if (grokking):
            for i in range(len(Xtest)):
                test_mask_idxs.append(Xtest[i].index('=') + 1)
        Xtest_tokenized = tokenizer(Xtest, padding=True, truncation=True,return_tensors="pt")
        Xtest = Xtest_tokenized['input_ids'].to(device)
        Xtestmasks = Xtest_tokenized['attention_mask'].to(device)

        test_labels = Xtest.clone()
        test_labels[:, :-1] = Xtest[:, 1:]  # Shift left
        test_labels[:, -1] = -100        # Ignore last token

        # grokking, ignore all output for the loss except past the  = sign
        if (len(test_mask_idxs) > 0):
            for idx in range(len(test_mask_idxs)):
                test_labels[idx, :test_mask_idxs[idx] + 1] = -100
        

    optimizer = model.configure_optimizers(weight_decay, lr, betas, device)
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    steps = 0
    
    for e in range(epochs):
        avg_loss = 0
        for i in range(num_batches):
            xb = Xtrain[i*batch_size:(i+1)*batch_size].to(device)
            xmb = Xmasks[i*batch_size:(i+1)*batch_size].to(device)
            labels = train_labels[i*batch_size:(i+1)*batch_size].to(device)

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
            steps += 1
            if (max_steps is not None and steps >= max_steps):
                print("Reached Max Optimizer Steps")
                torch.save(model.state_dict(), 'GPT_weights.pth')
                return model

        if (e % 100 == 0 and Xtest is not None):
                print(f"Epoch {e}: Average loss: {avg_loss/ num_batches}")
                evaluate(model, Xtest, Xtestmasks, test_labels)

    torch.save(model.state_dict(), 'GPT_weights.pth')
    return model

def evaluate(model,  Xtest, Xtestmasks, test_labels, test_batch_size=128, device='cpu'):
    with torch.no_grad():
        model.eval()
        num_batches = math.ceil(Xtest.shape[0]/test_batch_size)
        total_loss = 0
        for i in range(num_batches):
            xb = Xtest[i*test_batch_size:(i+1)*test_batch_size].to(device)
            xmb = Xtestmasks[i*test_batch_size:(i+1)*test_batch_size].to(device)
            labels = test_labels[i*test_batch_size:(i+1)*test_batch_size].to(device)

            logits = model(xb, xmb)
            B, T, V = logits.shape
            
            loss = F.cross_entropy(
                logits.view(-1, V), 
                labels.view(-1), 
                ignore_index=-100
            )

            total_loss += loss.item()
        print(f"Validation loss : {total_loss/ num_batches}")


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






# config = SanityConfig()
# # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# # config.vocab_size = tokenizer.vocab_size
# tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')

# tokenizer.pad_token = tokenizer.eos_token 
# # special_tokens_dict = {'additional_special_tokens': ['<|startoftext|>']}
# # tokenizer.add_special_tokens(special_tokens_dict)
# print("Starting Training... ")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = trainGPT(config, tokenizer, ["I love machine learning"], device=device)
# print(inference(model, tokenizer, ["I "]))