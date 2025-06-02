import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig
from tokenizers import Tokenizer, pre_tokenizers, models


"""
Train function: allows for lots of hyperparams
"""
def trainGPT(config, X, y, batch_size=128, epochs=10, lr=1e-3, weight_decay=1e-2, betas=[], device='cpu'):
    config = GPTConfig
    model = GPT(config)
    Xtrain  = config.tokenizer.encode(X)
    ytrain = config.tokenizer.encode(y)
    optimizer = model.configure_optimizers(weight_decay, lr, betas, device)
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    
    for e in range(epochs):
        avg_loss = 0
        for i in range(num_batches):
            xb = torch.tensor(Xtrain[i*batch_size:(i+1)*batch_size])
            yb = torch.tensor(ytrain[i*batch_size:(i+1)*batch_size])

            logits = model(xb)
            B, T, V = logits.shape
            logits = logits.reshape(B*T, V)
            truth = yb.reshape(B*T,)
            loss = F.cross_entropy(logits, truth)
            avg_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {e}: Average loss: {avg_loss/ num_batches}")

    torch.save(model.state_dict(), 'GPT_weights.pth')





        
