import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT
from transformers import AutoTokenizer
import transformers
from dataclasses import dataclass
from tqdm import tqdm


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
def trainGPT(config, tokenizer, X, Xtest=None, batch_size=128, epochs=50, lr=1e-3, weight_decay=1e-2, betas=[0.9, 0.999], device='cpu', grokking=False, max_steps=None, filename=""):
    model = GPT(config)
    model.to(device)

    X = tokenizer(X, padding=True, truncation=True,return_tensors="pt")
    Xtrain = X['input_ids'].to(device)
    Xmasks = X['attention_mask'].to(device)
    
    train_labels = Xtrain.clone()
    train_labels[:, :-1] = Xtrain[:, 1:]  # Shift left
    train_labels[:, -1] = -100 
    
    if (grokking):
        for idx in range(len(train_labels)):
            i = (train_labels[idx] == 100).nonzero(as_tuple=True)[0].item()
            # Mask everything up to and including that index
            train_labels[idx, :i+2] = -100

    train_labels[Xtrain == tokenizer.pad_token_id] = -100 

    test_labels = None
    if (Xtest is not None):
        Xtest_tokenized = tokenizer(Xtest, padding=True, truncation=True,return_tensors="pt")
        Xtest = Xtest_tokenized['input_ids'].to(device)
        Xtestmasks = Xtest_tokenized['attention_mask'].to(device)

        test_labels = Xtest.clone()
        test_labels[:, :-1] = Xtest[:, 1:]  # Shift left
        test_labels[:, -1] = -100        # Ignore last token
        # grokking, ignore all output for the loss except past the  = sign
        if (grokking):
            for idx in range(len(test_labels)):
                #train_labels[idx, :mask_idxs[idx] + 1] = -100 # mask everything but the output
                i = (test_labels[idx] == 100).nonzero(as_tuple=True)[0].item()
                # Mask everything up to and including that index
                test_labels[idx, :i+2] = -100
        test_labels[Xtest == tokenizer.pad_token_id] = -100
        

    optimizer = model.configure_optimizers(weight_decay, lr, betas, device)
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    steps = 0
    path = 'GPT_weights' + filename + '.pth'
    train_accuracies = []
    test_accuracies = []
    all_train_loss = []
    all_valid_loss = []

    progress_bar = tqdm(total=max_steps, desc="Training Progress", dynamic_ncols=True)
    for e in range(epochs):
        avg_loss = 0
        all_predictions = []
        all_labels = []

        indices = torch.randperm(Xtrain.shape[0])
        Xtrain_shuffled = Xtrain[indices]
        Xmasks_shuffled = Xmasks[indices]
        train_labels_shuffled = train_labels[indices]
        # batch_iterator = tqdm(range(num_batches), desc=f"Training (epoch {e+1})", leave=False)
        for i in range(num_batches):
            xb = Xtrain_shuffled[i*batch_size:(i+1)*batch_size].to(device)
            xmb = Xmasks_shuffled[i*batch_size:(i+1)*batch_size].to(device)
            labels = train_labels_shuffled[i*batch_size:(i+1)*batch_size].to(device)

            model.train()
            logits = model(xb, xmb)
            B, T, V = logits.shape

            predictions = torch.argmax(logits, dim = -1)
            all_predictions += [predictions]
            all_labels += [labels]

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

            avg_loss_so_far = avg_loss / (i + 1)
            # progress_bar.update(1)
            # progress_bar.set_postfix_str(f"loss={avg_loss_so_far} | step={steps}")

            if (max_steps is not None and steps >= max_steps):
                print("Reached Max Optimizer Steps")
                torch.save(model.state_dict(), path)
                return model, all_train_loss, all_valid_loss, train_accuracies, test_accuracies

            
            # print(steps)
            

        #compute train accuracy
        train_step_1 = None
        all_predictions = torch.cat(all_predictions, dim = 0)
        all_labels = torch.cat(all_labels, dim = 0)
        train_accuracy = compute_accuracy(all_predictions, all_labels, grokking=grokking)
        
        train_accuracies.append(train_accuracy)
        all_train_loss.append(avg_loss/ num_batches)

        # evaluate at every epoch on test
        if (Xtest is not None and len(test_labels) > 0):
            validation_loss, validation_accuracy = evaluate(model, Xtest, Xtestmasks, test_labels, test_batch_size = 8192, device=device, grokking=grokking)
            if (validation_accuracy >= 1.0):
                print("Validation Accuracy reached 1.0")
                torch.save(model.state_dict(), path)
                return model, all_train_loss, all_valid_loss, train_accuracies, test_accuracies

            test_accuracies.append(validation_accuracy)
            all_valid_loss.append(validation_loss)
            progress_bar.update(num_batches)
            progress_bar.set_postfix_str(
            f"loss={avg_loss / num_batches}, train_acc={train_accuracy}, "
            f"val_loss={validation_loss}, val_acc={validation_accuracy}, step={steps}")



    torch.save(model.state_dict(), path)
    return model, all_train_loss, all_valid_loss, train_accuracies, test_accuracies

def evaluate(model,  Xtest, Xtestmasks, test_labels, test_batch_size=128, device='cpu', grokking=False):
    model.eval()
    with torch.no_grad():
        num_batches = math.ceil(Xtest.shape[0]/test_batch_size)
        total_loss = 0

        xb = Xtest.to(device)
        xmb = Xtestmasks.to(device)
        labels = test_labels.to(device)

        logits = model(xb, xmb)
        B, T, V = logits.shape
        
        loss = F.cross_entropy(
            logits.view(-1, V), 
            labels.view(-1), 
            ignore_index=-100
        )

        accuracy = compute_accuracy(torch.argmax(logits, dim = -1), labels, grokking=grokking)

        total_loss += loss.item()
        return total_loss/num_batches, accuracy
        #print(f"Validation loss : {total_loss/ num_batches}")


def compute_accuracy(predictions, labels, grokking) -> float:
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum()
    if (grokking):
        correct_per_seq = correct.sum(dim=1) 
        mask_per_seq = mask.sum(dim=1)        
        perfect_sequences = (correct_per_seq == mask_per_seq).float()
        return perfect_sequences.mean().item()
    return accuracy.item()


def inference(model, tokenizer, input_str, max_new_tokens=50, device='cpu'):
    with torch.no_grad():
        model.to(device)
        model.eval()

        # Tokenize input
        inputs = tokenizer(input_str, return_tensors='pt', add_special_tokens=False)
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


class NumberLevelTokenizer:
    def __init__(self, p=113, block_size=1024):
        self.p = p
        self.block_size = block_size
        self.special_tokens = ['+', '-', '=', ' ', '/', '<PAD>']
        
        self.token2id = {}
        for i in range(self.p + 1):
            self.token2id[str(i)] = len(self.token2id)
        for tok in self.special_tokens:
            self.token2id[tok] = len(self.token2id)
        
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.pad_token = '<PAD>'
        self.pad_token_id = self.token2id[self.pad_token]

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            if text[i].isdigit():
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                tokens.append(text[i:j])
                i = j
            elif text[i] in self.special_tokens:
                tokens.append(text[i])
                i += 1
            else:
                raise ValueError(f"Unexpected character '{text[i]}' in input.")
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token2id[token] for token in tokens]

    def __call__(self, batch, padding=True, truncation=True, return_tensors=None):
        tokenized = [self.tokenize(example) for example in batch]
        input_ids = [self.convert_tokens_to_ids(tokens) for tokens in tokenized]

        max_len = max(len(ids) for ids in input_ids)
        # if truncation and max_len > self.block_size:
        #     input_ids = [ids[:self.block_size] for ids in input_ids]
        #     max_len = self.block_size

        if padding:
            input_ids = [
                ids + [self.pad_token_id] * (max_len - len(ids))
                for ids in input_ids
            ]

        attention_mask = [
            [1 if token_id != self.pad_token_id else 0 for token_id in ids]
            for ids in input_ids
        ]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        

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