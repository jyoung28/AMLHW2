import os
import random



"""
Generates the grokking data for a dataset
- all p in p_list are prime
- train_split is a decimal representing a percentage
"""
def gen_grok_data(p_list, train_split, a=True, s=True, d= True): 
    add = []
    subtract = []
    division = []
    for p in p_list:
        for i in range(p+1):
            for j in range(i, p+1):
                if (a):
                    c = (i+j) % p
                    add.append(f"{i} + {j} = {c}")
                    add.append(f"{j} + {i} = {c}")

                if (s):
                    c = (j - i) % p
                    subtract.append(f"{j} - {i} = {c}") # j >= i, original paper did have negatives

                if (d and i % p != 0):
                    division.append(f"{j} / {i} = { j * (i**(p-2))%p}") # works for prime p
                if (d and j % p != 0):
                    division.append(f"{i} / {j} = {i * (j**(p-2))%p}")

    # make the train, val, test split
    dataset = add + subtract + division
    random.shuffle(dataset)
    idx = int(train_split * len(dataset))
    train, test = dataset[:idx], dataset[idx:]

    with open('train.txt', 'w') as f:
        for l in train:
            f.write(f"{l}\n")
    
    with open('test.txt', 'w') as f:
        for l in test:
            f.write(f"{l}\n")
    return train, test