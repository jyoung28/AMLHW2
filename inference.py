import torch
import torch.nn.functional as F
from model import GPT, GPTConfig

def inference(config, model_path, tokenizer, input_str, max_new_tokens=50, device = 'cpu'):
    model = GPT(config)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with torch.no_grad():

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