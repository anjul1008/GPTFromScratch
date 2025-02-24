import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# GPT like model used to predict next word, 
# so we will prepare data that have next word in prediction to calculate loss
class GPTDataset(Dataset):
    def __init__(self, x, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_seq = [ tokenizer.encode(line) for line in x ]
        
        for token_ids in token_seq:
            for i in range(0, len(token_ids)-max_len, stride):
                self.input_ids.append(torch.tensor(token_ids[i:i+max_len]))
                self.target_ids.append(torch.tensor(token_ids[i+1:i+max_len+1]))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# PyTorch style dataloader
def create_dataloader(input_text, tokenizer, batch_size = 2, max_len = 4, stride = 1, shuffle = False, drop_last=True, num_workers = 0):
    dataset = GPTDataset(input_text, tokenizer=tokenizer, max_len=max_len, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


# Generate text with GPT model
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# Tokenize text into token ids
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.Tensor(encoded).unsqueeze(0).int()
    return encoded_tensor

# Convert token ids to string
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze()
    return tokenizer.decode(flat.tolist())


# Loss calcualation on batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

# Loss calcualation on pytorch dataloader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# model evaluation functions with pytorch dataloader
def evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss  = calc_loss_loader(train_dataloader, model, device, eval_iter)
        val_loss    = calc_loss_loader(val_dataloader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

# function to generate text with GPT model and print the result
def generate_and_print_samples(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.strip())
    model.train()
    
# Decoding Strategy to control randomness
# Temprature Scaling
def softmax_with_temprature(logits, temperature):
    logits = logits / temperature
    softmax_logits = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(softmax_logits, num_samples=1)
    return idx_next

# topk decoding
def topk_decoding(logits, k):
    logits = logits.topk(k, dim=-1)
    idx_next = logits.indices
    return idx_next

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            topk_logits = torch.topk(logits, top_k)
            min_val = topk_logits[:, -1]
            logits = torch.where(
                logits < min_val, #.unsqueeze(1),
                torch.tensor(-float('inf')).to(logits.device),
                logits
            )
            
        if temperature > 0.0:
            # logits = softmax_with_temprature(logits, temperature)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

