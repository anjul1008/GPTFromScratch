# imports
import torch
import torch.nn as nn
from gpt2 import GPT2
from utils import   generate_text_simple, text_to_token_ids, \
                    token_ids_to_text, create_dataloader, \
                    calc_loss_batch, evaluate_model, \
                    generate_and_print_samples

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
# device = torch.device('cpu')
print('default device:', device)

# tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')

# reduced orignal GPT2 context_len 1024 to 256, to fit in my laptop memeory
my_model_conf = {
    'vocab_size': 50257,
    'context_len': 256, #800, #256,   #1024
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': False,
}
model = GPT2(my_model_conf)
model.to(device)
model.eval()
print(model)

# ### Download small datasets
import os
import urllib.request

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

ratio = 0.9
split_idx = int(len(text_data.split()) * ratio)
train_data, val_data = text_data.split()[:split_idx], text_data.split()[split_idx:]
# print(len(train_data), len(val_data))

train_data = [' '.join(train_data)] # encapsulated in list since it's a single sentence text
val_data = [' '.join(val_data)]

# pytorch dataloader
train_dataloader = create_dataloader(train_data, 
                                     tokenizer, 
                                     batch_size = 2, 
                                     max_len = my_model_conf['context_len'], 
                                     stride = 1, 
                                     shuffle = True, 
                                     drop_last=True, 
                                     num_workers = 0)

val_dataloader = create_dataloader(val_data, 
                                     tokenizer, 
                                     batch_size = 2, 
                                     max_len = my_model_conf['context_len'], 
                                     stride = 1, 
                                     shuffle = False, 
                                     drop_last=True, 
                                     num_workers = 0)

def train_model_simple(model, train_dataloader, val_dataloader,
                       optimizer, device, num_epoch, eval_freq,
                       eval_iter, start_context, tokenizer):
    print("Training Started.")
    train_losses, val_losses, track_token_seen =[], [], []
    token_seen, global_setp = 0, -1
    best_train_loss = 100
    best_val_loss = 100
    
    for epoch in range(num_epoch):
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            token_seen += input_batch.numel()
            global_setp += 1
            
            if global_setp % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter)
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    print('Saving best train loss model')
                    torch.save({
                                'model': model,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'epoch': epoch+1,
                                'step': global_setp
                                }, 
                               f'best_train.pt')  

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print('Saving best val loss model')
                    torch.save({
                                'model': model,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'epoch': epoch+1,
                                'step': global_setp
                                }, 
                               f'best_eval.pt')  
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                
                # grads = torch.tensor([0])
                # for param in model.parameters():
                #     grads = torch.cat((grads, param.grad)).flatten()
                # print(grads.shape)
                
                print(f"Epoch {epoch+1} (step {global_setp:06d}) : "
                      f"Train Loss:{train_loss:.3f}, "
                      f"Val Loss:{val_loss:.3f}",)
                    #   f"mean gred:{grads.mean()}",
                    #   f"max gred:{grads.max()},",
                    #   f"min gred:{grads.min()}")
                
        
                generate_and_print_samples(model, tokenizer, device, start_context)
                
    return train_losses, val_losses, track_token_seen


# model training
LR = 0.0004
MAX_EPOCHS = 20

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.3)
train_model_simple(model, train_dataloader, val_dataloader, 
                   optimizer, device, MAX_EPOCHS, 
                   eval_freq=100, eval_iter=20, 
                   start_context="Every Effort moves you", 
                   tokenizer=tokenizer)

