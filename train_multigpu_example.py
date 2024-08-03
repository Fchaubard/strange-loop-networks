import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb

# Load tokenizer and model from Hugging Face
model_name = "EleutherAI/pythia-12b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure special tokens are added
special_tokens_dict = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<sep>",
    "mask_token": "<mask>"
}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Prompt and target response
prompt_text = "<bos> Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? <eos>"
target_response = "<bos> Let's break this down step by step. Step 1: Understand the problem: Natalia sold clips to 48 friends in April. Then, she sold half as many clips in May. We need to find the total number of clips she sold in April and May. Step 2: Calculate the number of clips sold in May If Natalia sold half as many clips in May as she did in April, that means she sold 48 (clips sold in April) / 2 = 24 clips in May Step 3: Add the number of clips sold in April and May To find the total number of clips sold, we add the number of clips sold in April and May, 48 (clips sold in April) + 24 (clips sold in May) = 72. Natalia sold a total of 72 clips in April and May. ### 72 <eos>"

# Tokenize the prompt and response
inputs = tokenizer(prompt_text, return_tensors='pt')
inputs_full = tokenizer(prompt_text+target_response, return_tensors='pt')

targets = tokenizer(target_response, return_tensors='pt')

# Device configuration

num_gpus = torch.cuda.device_count()
device_map = {i: f'cuda:{i}' for i in range(num_gpus)}

def move_model_to_gpus(model, device_map):
    num_gpus = torch.cuda.device_count()
    layers = model.gpt_neox.layers
    num_layers = len(layers)
    layers_per_gpu = num_layers // (num_gpus-1)
    remainder = num_layers % (num_gpus-1)

    current_gpu = 0
    for i, layer in enumerate(layers):
        layer.to(device_map[current_gpu])
        if (i + 1) % layers_per_gpu == 0 and current_gpu < num_gpus - 1:
            current_gpu += 1

    model.gpt_neox.embed_in.to(device_map[0])  # Word embedding to GPU 0
    model.gpt_neox.final_layer_norm.to(device_map[num_gpus - 1])  # Final layer norm to last GPU
    model.embed_out.to(device_map[num_gpus-1])  # LM head to last GPU

move_model_to_gpus(model, device_map)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = 1000
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

num_layers = len(model.gpt_neox.layers)
layers_per_gpu = num_layers // (num_gpus-1)

# Training loop
criterion = nn.CrossEntropyLoss()

inputs = {key: val.to(device_map[0]) for key, val in inputs.items()}
inputs_full = {key: val.to(device_map[0]) for key, val in inputs_full.items()}
targets = targets['input_ids'].to(device_map[0])


while True:
    model.train()
    optimizer.zero_grad()

    # Manually handle the forward pass
    hidden_states = inputs_full['input_ids'].to(device_map[0])
    attention_mask = inputs_full['attention_mask'].to(device_map[0])
    
    # Move embeddings to the first GPU
    hidden_states = model.gpt_neox.embed_in(hidden_states)

    # Pass through each layer manually
    for i, layer in enumerate(model.gpt_neox.layers):
        current_gpu = i // layers_per_gpu if i < layers_per_gpu * (num_gpus - 1) else num_gpus - 1
        
        hidden_states = hidden_states.to(device_map[current_gpu])
        attention_mask = attention_mask.to(device_map[current_gpu])
        hidden_states = layer(hidden_states, attention_mask)[0]

    hidden_states = hidden_states.to(device_map[num_gpus - 1])
    hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
    logits = model.embed_out(hidden_states)
    logits = logits.to(device_map[0])

    # Right shift the targets for autoregressive prediction
    shifted_targets = torch.zeros_like(inputs_full['input_ids'])
    shifted_targets[..., :-1] = inputs_full['input_ids'][..., 1:]
    shifted_targets[..., -1] = tokenizer.eos_token_id

    # Compute the loss
    loss = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1))

    # Backward pass
    loss.backward()

    optimizer.step()
    scheduler.step()

    # Calculate metrics
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == shifted_targets).float()
    accuracy = correct.sum() / correct.numel()
    perplexity = torch.exp(loss)

    # Logging
    print(f" Loss: {loss.item()}", end='')
    print(f" Accuracy: {accuracy.item() * 100:.2f}%", end='')
    print(f" Perplexity: {perplexity.item()}", end='')
    print(f" Learning Rate: {scheduler.get_last_lr()[0]}", end='')
    # pdb.set_trace()
    print(f" Prediction: {tokenizer.decode(predictions[0], skip_special_tokens=False)}")
    print("")
    print("-----"*50)

# Note: The while True loop will run indefinitely. Manually stop the loop as needed.
