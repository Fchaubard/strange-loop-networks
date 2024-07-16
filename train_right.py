# train_right.py 
# - picks up batches from ./batches/*, sorts by time to grab n most recent, and then trains right model on them. 
# - pushes results to wandb, and saves checkpoints to ./checkpoint_right/ with format 'right_checkpoint_timestamp_iter_loss'
# - train with Binary Cross Entropy.
import torch
from transformers import GPTNeoXModel, GPTNeoXForCausalLM, GPTNeoXConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from copy import deepcopy
from typing import Optional, Tuple, Union
from torch import nn, optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import glob
import pdb
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torchvision.ops import sigmoid_focal_loss

import wandb
import os

os.environ["WANDB_API_KEY"] = "cce47709d839921f0b13533529f31c8af7f3f4dc"


if __name__ == '__main__':
 
    #------------------
    # THINGS TO UPDATE:
    #------------------
    # Define the batches_directory file path
    batches_directory = "/sln_batches/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    
    # update what device you want to train this model on
    device = 'cuda:1'#'cpu'#

    # update what model you want to use WARNING: IF YOU WANT TO START FROM A CHECKPOINT OF RIGHT MODEL, THIS IS THE PLACE TO DO IT:
    model_id = "EleutherAI/pythia-410M" #"EleutherAI/pythia-1b" # "EleutherAI/pythia-70m-v0"
    right_model_checkpoint_name = "/right-strange-loop-network-410m/right_checkpoint_20240709113005_iter_2000_loss_0.52.pth" # use this if you want to load from a checkpoint, else will load from pythia pretrained
    
    right_model_directory = "/right_checkpoints/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    
    macro_batch_size = 1 
    pad_tok = '[PAD]'

    # Initialize wandb
    wandb_project_name = "reward_training_pythia_"+model_id.replace("/","_")

    lr = 1e-5
    weight_decay = 0.0001
    betas = (0.99,0.999)

    # for scheduler
    factor=0.5
    patience=200
    cooldown=200

    max_microbatch_size = 20 # IMPORTANT TO INCREASE IF YOU HAVE MORE GPU RAM

    save_checkpoint_every_n_batches = 1000
    
    include_wandb = True
    #------------------
    # DO SETUP:
    #------------------
    if include_wandb:
        wandb.init(project=wandb_project_name)

    # Check if the batches_directory exists
    if not os.path.exists(batches_directory):
        # If the directory does not exist, create it
        print(f"Directory {batches_directory} doesnt exit! Go create some batches to train on.")
        exit()
    else:
        print(f"Directory {batches_directory} exists.")

    # Check if the right_model_directory exists
    if not os.path.exists(right_model_directory):
        # If the directory does not exist, create it
        os.mkdir(right_model_directory)
        print(f"Directory {batches_directory} doesnt exit! Making it.")
    else:
        print(f"Directory {batches_directory} exists.")
    
    # Grab a tokenizer for reward_mask creation
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_tok})
        print('setting tokenizer.pad_token_id to: ' + str(tokenizer.pad_token_id) + " with token: "+pad_tok)
        # tokenizer.padding_side = "left"

    # create initial model 
    if right_model_checkpoint_name:
        print("LOADING MODEL FROM: " +right_model_checkpoint_name)
        # Implement load from a checkpoint
        checkpoint = torch.load(right_model_checkpoint_name)
        current_right_model = AutoModelForCausalLM.from_pretrained(model_id)
        #current_right_model.load_state_dict(checkpoint['model_state_dict'])
        current_right_model.config.pad_token_id = tokenizer.pad_token_id
        current_right_model.resize_token_embeddings(len(tokenizer))
        current_right_model.load_state_dict(checkpoint['model_state_dict'])
        current_right_model = current_right_model.to(device)

        vocab_size = current_right_model.config.vocab_size

        reward_layer = nn.Sequential(
            nn.LayerNorm(vocab_size, current_right_model.config.layer_norm_eps),
            nn.Linear(vocab_size, 2),
            nn.Softmax(dim=-1)
        ).to(device)
        reward_layer.load_state_dict(checkpoint['reward_layer_state_dict'])
        reward_layer = reward_layer.to(device)

        optimizer_right = optim.AdamW(list(current_right_model.parameters()) + list(reward_layer.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        optimizer_right.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # load from hf default
        print("STARTING WEIGHTS FROM DEFAULT")
        current_right_model = AutoModelForCausalLM.from_pretrained(model_id)
        current_right_model.config.pad_token_id = tokenizer.pad_token_id
        current_right_model.resize_token_embeddings(len(tokenizer))
        
        vocab_size = current_right_model.config.vocab_size
        print('vocab_size:' + str((vocab_size)))
        reward_layer = nn.Sequential(
            nn.LayerNorm(vocab_size, current_right_model.config.layer_norm_eps),
            nn.Linear(vocab_size, 2),
            nn.Softmax(dim=-1)
        ).to(device)
        optimizer_right = optim.AdamW(list(current_right_model.parameters()) + list(reward_layer.parameters()), lr=lr, betas=betas,weight_decay=weight_decay)
    
    current_right_model = current_right_model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer_right, mode='min', factor=factor, patience=patience, cooldown=cooldown, threshold=0.0001, threshold_mode='rel',  min_lr=1e-9, eps=1e-08, verbose=True)

    

    # prepare current_right_model for training
    
    current_right_model.train()
    #loss_fn = nn.CrossEntropyLoss()

    #------------------
    # Start training!
    #------------------
    # TODO: load a batch from ./batch/*.json. Grab the top 100 most recent files, and then select randomly one of them.
    # each b in batch is of format: 
    #     ({'input_text':input_text, 'true_answer':true_answer, 'reward_mask':reward_mask})
    #     we can ignore true_answer, and just use input_text as input, and reward_mask as a "per token" reward [0 or 1].
    #         input_text = "Question: blah blah, Answer: blah blah blah"
    #         reward_mask = [1,1,1,0,0,0,0,1,1,...] which is len(input_text)
    # then we want to train the model with binary cross entropy 
    num_batches_since_last_checkpoint = 0
    iterr=0
    print("STARTING TRAINING")
    while True:
        # Load a batch from ./batches/*.json. Grab the top 100 most recent files, and then select randomly one of them.
        batch_files = sorted(glob.glob(os.path.join(batches_directory, '*.json')), key=os.path.getmtime, reverse=True)[:100]
        if not batch_files:
            print("No batch files found. Please create some batches.")
            exit()
        
        selected_batch_file = random.choice(batch_files)
        with open(selected_batch_file, 'r') as f:
            batch = json.load(f)
        
        input_texts = [sample['input_text'] for sample in batch]
        reward_masks = [sample['valence_mask'] for sample in batch]

        maxx = max([len(sample['input_text']) for sample in batch])
        if maxx > 2000:
           print("MAXXXX LENGTH IS SURPASSED SO SKIPPING, MAXX: " + str(maxx) )
           continue
        inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # reward_masks = torch.tensor(reward_masks).to(device)
        reward_masks_tensors = [torch.tensor(mask) for mask in reward_masks]
        reward_masks_tensors_padded = nn.utils.rnn.pad_sequence(reward_masks_tensors, batch_first=True, padding_value=0).to(device)
        batch_size = input_ids.size(0)
        
        # Initialize loss for accumulation
        total_loss = 0.0
        total_correct = 0
        iterss = 0
        optimizer_right.zero_grad()
        num_microbatches = (batch_size + max_microbatch_size - 1) // max_microbatch_size
        for i in range(num_microbatches):
            start_idx = i * max_microbatch_size
            end_idx = min(start_idx + max_microbatch_size, batch_size)
            
            microbatch_input_ids = input_ids[start_idx:end_idx, :]
            microbatch_attention_mask = attention_mask[start_idx:end_idx, :]
            microbatch_reward_masks = reward_masks_tensors_padded[start_idx:end_idx, :]

            
            # Forward pass through the right model and reward layer
            logits = current_right_model(microbatch_input_ids, attention_mask=microbatch_attention_mask, return_dict=True).logits
            outputs = reward_layer(logits)
            # outputs = outputs.view(microbatch_input_ids.size(0), -1, 2)  # Reshape back to (batch_size, seq_length, 2)
            
            # Calculate binary cross-entropy loss for each token
            microbatch_reward_masks_flat = microbatch_reward_masks.view(-1).long()
            outputs_flat = outputs.view(-1, 2)  # Flatten the outputs

            #loss_fn = nn.CrossEntropyLoss()
            #loss = loss_fn(outputs_flat, microbatch_reward_masks_flat.long())

            targets_one_hot = F.one_hot(microbatch_reward_masks_flat, num_classes=2).float()
            
            
            
            # Calculate focal loss
            loss = -1 * sigmoid_focal_loss(outputs_flat, targets_one_hot, alpha=2.0, gamma=4.0, reduction='mean')
            #pdb.set_trace()
            
            # Backpropagation
            loss.backward(retain_graph=True)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate accuracy
            per_sample_acc = ((outputs.argmax(dim=-1) == microbatch_reward_masks.long()).float().sum().item())
            total_correct += per_sample_acc
            iterss +=1
            # message = {"per iter loss": loss.item(), "per iter acc": per_sample_acc, "batch_size":input_ids.size(0),"batch_seq_size":input_ids.size(1), "iterss":iterss}
            # print(message)
        if iterr % macro_batch_size == (macro_batch_size-1):

            normed_grad = torch.nn.utils.clip_grad_norm_(current_right_model.parameters(), max_norm=1.).item()
            normed_grad_r = torch.nn.utils.clip_grad_norm_(reward_layer.parameters(), max_norm=1.).item()
            # Print detailed gradient norms
    
            # for name, param in current_right_model.named_parameters():
            #     if param.grad is not None:
            #         normed_grad_param = param.grad.norm().item()
            #         print(f"Gradient norm for {name}: {normed_grad_param}")
            # for name, param in reward_layer.named_parameters():
            #     if param.grad is not None:
            #         normed_grad_param = param.grad.norm().item()
            #         print(f"Gradient norm for {name}: {normed_grad_param}")
            optimizer_right.step()
            scheduler.step(total_loss)
            optimizer_right.zero_grad()

            accuracy = total_correct / (input_ids.size(0) * input_ids.size(1))  # Per-token accuracy
            message = {"right_BCE_loss": round(float(total_loss),2), "right_per_tok_acc": round(float(accuracy),2), "lr": optimizer_right.param_groups[0]['lr'], "norm_grad": round(float(normed_grad),2),"norm_grad_r": round(float(normed_grad_r),2)}

            if include_wandb:
                wandb.log(message)
            print(message)
        
        iterr+=1
        num_batches_since_last_checkpoint += 1
        
        if num_batches_since_last_checkpoint >= save_checkpoint_every_n_batches:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            checkpoint_filename = f"right_checkpoint_{timestamp}_iter_{iterr}_loss_{total_loss:.2f}.pth"
            checkpoint_filepath = os.path.join(right_model_directory, checkpoint_filename)
            torch.save({
                'model_state_dict': current_right_model.state_dict(),
                'reward_layer_state_dict': reward_layer.state_dict(),
                'optimizer_state_dict': optimizer_right.state_dict(),
                'loss': total_loss,
            }, checkpoint_filepath)
            print(f"Checkpoint saved at {checkpoint_filepath}")
            num_batches_since_last_checkpoint = 0
