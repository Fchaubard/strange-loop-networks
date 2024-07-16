# train_left.py 
# - picks up batches from ./batches/*, sorts by time to grab n most recent, and then trains left model on them. 
# - pushes results to wandb, and saves checkpoints to ./checkpoint_left/ with format 'left_checkpoint_timestamp_iter_loss.h5'
# - train with either Cross Entropy, Focal Loss, or with policy gradient optimization.. or PPO? or RLOO? or DPO? ... TBD! IMPLEMENT THIS LAST! Just get it working.

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


def forward_with_valence_input(current_left_model, input_ids, token_valences, attention_mask=None, return_dict=True, alpha=1.,baseline=0.5):
  
  original_embeddings = current_left_model.get_input_embeddings()

  embeddings = original_embeddings(input_ids)

  token_valences = (torch.tensor(token_valences).unsqueeze(-1).float() - baseline)*alpha
  token_valences = token_valences.expand(-1, -1, embeddings.size(-1))
  modified_embeddings = embeddings + token_valences

  logits = current_left_model(
    inputs_embeds=modified_embeddings,
    attention_mask=attention_mask,
    return_dict=return_dict
  ).logits

  # outputs = (batch x seq x V) (but unnormalized?)

  if not return_dict:
    return logits

  return {"logits": logits}

if __name__ == '__main__':
 
    #------------------
    # THINGS TO UPDATE:
    #------------------
    # Define the batches_directory file path
    batches_directory = "/sln_batches/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    
    # update what device you want to train this model on
    device = 'cuda:0'#'cpu'#

    # update what model you want to use WARNING: IF YOU WANT TO START FROM A CHECKPOINT OF LEFT MODEL, THIS IS THE PLACE TO DO IT:
    model_id = "EleutherAI/pythia-410M" #"EleutherAI/pythia-1b" #"EleutherAI/pythia-70m-v0"
    left_model_checkpoint_name = "/left-strange-loop-network-410m/left_checkpoint_20240715173212_iter_800_loss_37.63.pth" # use this if you want to load from a checkpoint, else will load from pythia pretrained
    
    left_model_directory = "/left_checkpoints/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    
    macro_batch_size = 1 
    pad_tok = '[PAD]'
    left_model_tok = '<left model>'

    # Initialize wandb
    wandb_project_name = "left_training_pythia_"+model_id.replace("/","_")

    lr = 1e-5
    weight_decay = 0.0001
    betas = (0.99,0.999)

    # for scheduler
    factor=0.5
    patience=500
    cooldown=500

    max_microbatch_size = 10 # IMPORTANT TO INCREASE IF YOU HAVE MORE GPU RAM
    max_ctx_len = 2000 # IMPORTANT TO INCREASE IF YOU HAVE MORE GPU RAM
  
    save_checkpoint_every_n_batches = 500

    # for handling the valence input signal
    valence_input_baseline = 0.5
    valence_input_alpha = 2.
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

    # Check if the left_model_directory exists
    if not os.path.exists(left_model_directory):
        # If the directory does not exist, create it
        os.mkdir(left_model_directory)
        print(f"Directory {batches_directory} doesnt exit! Making it.")
    else:
        print(f"Directory {batches_directory} exists.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_tok})
        print('setting tokenizer.pad_token_id to: ' + str(tokenizer.pad_token_id) + " with token: "+pad_tok)
        

    # create initial model 
    if left_model_checkpoint_name:
        print("LOADING MODEL FROM: " +left_model_checkpoint_name)
        # Implement load from a checkpoint
        checkpoint = torch.load(left_model_checkpoint_name)
        current_left_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        current_left_model.config.pad_token_id = tokenizer.pad_token_id
        current_left_model.resize_token_embeddings(len(tokenizer))
        current_left_model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_left = optim.AdamW(list(current_left_model.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        optimizer_left.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # load from hf default
        print("STARTING WEIGHTS FROM DEFAULT")
        current_left_model = AutoModelForCausalLM.from_pretrained(model_id)
        current_left_model.config.pad_token_id = tokenizer.pad_token_id
        current_left_model.resize_token_embeddings(len(tokenizer))
        optimizer_left = optim.AdamW(list(current_left_model.parameters()), lr=lr, betas=betas,weight_decay=weight_decay)
    
    current_left_model = current_left_model.to(device)
    scheduler = ReduceLROnPlateau(optimizer_left, mode='min', factor=factor, patience=patience, cooldown=cooldown, threshold=0.0001, threshold_mode='rel',  min_lr=1e-9, eps=1e-08, verbose=True)

    # prepare current_left_model for training
    current_left_model.train()

    

    #------------------
    # Start training!
    #------------------
    # TODO: load a batch from ./batch/*.json. Grab the top 100 most recent files, and then select randomly one of them.
    # each b in batch is of format: 
    #     ({'input_text':input_text, 'true_answer':true_answer, 'valence_mask':valence_mask})
    #     we can ignore true_answer, and just use input_text as input, and valence_mask as a "per token" valence [0 or 1].
    #         input_text = "Question: blah blah, Answer: blah blah blah"
    #         valence_mask = [1,1,1,0,0,0,0,1,1,...] which is len(input_text)
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

        # get the inputs and outputs ready
        input_texts = [sample['input_text'] for sample in batch]
        valence_masks = [sample['valence_mask'] for sample in batch]
        true_answers = [sample['true_answer'] for sample in batch]
        
        output_texts = []
        for input_text, true_answer in zip(input_texts,true_answers):
          split_point = input_text.find(left_model_tok) 
          if split_point==-1:
            output_target = input_text + left_model_tok + true_answer
          else:
            split_point += len(left_model_tok)
            output_target = input_text[:split_point] + "" + true_answer
          output_texts.append(output_target)
        

        maxx_input = max([len(i) for i in input_texts])
        maxx_output = max([len(o) for o in output_texts])
        
        print('training:'+str(iterr)) 
      
        if max(maxx_input,maxx_output) > max_ctx_len:
           print("MAXXXX LENGTH IS SURPASSED SO SKIPPING, maxx_input:" + str(maxx_input) +  " maxx_output:" + str(maxx_output) )
           continue

        
        inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True) #max_length=max_ctx_len, add_special_tokens=True) # TODO: MAKE SURE THIS INCLUDES BOS token
        outputs = tokenizer(output_texts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True) #padding='max_length', truncation=True, max_length=max_ctx_len,  add_special_tokens=True) # TODO: MAKE SURE THIS DOES NOT! 
        
        input_ids = inputs['input_ids'][:,:-1] #.to(device) # drop the last column
        output_ids = outputs['input_ids'][:,1:] #.to(device) # shift one right 
        attention_mask = torch.ones_like(input_ids)
        #attention_mask = inputs['attention_mask'][:,:-1].to(device)
        
        valence_masks_tensors = [torch.tensor(mask) for mask in valence_masks]
        valence_masks_tensors_padded = nn.utils.rnn.pad_sequence(valence_masks_tensors, batch_first=True, padding_value=0)
        
        # Ensure the padded sequences have the desired length
        #valence_masks_tensors_padded = valence_masks_tensors_padded[:, :max_ctx_len]
        #if valence_masks_tensors_padded.size(1) < max_ctx_len:
        #    padding = torch.zeros((valence_masks_tensors_padded.size(0), max_ctx_len - valence_masks_tensors_padded.size(1)))
        #    valence_masks_tensors_padded = torch.cat([valence_masks_tensors_padded, padding], dim=1)
              
        valence_masks_tensors_padded = valence_masks_tensors_padded[:,:-1] #.to(device)
        
        batch_size = input_ids.size(0)

        # Initialize loss for accumulation
        total_loss = 0.0
        total_correct = 0
        iterss = 0
        optimizer_left.zero_grad()
        num_microbatches = (batch_size + max_microbatch_size - 1) // max_microbatch_size
        for i in range(num_microbatches):
            start_idx = i * max_microbatch_size
            end_idx = min(start_idx + max_microbatch_size, batch_size)
            
            microbatch_input_ids = input_ids[start_idx:end_idx, :].to(device)
            microbatch_output_ids = output_ids[start_idx:end_idx, :].to(device)
            microbatch_attention_mask = attention_mask[start_idx:end_idx, :].to(device)
            microbatch_valence_masks = valence_masks_tensors_padded[start_idx:end_idx, :].to(device)

          
            # Forward pass through the left model
            model_outputs = forward_with_valence_input(current_left_model, 
                                                microbatch_input_ids, 
                                                microbatch_valence_masks, 
                                                attention_mask=microbatch_attention_mask, 
                                                return_dict=True, 
                                                alpha=valence_input_alpha, 
                                                baseline=valence_input_baseline)

            model_outputs = model_outputs['logits']
            #pdb.set_trace()
            # Calculate binary cross-entropy loss for each token
            
            microbatch_output_ids_padded = torch.nn.functional.pad(microbatch_output_ids, (0, model_outputs.shape[1] - microbatch_output_ids.shape[1]), value=tokenizer.eos_token_id)
            microbatch_output_ids_flat = microbatch_output_ids_padded.view(-1).long()
            model_outputs_flat = model_outputs.view(-1, model_outputs.shape[-1])
            #model_outputs_flat = model_outputs.view(-1, 2)  # Flatten the outputs
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(model_outputs_flat, microbatch_output_ids_flat.long())

            # Calculate focal loss
            # targets_one_hot = F.one_hot(microbatch_valence_masks_flat, num_classes=2).float()
            # loss = -1 * sigmoid_focal_loss(outputs_flat, targets_one_hot, alpha=2.0, gamma=4.0, reduction='mean')
            
            # Backpropagation
            loss.backward(retain_graph=True)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate accuracy
            per_sample_acc = ((model_outputs_flat.argmax(dim=-1) == microbatch_output_ids_flat).float().sum().item())
            total_correct += per_sample_acc
            iterss +=1
            # message = {"per iter loss": loss.item(), "per iter acc": per_sample_acc, "batch_size":input_ids.size(0),"batch_seq_size":input_ids.size(1), "iterss":iterss}
            # print(message)
        if iterr % macro_batch_size == (macro_batch_size-1):

            normed_grad = torch.nn.utils.clip_grad_norm_(current_left_model.parameters(), max_norm=1.).item()
            # Print detailed gradient norms
            # for name, param in current_left_model.named_parameters():
            #     if param.grad is not None:
            #         normed_grad_param = param.grad.norm().item()
            #         print(f"Gradient norm for {name}: {normed_grad_param}")
            optimizer_left.step()
            scheduler.step(total_loss)
            optimizer_left.zero_grad()

            accuracy = total_correct / (input_ids.size(0) * input_ids.size(1))  # Per-token accuracy.. TODO: This is actually just the last iter.. not the full macro_batch.. need to fix.. too lazy..
            message = {"left_BCE_loss": round(float(total_loss),2), "left_per_tok_acc": round(float(accuracy),2), "lr": optimizer_left.param_groups[0]['lr'], "norm_grad": round(float(normed_grad),2)}

            if include_wandb:
                wandb.log(message)
            print(message)
        
        iterr += 1
        num_batches_since_last_checkpoint += 1
        
        if num_batches_since_last_checkpoint >= save_checkpoint_every_n_batches:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            checkpoint_filename = f"left_checkpoint_{timestamp}_iter_{iterr}_loss_{total_loss:.2f}.pth"
            checkpoint_filepath = os.path.join(left_model_directory, checkpoint_filename)
            torch.save({
                'model_state_dict': current_left_model.state_dict(),
                'optimizer_state_dict': optimizer_left.state_dict(),
                'loss': total_loss,
            }, checkpoint_filepath)
            print(f"Checkpoint saved at {checkpoint_filepath}")
            num_batches_since_last_checkpoint = 0
