# train_left-COT-masking.py 
# - starts off with no masking, and then once the model hits >X% acc, it will subtract one token. 
# - INPUT: [PAD] + input_text + <left model> + true_answer[:-n] + [PAD] * n
# - OUTPUT: [PAD] + input_text + <left model> + true_answer

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
import sys
from torch.optim.lr_scheduler import LambdaLR
import math

import wandb
import os
sys.path.append('.')
sys.path.append(os.path.abspath(os.path.join('..', 'sentence_augs')))
from text_corrupter import text_corrupter_negative, generate_match_mask

from print_colored_text import print_colored_text


os.environ["WANDB_API_KEY"] = "cce47709d839921f0b13533529f31c8af7f3f4dc"


class ChainOfThoughtMasking:
    def __init__(self, vocab_size, accuracy_threshold, cooldown, pad_token_id, random_replacement=False):
        self.vocab_size = vocab_size
        self.accuracy_threshold = accuracy_threshold
        self.cooldown = cooldown
        self.n = 2
        self.pad_token_id = pad_token_id
        self.random_replacement = random_replacement
        self.accuracy_vector = []

    def mask_tokens(self, output_tokens, true_answer_len):
        input_tokens = output_tokens.clone()
        
        if self.n < true_answer_len and self.n > 0:
            if self.random_replacement:
                random_tokens = torch.tensor(random.choices(range(self.vocab_size), k=self.n))
                input_tokens[..., -self.n:] = random_tokens  # Masking the rightmost n tokens
            else:
                input_tokens[..., -self.n:] = self.pad_token_id  # Masking the rightmost n tokens

        return input_tokens

    def update_n(self, current_accuracy):
        self.accuracy_vector.append(current_accuracy)
        if np.mean(self.accuracy_vector) >= self.accuracy_threshold and len(self.accuracy_vector)>self.cooldown:
                self.n += 1
                self.cooldown_counter = 0
                self.accuracy_vector = []
    def __call__(self, output_tokens, current_accuracy,true_answer_len):
        input_tokens = self.mask_tokens(output_tokens,true_answer_len)
        self.update_n(current_accuracy)
        return input_tokens





# Define the warm-up and cosine decay scheduler
def lr_lambda(current_step):
    # if current_step < warmup_steps:
    #     # Linear warm-up
    #     return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay
    return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))


def forward_with_valence_input(current_left_model, input_ids, token_valences, attention_mask=None, return_dict=True, alpha=1.,baseline=0.5):

      add_to_embeddings_method = False
      
      if add_to_embeddings_method:
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
      else:
    
          original_embeddings = current_left_model.get_input_embeddings()
        
          embeddings = original_embeddings(input_ids)
          new_embeddings = embeddings.clone()
          
          new_embeddings[:, :, -1] = (token_valences.clone().float() - baseline)*alpha
    
          logits = current_left_model(
            inputs_embeds=new_embeddings,
            attention_mask=attention_mask,
            return_dict=return_dict
          ).logits
        
          # outputs = (batch x seq x V) (but unnormalized?)
          del token_valences, new_embeddings
      torch.cuda.empty_cache()
    
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
    device = 'cuda:2'#'cpu'#
    COT_accuracy_threshold = 0.98  # Example accuracy threshold
    COT_cooldown = 100  # Example cooldown period
    random_replacement = False
    
    # update what model you want to use WARNING: IF YOU WANT TO START FROM A CHECKPOINT OF LEFT MODEL, THIS IS THE PLACE TO DO IT:
    model_id = "EleutherAI/pythia-2.8b" #"EleutherAI/pythia-1b" #"EleutherAI/pythia-70m-v0"
    # left_model_checkpoint_name = "/left-strange-loop-network-410m/left_checkpoint_20240715173212_iter_800_loss_37.63.pth" # use this if you want to load from a checkpoint, else will load from pythia pretrained
    # left_model_directory = "/left_checkpoints_masking/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING

    # # # Get list of all checkpoint files in the directory
    # files = glob.glob(os.path.join(left_model_directory, "left_checkpoint_masking_*.pth"))
    # left_model_directory = "/left_checkpoints_masking_random_replacement/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    left_model_directory = "/left_checkpoints_masking/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    
    # # Get list of all checkpoint files in the directory
    files = glob.glob(os.path.join("/left_checkpoints/", "left_checkpoint_*.pth"))
    
    # Find the most recent file based on modification time
    left_model_checkpoint_name = max(files, key=os.path.getmtime)
    # left_model_checkpoint_name=None

    
    
    pad_tok = '[PAD]'
    left_model_tok = '<left model>'

    # Initialize wandb
    wandb_project_name = "left_training_pythia_"+model_id.replace("/","_")

    lr = 1e-5 * .5
    weight_decay = 0.0001
    betas = (0.99,0.999)

    # for scheduler
    factor=0.5
    patience=250
    cooldown=250

    macro_batch_size = 2
    max_microbatch_size = 2 # IMPORTANT TO INCREASE IF YOU HAVE MORE GPU RAM
    max_ctx_len = 3000 # IMPORTANT TO INCREASE IF YOU HAVE MORE GPU RAM
  
    save_checkpoint_every_n_batches = 500

    # for handling the valence input signal
    valence_input_baseline = 0.5
    valence_input_alpha = 2.
    include_wandb = True

    last_batches_to_sample_from = -1
    reset_lr = True

    decode_every_n_batches = 100

    
    
    


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
        print(f"Directory {left_model_directory} doesnt exit! Making it.")
    else:
        print(f"Directory {left_model_directory} exists.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Change attributes
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    # tokenizer.bos_token = ''
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_tok})
        # tokenizer.add_special_tokens({'bos_token': "<|begoftext|>"})
        print('setting tokenizer.pad_token_id to: ' + str(tokenizer.pad_token_id) + " with token: "+pad_tok)
        # print('setting tokenizer.pad_token_id to: ' + str(tokenizer.bos_token_id) + " with token: "+"<|begoftext|>")

    
    
    # create initial model 
    if left_model_checkpoint_name:
        print("LOADING MODEL FROM: " +left_model_checkpoint_name)
        # Implement load from a checkpoint
        checkpoint = torch.load(left_model_checkpoint_name,map_location=device)
        current_left_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        current_left_model.config.pad_token_id = tokenizer.pad_token_id
        current_left_model.resize_token_embeddings(len(tokenizer))
        current_left_model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_left = optim.AdamW(list(current_left_model.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        optimizer_left.load_state_dict(checkpoint['optimizer_state_dict'])
        if reset_lr:
            for param_group in optimizer_left.param_groups:
                param_group['lr'] = lr
        # optimizer_left = optim.AdamW(list(current_left_model.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        # load from hf default
        print("STARTING WEIGHTS FROM DEFAULT")
        current_left_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        current_left_model.config.pad_token_id = tokenizer.pad_token_id
        current_left_model.resize_token_embeddings(len(tokenizer))
        optimizer_left = optim.AdamW(list(current_left_model.parameters()), lr=lr, betas=betas,weight_decay=weight_decay)
    
    current_left_model = current_left_model.to(device)
    
    
    left_model_token_id = tokenizer(left_model_tok,  return_tensors='pt').input_ids[0,-1]
    COT_masking_func = ChainOfThoughtMasking(current_left_model.config.vocab_size, 
                                             COT_accuracy_threshold, 
                                             COT_cooldown, 
                                             tokenizer.pad_token_id,
                                             random_replacement=random_replacement)

    # scheduler = ReduceLROnPlateau(optimizer_left, mode='min', factor=factor, patience=patience, cooldown=cooldown, threshold=0.0001, threshold_mode='rel',  min_lr=1e-9, eps=1e-08, verbose=True)
    total_steps = 10000  # Example total number of steps
    warmup_steps = 100  # Example number of warm-up steps
    scheduler = LambdaLR(optimizer_left, lr_lambda)
    optimizer_left.zero_grad()
    torch.cuda.empty_cache()


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
    # Initialize loss for accumulation
    total_loss = 0.0
    total_correct = 0.0
    accuracy = 0.0
    denom = 0
    print("STARTING TRAINING")
    while True:
        # Load a batch from ./batches/*.json. Grab the top 100 most recent files, and then select randomly one of them.
        batch_files = sorted(glob.glob(os.path.join(batches_directory, '*.json')), key=os.path.getmtime, reverse=True)[:last_batches_to_sample_from]
        if not batch_files:
            print("No batch files found. Please create some batches.")
            exit()
        
        selected_batch_file = random.choice(batch_files)
        try:
            with open(selected_batch_file, 'r') as f:
                batch = json.load(f)
        except Exception as e:
            print("!!!could not load: " + selected_batch_file)
            continue
            
        # get the inputs and outputs ready
        
        input_texts_original = [sample['input_text'] for sample in batch]
        true_answers = [sample['true_answer'] for sample in batch]
        
                            
        output_texts = []
        # valence_masks = []
        # input_texts = []
        max_true_answer = []
        for input_text, true_answer in zip(input_texts_original,true_answers):
            split_point = input_text.find(left_model_tok) 
            if split_point==-1:
                output_target = tokenizer.eos_token  + " " + input_text + " "+left_model_tok +" "+ true_answer + pad_tok
            else:
                split_point += len(left_model_tok)
                output_target = tokenizer.eos_token  + " " + input_text[:split_point] + " " + true_answer + pad_tok
            
            # TODO! clean up case where we have 2 <left_model> tokens.. 
            # input_text = tokenizer.eos_token + " "+ input_text + pad_tok
            output_texts.append(output_target)
            # input_texts.append(input_text)
            true_answer_toks_len = tokenizer(true_answer, return_tensors='pt').input_ids.shape[-1]
            max_true_answer.append(true_answer_toks_len)
            
            # valence_masks.append(generate_match_mask(tokenizer,
            #                                        input_text,
            #                                        output_target)
            #                     )
        

        # maxx_input = max([len(i) for i in input_texts])
        maxx_output = max([len(o) for o in output_texts])
        
        print('training:'+str(iterr)) 
      
        if maxx_output > max_ctx_len:
           print("MAXXXX LENGTH IS SURPASSED SO SKIPPING, maxx_output:" + str(maxx_output) )
           continue
    
        optimizer_left.zero_grad()
        batch_size = len(output_texts)
        num_microbatches = (batch_size + max_microbatch_size - 1) // max_microbatch_size
        
        try:
            for i in range(num_microbatches):
                with torch.no_grad(): 
                    start_idx = i * max_microbatch_size
                    end_idx = min(start_idx + max_microbatch_size, batch_size)
                    
                    batch_output_texts = output_texts[start_idx:end_idx]
                    batch_max_true_answer = max(max_true_answer[start_idx:end_idx])
                    
                    
                    outputs = tokenizer(batch_output_texts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True) #padding='max_length', truncation=True, max_length=max_ctx_len,  add_special_tokens=True) # TODO: MAKE SURE THIS DOES NOT! 
                    maxx = outputs.input_ids.shape[-1]
                    
                    
                    inputs = COT_masking_func(outputs.input_ids, accuracy, batch_max_true_answer)
                    
 
                    # inputs = tokenizer(batch_input_text, return_tensors='pt', padding="max_length", max_length=maxx, truncation=True, add_special_tokens=True) #max_length=max_ctx_len, add_special_tokens=True) # TODO: MAKE SURE THIS INCLUDES BOS token
                    
                    input_ids = inputs[:,:-1].to(device) #.to(device) # drop the last column
                    output_ids = outputs['input_ids'][:,1:].to(device) #.to(device) # shift one right  (drop the BOS token)
                    attention_mask = torch.ones_like(input_ids).to(device)
                    # valence_masks_tensors = [torch.tensor(mask) for mask in batch_valence_masks]

                    
                    valence_masks_tensors = (outputs.input_ids.to(device) == inputs.to(device)).int()
                    
                    

                    valence_masks_tensors_padded = nn.utils.rnn.pad_sequence(valence_masks_tensors, batch_first=True, padding_value=0)
            
                    valence_masks_tensors_padded = valence_masks_tensors_padded[:,:-1] #.to(device)
            
        
                    
                # Forward pass through the left model
                model_outputs = forward_with_valence_input(current_left_model, 
                                                    input_ids, 
                                                    valence_masks_tensors_padded,
                                                    attention_mask=attention_mask, 
                                                    return_dict=True, 
                                                    alpha=valence_input_alpha, 
                                                    baseline=valence_input_baseline)
    
                model_outputs = model_outputs['logits']
                
                
                
                if iterr % decode_every_n_batches == 0:
                    output_toks = model_outputs.argmax(dim=-1) # batch x seq (value = tok)
                    for j in range(output_toks.shape[0]):
                        print("-"*5)
                        print("input_texts:",tokenizer.decode(input_ids[j], skip_special_tokens=False))
                        print("-"*5)
                        print("true_answers:",tokenizer.decode(output_ids[j], skip_special_tokens=False))
                        print("-"*5)
                        print("microbatch_attention_mask:",attention_mask[j])
                        print("-"*5)
                        print("valence_masks:",valence_masks_tensors_padded[j])
                        print("-"*5)
                        decoded_string = tokenizer.decode(output_toks[j], skip_special_tokens=False) 
                        print("model_output:",decoded_string)
                        print("-"*5)
                        
                # pdb.set_trace()
                # Calculate binary cross-entropy loss for each token
                output_ids_flat = output_ids.view(-1).long() # torch.Size([batch, seq]) -> batch*seq x 1
                model_outputs_flat = model_outputs.view(-1, model_outputs.shape[-1])   # torch.Size([batch, seq, Vocab])  -> batch*seq x 1
                # TODO microbatch_output_ids_padded vs. microbatch_output_ids????????
                
                
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(model_outputs_flat, output_ids_flat.long())
                # Calculate focal loss
                # targets_one_hot = F.one_hot(microbatch_valence_masks_flat, num_classes=2).float()
                # loss = -1 * sigmoid_focal_loss(outputs_flat, targets_one_hot, alpha=2.0, gamma=4.0, reduction='mean')
                
                # Backpropagation
                loss.backward()
                
                # Accumulate loss
                total_loss += loss.item()

                
                
                # Calculate accuracy
                
                per_sample_acc = ((model_outputs.argmax(dim=-1) == output_ids).float().sum().item())
                denom += output_ids.shape[0]*output_ids.shape[1]
                total_correct += per_sample_acc
                
                # message = {"per iter loss": loss.item(), "per iter acc": per_sample_acc, "batch_size":input_ids.size(0),"batch_seq_size":input_ids.size(1), "iterss":iterss}
                # print(message)
            if iterr % macro_batch_size == (macro_batch_size-1):
    
                normed_grad = torch.nn.utils.clip_grad_norm_(current_left_model.parameters(), max_norm=1.).item()
                # Print detailed gradient norms
                # for name, param in current_left_model.named_parameters():
                #     if param.grad is not None:
                #         normed_grad_param = param.grad.norm().item()
                #         print(f"Gradient norm for {name}: {normed_grad_param}")
                total_loss = total_loss / denom
                
                optimizer_left.step()
                # scheduler.step(total_loss)
                scheduler.step()
                optimizer_left.zero_grad()
    
                
                accuracy = total_correct / denom  # Per-token accuracy.. TODO: This is actually just the last iter.. not the full macro_batch.. need to fix.. too lazy..
                message = {"iterr":iterr, "left_BCE_loss": round(float(total_loss),5), "left_per_tok_acc": round(float(accuracy),5), "lr": optimizer_left.param_groups[0]['lr'], "norm_grad": round(float(normed_grad),2), "COT_masking_n": COT_masking_func.n }
    
                if include_wandb:
                    wandb.log(message)
                print(message)
                
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
                
                total_loss = 0.0
                total_correct = 0.0
                denom = 0
                
        except Exception as e:
            print("!!!could not fpass" )
            print(e)
            raise Exception(str(e))
            # pdb.set_trace()
            # continue
        iterr += 1
        num_batches_since_last_checkpoint += 1
        

        
        
