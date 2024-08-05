import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import pdb
import sys
from types import SimpleNamespace
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
from print_colored_text import print_colored_text

# Define the warm-up and cosine decay scheduler
def lr_lambda(current_step):
    if current_step < warmup_steps:
        # Linear warm-up
        return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay
    return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))


def split_model_across_gpus(model, gpu_list):
    device_map = {}
    num_gpus = len(gpu_list)
    layers = model.gpt_neox.layers
    num_layers = len(layers)
    layers_per_gpu = num_layers // (num_gpus-1)
    remainder = num_layers % (num_gpus-1)

    current_gpu_num = 0
    for i, layer in enumerate(layers):
        device_map[i] = gpu_list[current_gpu_num]
        layer.to(gpu_list[current_gpu_num])
        if (i + 1) % layers_per_gpu == 0 and current_gpu_num < num_gpus - 1:
            current_gpu_num += 1

    device_map["embed_in"] = gpu_list[0]
    model.gpt_neox.embed_in.to(gpu_list[0])  # Word embedding to GPU 0
    model.gpt_neox.final_layer_norm.to(gpu_list[-1])  # Final layer norm to last GPU
    model.embed_out.to(gpu_list[-1])  # LM head to last GPU
    device_map["final_layer_norm"] = gpu_list[-1]
    device_map["embed_out"] = gpu_list[-1]
    torch.cuda.empty_cache()
    return device_map
        
def generate_match_mask(tokenizer, string_true, string_corrupted):
    tokens_true = tokenizer.encode(string_true, return_tensors='pt')[0]
    tokens_corrupted = tokenizer.encode(string_corrupted, return_tensors='pt')[0]
    
    min_length = min(len(tokens_true), len(tokens_corrupted))
    match_mask = (tokens_true[:min_length] == tokens_corrupted[:min_length]).int()
    match_mask = torch.cat([match_mask, torch.zeros(len(tokens_corrupted) - min_length, dtype=torch.int)], dim=0)
    
    return match_mask.tolist()

def print_colored_text(valence_mask, token_ids, tokenizer):
    """
    Prints tokenized text with background colors based on valence_mask.
    
    Parameters:
    valence_mask (list of int): List of 0s and 1s indicating color (0 for red, 1 for green).
    token_ids (list of int): List of token IDs.
    tokenizer (object): Tokenizer with a decode method to decode token IDs.
    """
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"  # Reset to default color
    for valence, token_id in zip(valence_mask, token_ids[0]):
        # Decode the token ID
        token = tokenizer.decode(token_id)
        
        # Determine the color based on valence_mask
        color = RED if valence  <= 0.5  else GREEN
        
        # Print the token with the appropriate background color
        print(f"{color}{token}{RESET}", end='')
    
    # Ensure the final reset of color
    print(RESET)
    

class SLN:

    def __init__(self, 
                 base_model_id,
                 left_model_device_list,
                 right_model_device_list,
                 right_model_checkpoint_name=None, 
                 left_model_checkpoint_name=None, 
                 verbose=True, 
                 return_type1_answer=False, 
                 return_highest_valence=True, 
                 return_all_IDLs=False,
                 decrement_future_negative_logits_with_rewards=False,
                 add_thinking_space=False,
                 trajectories_per_IDL=3,
                 temperature=0.7,
                 train_configs=None
                ):
      
        self.base_model_id = base_model_id
        self.right_model_checkpoint_name = right_model_checkpoint_name
        self.left_model_checkpoint_name = left_model_checkpoint_name
        self.return_type1_answer = return_type1_answer
        self.return_highest_valence = return_highest_valence # if false, returns LAST IDL as the answer vs. highest valence
        self.return_all_IDLs = return_all_IDLs
        self.decrement_future_negative_logits_with_rewards = decrement_future_negative_logits_with_rewards
        self.add_thinking_space = add_thinking_space
        self.verbose = verbose
        
        self.left_model_device_list = left_model_device_list
        self.right_model_device_list = right_model_device_list
        
        self.max_ctx_len = 2000
        self.valence_input_baseline = 0.5
        self.valence_input_alpha = 2.
        self.valence_threshold = 0.97
        self.base_model_id = base_model_id
        self.trajectories_per_IDL = trajectories_per_IDL
        self.temperature = temperature
        self.train_configs = train_configs

        
        if train_configs: # just for simplicity.. 
            train_configs = SimpleNamespace(**train_configs)

        
        ####################
        # Setup Tokenizer
        ####################
        self.special_tokens_dict = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "sep_token": "<sep>",
            "left_model_token": "<left>",
            "right_model_token": "<right>",
            "mask_token": "<mask>"
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.add_special_tokens(self.special_tokens_dict)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        vocab_size = self.tokenizer.vocab_size
        
        ####################
        # Setup left model
        ####################
        if self.left_model_checkpoint_name:
            print("LOADING MODEL FROM: " +self.left_model_checkpoint_name)
            # Implement load from a checkpoint
            checkpoint = torch.load(self.left_model_checkpoint_name) #map_location="cpu"
            self.current_left_model = AutoModelForCausalLM.from_pretrained(self.base_model_id)
            self.current_left_model.resize_token_embeddings(len(self.tokenizer))
            self.current_left_model.load_state_dict(checkpoint['model_state_dict'])
            
            if train_configs:
                
                self.optimizer_left = optim.AdamW(list(self.current_left_model.parameters()), lr=train_configs.lr, betas=train_configs.betas, weight_decay=train_configs.weight_decay)
                self.optimizer_left.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if train_configs.reset_lr:
                for param_group in self.optimizer_left.param_groups:
                    param_group['lr'] = train_configs.lr
        else:
            # load from hf default
            print("STARTING LEFT WEIGHTS FROM DEFAULT")
            self.current_left_model = AutoModelForCausalLM.from_pretrained(self.base_model_id).to("cpu")
            self.current_left_model.resize_token_embeddings(len(self.tokenizer))
            if train_configs:
                
                self.optimizer_left = optim.AdamW(list(self.current_left_model.parameters()), lr=train_configs.lr, betas=train_configs.betas, weight_decay=train_configs.weight_decay)

                
        self.left_device_map = split_model_across_gpus(self.current_left_model, self.left_model_device_list)
          
        ####################
        # Setup right model
        ####################
        if self.right_model_checkpoint_name:
            print("LOADING MODEL FROM: " +self.right_model_checkpoint_name)
            # Implement load from a checkpoint
            checkpoint = torch.load(self.right_model_checkpoint_name)
            self.current_right_model = AutoModelForCausalLM.from_pretrained(self.base_model_id)
            self.current_right_model.resize_token_embeddings(len(tokenizer))
            self.current_right_model.load_state_dict(checkpoint['model_state_dict'])
            
            self.valence_layer = nn.Sequential(
                nn.LayerNorm(vocab_size, self.current_right_model.config.layer_norm_eps),
                nn.Linear(vocab_size, 2),
                nn.Softmax(dim=-1)
            )
            
            self.valence_layer.load_state_dict(checkpoint['valence_layer_state_dict'])
            
            if train_configs:
                self.optimizer_right = optim.AdamW(list(self.current_right_model.parameters()) + list(self.valence_layer.parameters()), lr=train_configs.lr, betas=train_configs.betas, weight_decay=train_configs.weight_decay)
                
                self.optimizer_right.load_state_dict(checkpoint['optimizer_state_dict'])
                if train_configs.reset_lr:
                    for param_group in self.optimizer_right.param_groups:
                        param_group['lr'] = train_configs.lr
        else:
            # load from hf default
            print("STARTING RIGHT WEIGHTS FROM DEFAULT")
            self.current_right_model = AutoModelForCausalLM.from_pretrained(self.base_model_id)
            self.current_right_model.resize_token_embeddings(len(self.tokenizer))
            self.valence_layer = nn.Sequential(
                nn.LayerNorm(vocab_size, self.current_right_model.config.layer_norm_eps),
                nn.Linear(vocab_size, 2),
                nn.Softmax(dim=-1)
            )
            
            if train_configs:
                self.optimizer_right = optim.AdamW(list(self.current_right_model.parameters()) + list(self.valence_layer.parameters()), lr=train_configs.lr, betas=train_configs.betas, weight_decay=train_configs.weight_decay)
        
        self.right_device_map = split_model_across_gpus(self.current_right_model, self.right_model_device_list)
        self.right_device_map["valence_layer"] = self.right_model_device_list[-1]
        self.valence_layer.to(self.right_device_map["valence_layer"])

        if train_configs:
            self.optimizer_left.zero_grad()
            self.optimizer_right.zero_grad()
            self.left_scheduler = LambdaLR(self.optimizer_left, lr_lambda)
            self.right_scheduler = LambdaLR(self.optimizer_right, lr_lambda)
            torch.cuda.empty_cache()
            
        print("consciousness booted! Give me a prompt:") 
    


    def _forward_model_multigpu(self,model, embeddings, attention_mask, device_map):
        
        hidden_states = embeddings.clone()
        
        # Pass through each layer manually
        for i, layer in enumerate(model.gpt_neox.layers):
            current_gpu = device_map[i] #i // layers_per_gpu if i < layers_per_gpu * (num_gpus - 1) else num_gpus - 1
            hidden_states = hidden_states.to(device_map[current_gpu])
            attention_mask = attention_mask.to(device_map[current_gpu])
            hidden_states = layer(hidden_states, attention_mask)[0]
    
        hidden_states = model.gpt_neox.final_layer_norm(hidden_states.to(device_map["final_layer_norm"])
        logits = model.embed_out(hidden_states.device_map["embed_out"])
        return logits

    
    def _forward_right(self, input_ids, attention_mask, round=False):
        input_ids = input_ids.to(self.right_device_map["embed_in"])
        attention_mask = attention_mask.to(self.right_device_map["embed_in"])
        embeddings = model.gpt_neox.embed_in(input_ids)
        logits = self._forward_model_multigpu(self.current_right_model, embeddings, attention_mask, self.right_device_map)
        # logits = self.current_right_model(input_ids, attention_mask=attention_mask, return_dict=True).logits
        outputs = self.valence_layer(logits.to(self.right_device_map["valence_layer"])) # which is batch x seq x 2 (the second channel is the positive valence )
        
        if round:
            valence_mask = torch.round(softmax(outputs, dim=-1))
        else:
            valence_mask = softmax(outputs, dim=-1)
        return valence_mask


    
    def _forward_left(self, input_ids, input_valence, attention_mask, zero_out_bit_flag=None):
        
        input_ids = input_ids.to(self.left_device_map["embed_in"])
        attention_mask = attention_mask.to(self.left_device_map["embed_in"])
        input_valence = input_valence.to(self.left_device_map["embed_in"])
        
        logits = self._forward_left_with_valence_input(
            self.current_left_model,
            input_ids,
            input_valence,
            attention_mask=attention_mask,
            alpha=self.valence_input_alpha,
            baseline=self.valence_input_baseline
        )

        return logits


        def _generate_samples_from_logits(self, logits):
    
            generated_tokens = []
            
            # we will generate self.trajectories_per_IDL samples 
            for i in range(self.trajectories_per_IDL):
                # generated_tokens = torch.argmax(model_outputs, dim=-1) # TODO: greedy sampling?? or be smarter..
                    
                # Apply softmax to convert logits to probabilities
                probabilities = torch.softmax(logits / self.temperature, dim=-1)
                    
                # Sample from the probabilities
                sampled_tokens = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1)
                
                # Reshape the sampled tokens to match the original shape
                sampled_tokens = sampled_tokens.view(probabilities.size()[:-1])
                
                generated_tokens.append(sampled_tokens)
            
            # Concatenate all the generated tokens along the first dimension
            return torch.cat(generated_tokens, dim=0)

    def _forward_left_with_valence_input(self, 
                                        current_left_model, 
                                        input_ids, 
                                        token_valences, 
                                        attention_mask=None, 
                                        alpha=1., 
                                        baseline=0.5):
        
        original_embeddings = current_left_model.get_input_embeddings()
        embeddings = original_embeddings(input_ids.to(self.left_device_map["embed_in"]))

        token_valences = (token_valences.clone().float() - baseline)*alpha
        # token_valences = token_valences.expand(-1, -1, embeddings.size(-1))
        # modified_embeddings = embeddings + token_valences
        embeddings[:, :, -1] = token_valences.to(self.left_device_map["embed_in"])

        logits = self._forward_model_multigpu(
                        self.current_left_model,
                        embeddings,
                        attention_mask,
                        self.left_device_map
                        )
        
        return logits

    def _stopping_criteria(self, IDL_count, valence):
        # return True if we should stop IDL, False if we should not
        if IDL_count > self.IDL_limit:
            print("hit IDL_count limit, stopping..")
            return False #break out of IDL
        else:
            average_valence = 1.0*sum(valence) / len(valence)
            assert average_valence <=1 and average_valence >= 0
            if average_valence >= self.valence_threshold: # meaning mostly 1s valence in the left response. 
                
                print(f"hit average_valence {average_valence} > thresh {self.valence_threshold}")
                
                return False #we got a good answer! Break out of IDL! 
                
            return True #continue with IDL.. not there yet.. 
        
    def _decode(self, output_tokens):
        tokens = output_tokens.cpu().numpy()
        txt = []
        for i in range(tokens.shape[0]):
            txt.append(self.tokenizer.decode(output_tokens.cpu().numpy()[i], skip_special_tokens=False))
        return txt 


    def forward(self, input_token_ids):

        pdb.set_trace()
        generated_samples = input_token_ids
        attention_mask = torch.ones_like(generated_samples).to(self.right_model_device_list[0])
        
        IDL_ids = {}
        IDL_iterr = 0
    
        while True:
            
            IDL_ids[IDL_iterr] = {}
            # Forward pass right model on all samples (initially will be just the input)
            valence_mask = self._forward_right(generated_samples, attention_mask)[:,:,1]
            
            for i in range(valence_mask.shape[0]):
                
                total_valence = torch.sum(valence_mask[i,...]).item() 
                sample = generated_samples[i,...].clone().detach()
                IDL_ids[IDL_iterr][i] = {"valence_mask":valence_mask[i,...].clone().detach(),
                                        "IDL_ids":sample,
                                        "IDL_string":sln.tokenizer._decode(sample),
                                        "total_valence":total_valence
                                        }
                                        
                if self.verbose:
                    print(f"IDL count {IDL_iterr}, sample count {i}: total_valence: {total_valence} on : { valence_mask.shape[-1] } IDL: ", end='')
                    print_colored_text(valence_mask[i,...], sample, self.tokenizer)
                    print("------------------------------------")

            # find best valence trajectory so far, and continue on from there
            best_sample = max(IDL_ids[IDL_iterr], key = lambda x: x["total_valence"])
            
            best_valence_mask = best_sample["valence_mask"]
            best_IDL_ids = best_sample["IDL_ids"]
            best_IDL_string = best_sample["IDL_string"]
            best_total_valence = best_sample["total_valence"]
            
            if not self._stopping_criteria(IDL_iterr,best_valence_mask):
                # score is high enough or we have hit our limit
                break
            
            # score not high enough.. we need to keep trying
            if self.verbose:
                print('Now doing Type 2 thinking to really think about it...')
        
            # Forward pass left model and generate samples from them
            logits = self._forward_left(best_IDL_ids, best_valence_mask, attention_mask, zero_out_bit_flag=None)
            generated_samples = self._generate_samples_from_logits(logits)
            
            IDL_iterr += 1

        return IDL_ids

    def learn_left(self, IDL_ids, target_ids): 

        IDL_ids = torch.cat([d["IDL_ids"].to(self.left_device_map["embed_in"]) for sub_dict in IDL_ids.values() for d in sub_dict])
        valence_masks = torch.cat([d["valence_mask"].to(self.left_device_map["embed_in"]) for sub_dict in IDL_ids.values() for d in sub_dict])
        attention_mask = torch.ones_like(IDL_ids).to(self.left_device_map["embed_in"])

        # fpass the model on all IDLs
        logits = self._forward_left(IDL_ids, valence_masks, attention_mask, zero_out_bit_flag=None)

        assert IDL_ids.shape[-1] == target_ids[-1]
        
        # Right shift the targets for autoregressive prediction
        
        batch, seq_len, vocab_size = logits.shape
        target_ids = target_ids.expand(batch_size, -1)

        shifted_targets = torch.zeros_like(IDL_ids)
        shifted_targets[..., :-1] = target_ids[..., 1:]
        shifted_targets[..., -1] = tokenizer.eos_token_id
        
        # Compute the loss
        loss = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Calculate metrics
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == shifted_targets).float()
        accuracy = (correct.sum() / correct.numel()).item()
        perplexity = torch.exp(loss).item()
        loss = loss.item()
        if verbose:# Logging
            print(f" Loss: {loss}", end='')
            print(f" Accuracy: {accuracy * 100:.2f}%", end='')
            print(f" Perplexity: {perplexity}", end='')
            print(f" Learning Rate: {self.left_scheduler.get_last_lr()[0]}")
            # pdb.set_trace()
            # for i in range(batch):
            #     print(f" Prediction: {self.tokenizer.decode(predictions[i,...], skip_special_tokens=False)}")
            #     print("")
            #     print("")
            print("-----"*50)
    
        #	 - loop over all IDLs, calc loss, and grad accumulate for left model
        #	 - return metrics 
        #		 - perplexity
        #		 - loss
        #		 - per tok accuracy

        return loss, perplexity, accuracy
    
    def learn_right(IDL_ids, target_valences):# loss_fn = [CE,Focal,RLOO,PG,etc..]):
        #	 - loop over all IDLs, calc loss, and grad accumulate for right model
        #	 - return metrics 
        #		 - loss
        #		 - classification_accuracy
        
        IDL_ids = torch.cat([d["IDL_ids"].to(self.right_device_map["embed_in"]) for sub_dict in IDL_ids.values() for d in sub_dict])
        valence_masks = torch.cat([d["valence_mask"].to(self.right_device_map["embed_in"]) for sub_dict in IDL_ids.values() for d in sub_dict])
        attention_mask = torch.ones_like(IDL_ids).to(self.right_device_map["embed_in"])

        # fpass the model on all IDLs to get the valence_masks 
        valence_probs = self._forward_right(IDL_ids, attention_mask)
        
        # Right shift the targets for autoregressive prediction
        target_valences = target_valences.view(-1).long()
        outputs_flat = valence_probs.view(-1, 2)  # Flatten the outputs

        focal_loss = False
        if focal_loss:
            # Calculate focal loss
            targets_one_hot = F.one_hot(target_valences, num_classes=2).float()
            loss = -1 * sigmoid_focal_loss(outputs_flat, targets_one_hot, alpha=2.0, gamma=4.0, reduction='mean')                
        else:

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs_flat, target_valences.long())

        # Backward pass
        loss.backward()
        
        # Calculate metrics
        predictions = torch.argmax(valence_probs, dim=-1)
        correct = (predictions == target_valences).float()
        accuracy = (correct.sum() / correct.numel()).item()
        loss = loss.item()
        if verbose:# Logging
            print(f" Loss: {loss}", end='')
            print(f" Accuracy: {accuracy * 100:.2f}%", end='')
            print(f" Learning Rate: {self.right_scheduler.get_last_lr()[0]}")
            # pdb.set_trace()
            # for i in range(batch):
            #     print(f" Prediction: {self.tokenizer.decode(predictions[i,...], skip_special_tokens=False)}")
            #     print("")
            #     print("")
            print("-----"*50)

        return loss, accuracy
        
    def update_weights_left(total_loss):
        # optim.step for both left / right
        normed_grad = torch.nn.utils.clip_grad_norm_(current_left_model.parameters(), max_norm=1.).item()
        self.optimizer_left.step()
        scheduler.step(total_loss)
        self.optimizer_left.zero_grad()
        torch.cuda.empty_cache()
        return normed_grad
        
    def update_weights_right(total_loss):
        # optim.step for both left / right
        normed_grad = torch.nn.utils.clip_grad_norm_(self.current_right_model.parameters(), max_norm=1.).item()
        normed_grad_r = torch.nn.utils.clip_grad_norm_(self.valence_layer.parameters(), max_norm=1.).item()
        self.optimizer_right.step()
        scheduler.step(total_loss)
        self.optimizer_right.zero_grad()
        torch.cuda.empty_cache()
        return normed_grad + normed_grad_r
        
    def save_checkpoints(iterr,left_loss,right_loss):
        
        # save left checkpoint
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_filename = f"left_checkpoint_{timestamp}_iter_{iterr}_loss_{left_loss:.2f}.pth"
        checkpoint_filepath = os.path.join(self.left_model_directory, checkpoint_filename)
        torch.save({
            'model_state_dict': self.current_left_model.state_dict(),
            'optimizer_state_dict': self.optimizer_left.state_dict(),
            'loss': left_loss,
        }, checkpoint_filepath)
        print(f"Checkpoint saved at {checkpoint_filepath}")
        
        # save right checkpoint
        checkpoint_filename = f"right_checkpoint_{timestamp}_iter_{iterr}_loss_{right_loss:.2f}.pth"
        checkpoint_filepath = os.path.join(self.right_model_directory, checkpoint_filename)
        torch.save({
            'model_state_dict': self.current_right_model.state_dict(),
            'valence_layer_state_dict': self.valence_layer.state_dict(),
            'optimizer_state_dict': self.optimizer_right.state_dict(),
            'loss': right_loss,
        }, checkpoint_filepath)
        print(f"Checkpoint saved at {checkpoint_filepath}")

if __name__ == "__main__":
    print("Booting consciousness... one sec.. :)")
    left_model_checkpoint = "/left_checkpoints/left_checkpoint_20240715113638_iter_700_loss_43.06.pth" #"<path to right model checkpoint>"
    # right_model_checkpoint = "/right_checkpoints/right_checkpoint_20240715155144_iter_10_loss_6.31.pth" #"<path to left model checkpoint>"
    # left_model_checkpoint = "./left_checkpoint_20240711093303_iter_100_loss_55.50.pth" #"<path to right model checkpoint>"
    right_model_checkpoint = "/root/right_checkpoint_20240709113005_iter_2000_loss_0.52.pth" #"<path to left model checkpoint>"
    
    sln = SLN(right_model_checkpoint, left_model_checkpoint)
  
    prompt_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    target_response = "Let's break this down step by step.\n\n**Step 1: Understand the problem**\nNatalia sold clips to 48 friends in April. Then, she sold half as many clips in May. We need to find the total number of clips she sold in April and May.\n\n**Step 2: Calculate the number of clips sold in May**\nIf Natalia sold half as many clips in May as she did in April, that means she sold:\n\n48 (clips sold in April) / 2 = 24 clips in May\n\n**Step 3: Add the number of clips sold in April and May**\nTo find the total number of clips sold, we add the number of clips sold in April and May:\n\n48 (clips sold in April) + 24 (clips sold in May) = 72\n\n**Conclusion**\nNatalia sold a total of 72 clips in April and May. #### 72"

    model_response = sln.forward(prompt_text)
    
    print("Target Response:", target_response)
    print("="*50)
    print("Model Response:", model_response)
    
