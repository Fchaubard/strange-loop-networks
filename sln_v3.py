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
import concurrent
import traceback
import wandb
import os
sys.path.append('.')
from print_colored_text import print_colored_text



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
    print(device_map)
    
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
    for valence, token_id in zip(valence_mask, token_ids):
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
                 IDL_limit = 1,
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
        self.IDL_limit = IDL_limit
        self.temperature = temperature
        self.train_configs = train_configs

        

        
        ####################
        # Setup Tokenizer
        ####################
        self.left_model_token =  "<left_model>"
        self.right_model_token =  "<right_model>"
        
        self.special_tokens_dict = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "sep_token": "<sep>",
            "mask_token": "<mask>",
            "additional_special_tokens": 
            [
                self.left_model_token, 
                self.right_model_token
            ]
        }
        

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.add_special_tokens(self.special_tokens_dict)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        # vocab_size = self.tokenizer.vocab_size
        
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
            self.left_device_map = split_model_across_gpus(self.current_left_model, self.left_model_device_list)
            
            if train_configs:
                
                self.optimizer_left = optim.AdamW(list(self.current_left_model.parameters()), lr=train_configs.left_lr, betas=train_configs.left_betas, weight_decay=train_configs.weight_decay)
                self.optimizer_left.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # if train_configs.reset_lr:
            #     for param_group in self.optimizer_left.param_groups:
            #         param_group['lr'] = train_configs.left_lr
        else:
            # load from hf default
            print("STARTING LEFT WEIGHTS FROM DEFAULT")
            self.current_left_model = AutoModelForCausalLM.from_pretrained(self.base_model_id).to("cpu")
            self.current_left_model.resize_token_embeddings(len(self.tokenizer))
            if train_configs:
                
                self.optimizer_left = optim.AdamW(list(self.current_left_model.parameters()), lr=train_configs.left_lr, betas=train_configs.left_betas, weight_decay=train_configs.weight_decay)

                
            self.left_device_map = split_model_across_gpus(self.current_left_model, self.left_model_device_list)
          
        ####################
        # Setup right model
        ####################
        if self.right_model_checkpoint_name:
            print("LOADING MODEL FROM: " +self.right_model_checkpoint_name)
            # Implement load from a checkpoint
            checkpoint = torch.load(self.right_model_checkpoint_name)
            self.current_right_model = AutoModelForCausalLM.from_pretrained(self.base_model_id)
            self.current_right_model.resize_token_embeddings(len(self.tokenizer))
            self.current_right_model.load_state_dict(checkpoint['model_state_dict'])
            
            self.valence_layer = nn.Sequential(
                nn.LayerNorm(len(self.tokenizer), self.current_right_model.config.layer_norm_eps),
                nn.Linear(len(self.tokenizer), 2),
                nn.Softmax(dim=-1)
            )
            
            self.valence_layer.load_state_dict(checkpoint['valence_layer_state_dict'])
            self.right_device_map = split_model_across_gpus(self.current_right_model, self.right_model_device_list)
            self.right_device_map["valence_layer"] = self.right_model_device_list[-1]
            self.valence_layer.to(self.right_device_map["valence_layer"])
            
            if train_configs:
                self.optimizer_right = optim.AdamW(list(self.current_right_model.parameters()) + list(self.valence_layer.parameters()), lr=train_configs.right_lr, betas=train_configs.right_betas, weight_decay=train_configs.weight_decay)
                
                self.optimizer_right.load_state_dict(checkpoint['optimizer_state_dict'])
                # if train_configs.reset_lr:
                #     for param_group in self.optimizer_right.param_groups:
                #         param_group['lr'] = train_configs.right_lr
        else:
            # load from hf default
            print("STARTING RIGHT WEIGHTS FROM DEFAULT")
            self.current_right_model = AutoModelForCausalLM.from_pretrained(self.base_model_id)
            self.current_right_model.resize_token_embeddings(len(self.tokenizer))
            self.valence_layer = nn.Sequential(
                nn.LayerNorm(len(self.tokenizer), self.current_right_model.config.layer_norm_eps),
                nn.Linear(len(self.tokenizer), 2),
                nn.Softmax(dim=-1)
            )
            
            if train_configs:
                self.optimizer_right = optim.AdamW(list(self.current_right_model.parameters()) + list(self.valence_layer.parameters()), lr=train_configs.right_lr, betas=train_configs.right_betas, weight_decay=train_configs.weight_decay)
        
            self.right_device_map = split_model_across_gpus(self.current_right_model, self.right_model_device_list)
            self.right_device_map["valence_layer"] = self.right_model_device_list[-1]
            self.valence_layer.to(self.right_device_map["valence_layer"])

        if train_configs:
            self.optimizer_left.zero_grad()
            self.optimizer_right.zero_grad()
            # Define the warm-up and cosine decay scheduler
            def lr_lambda(current_step, warmup_steps=train_configs.warmup_steps, total_steps=train_configs.total_steps):
                if current_step < warmup_steps:
                    # Linear warm-up
                    return float(current_step) / float(max(1, warmup_steps))
                # Cosine decay
                return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))


            self.left_scheduler = LambdaLR(self.optimizer_left, lr_lambda)
            self.right_scheduler = LambdaLR(self.optimizer_right, lr_lambda)

        # this is for RL loss functions for the left lobe
        self.baseline_reward = self.train_configs.baseline_reward_for_rl_loss   # -1.0
        
        print("consciousness booted! Give me a prompt:") 
    


    def _forward_model_multigpu(self, model, embeddings, attention_mask, device_map):

        hidden_states = embeddings.clone()
        
        if attention_mask.dim() == 2:
                # Assuming attention_mask shape is [batch_size, seq_len]
                attention_mask = attention_mask[:, None, None, :]  # Convert to [batch_size, 1, 1, seq_len]
            
        # Pass through each layer manually
        for i, layer in enumerate(model.gpt_neox.layers):
            hidden_states = hidden_states.to(device_map[i])
            attention_mask = attention_mask.to(device_map[i])
            hidden_states = layer(hidden_states, attention_mask)[0]
    
        hidden_states = model.gpt_neox.final_layer_norm(hidden_states.to(device_map["final_layer_norm"]))
        logits = model.embed_out(hidden_states.to(device_map["embed_out"]))
        return logits

    
    def _forward_right(self, input_ids, attention_mask, round=False):
        input_ids = input_ids.clone().detach().to(self.right_device_map["embed_in"])
        attention_mask = attention_mask.clone().detach().to(self.right_device_map["embed_in"])
        embeddings = self.current_right_model.gpt_neox.embed_in(input_ids)
        logits = self._forward_model_multigpu(self.current_right_model, embeddings, attention_mask, self.right_device_map)
        # logits = self.current_right_model(input_ids, attention_mask=attention_mask, return_dict=True).logits
        outputs = self.valence_layer(logits.to(self.right_device_map["valence_layer"])) # which is batch x seq x 2 (the second channel is the positive valence )
        
        if round:
            valence_mask = torch.round(softmax(outputs, dim=-1))
        else:
            valence_mask = softmax(outputs, dim=-1)
        return valence_mask


    
    def _forward_left(self, input_ids, input_valence, attention_mask, zero_out_bit_flag=None):
        
        input_ids = input_ids.clone().detach().to(self.left_device_map["embed_in"])
        attention_mask = attention_mask.clone().detach().to(self.left_device_map["embed_in"])
        input_valence = input_valence.clone().detach().to(self.left_device_map["embed_in"])
        
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
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
              
        # we will generate self.trajectories_per_IDL samples 
        for i in range(self.trajectories_per_IDL):
            # generated_tokens = torch.argmax(model_outputs, dim=-1) # TODO: greedy sampling?? or be smarter..
            
            # Sample from the probabilities
            sampled_tokens = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1).T
            
            bos_token_tensor = torch.full((1, 1), 
                                          self.tokenizer.bos_token_id, 
                                          dtype=sampled_tokens.dtype, 
                                          device=sampled_tokens.device)
            
            # Concatenate bos_token_tensor with sampled_tokens along the sequence dimension
            sampled_tokens = torch.cat((bos_token_tensor, sampled_tokens[:,:-1]), dim=1)
            
            generated_tokens.append(sampled_tokens)
        
        # Concatenate all the generated tokens along the first dimension
        return torch.cat(generated_tokens, dim=0)

    def _forward_left_with_valence_input(self, 
                                        current_left_model, 
                                        input_ids, 
                                        token_valences, 
                                        attention_mask=None, 
                                        alpha=2., 
                                        baseline=0.5):
        
        original_embeddings = current_left_model.get_input_embeddings()
        
        embeddings = original_embeddings(input_ids.to(self.left_device_map["embed_in"]))

    
        token_valences = (token_valences.clone().float() - baseline)*alpha
        # token_valences = token_valences.expand(-1, -1, embeddings.size(-1))
        # modified_embeddings = embeddings + token_valences

        update_embeddings_with_valences = False
        if update_embeddings_with_valences:
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
        if IDL_count >= self.IDL_limit:
            if self.verbose:
                print("Hit IDL_count limit.")
            return False #break out of IDL
        else:
            average_valence = torch.mean(valence)
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

    
    def compute_policy_gradients(self, probs, generated_samples, token_valences, moving_average_size=300):
        # Compute the log probabilities of the actions taken (i.e., generated_samples)
        device = probs.device
        log_probs = torch.log(probs + 1e-9)
        
        # Gather the log probabilities corresponding to the actions (generated_samples)
        log_probs = log_probs.gather(2, generated_samples.unsqueeze(-1).to(device)).squeeze(-1)
        
        token_valences_centered = (token_valences - self.valence_input_baseline)*self.valence_input_alpha
        
        seq_len = token_valences.size(1)
        
        # Calculate the sum of rewards across the sequence
        reward_sum = token_valences_centered.sum(dim=1, keepdim=True).to(device)

        self.baseline_reward.extend(reward_sum.tolist()) 
        if len(self.baseline_reward) > moving_average_size:
            self.baseline_reward = self.baseline_reward[-1*moving_average_size:]
        
        reward_sum_normalized = (reward_sum - torch.mean(self.baseline_reward) )/ (torch.std(self.baseline_reward) + 1)
        
        # Compute the policy gradient loss
        loss = -(log_probs * reward_sum_normalized).mean()
        
        return loss


    def compute_policy_gradients_with_rloo(self, probs, generated_samples, token_valences):
        # if generated_samples.shape[0]<2:
        #     raise Exception("generated_samples must be >2 for RLOO, we recommend big, like 30")
        device = probs.device
        
        # Compute log probabilities
        log_probs = torch.log(probs + 1e-9)
    
        # Gather the log probabilities corresponding to the actions (generated_samples)
        log_probs = log_probs.gather(2, generated_samples.unsqueeze(-1).to(device)).squeeze(-1)
    
        # Sum log probabilities across the sequence to treat the entire sequence as a single action
        log_probs_sum = log_probs.sum(dim=1, keepdim=True).to(device)
    
        # Calculate the Leave-One-Out (RLOO) baseline
        seq_len = token_valences.size(0)
        reward_sum_per_sample = token_valences.mean(dim=1, keepdim=True).to(device)
        reward_sum_total = reward_sum_per_sample.sum()
        rloo_baseline_mean_per_sample = (reward_sum_total - reward_sum_per_sample) / (seq_len - 1)
        # rloo_baseline_std = reward_sum.std()
    
        # Adjust the rewards with the baseline and normalize
        #advantage = (reward_sum_per_sample - rloo_baseline_mean_per_sample) #/ (1. + rloo_baseline_mean_per_sample.std())
        advantage = reward_sum_per_sample
        # Compute the policy gradient loss using the summed log probabilities
        loss = -(log_probs_sum * advantage).mean()

        # loss = loss/(1.0 + torch.abs(loss))
        
    
        return loss

    # def compute_policy_gradients_with_rloo(self, probs, generated_samples, token_valences):
    #     if generated_samples.shape[0]<2:
    #         raise Exception("generated_samples must be >2 for RLOO, we recommend big, like 30")
    #     device = probs.device
        
    #     # Compute log probabilities
    #     log_probs = torch.log(probs + 1e-9)
    
    #     # Gather the log probabilities corresponding to the actions (generated_samples)
    #     log_probs = log_probs.gather(2, generated_samples.unsqueeze(-1).to(device)).squeeze(-1)
    
    #     # Sum log probabilities across the sequence to treat the entire sequence as a single action
    #     log_probs_sum = log_probs.sum(dim=1, keepdim=True).to(device)
    
    #     # Calculate the Leave-One-Out (RLOO) baseline
    #     seq_len = token_valences.size(1)
    #     reward_sum = token_valences.sum(dim=1, keepdim=True).to(device)
    #     rloo_baseline_average = reward_sum.mean()
    #     rloo_baseline_std = reward_sum.std()
    
    #     # Adjust the rewards with the baseline and normalize
    #     advantage = (reward_sum - rloo_baseline_average) / (rloo_baseline_std + 1.0)
    
    #     # Compute the policy gradient loss using the summed log probabilities
    #     loss = -(log_probs_sum * advantage).mean()
        
    #     loss = loss/(1.0 + torch.abs(loss))
        
    #     # pdb.set_trace()
    
    #     return loss

    # def compute_policy_gradients_with_rloo(self, probs, generated_samples, token_valences):
    #     # Compute the log probabilities of the actions taken (i.e., generated_samples)
    #     device = probs.device
    #     log_probs = torch.log(probs + 1e-9)
        
    #     # Gather the log probabilities corresponding to the actions (generated_samples)
    #     log_probs = log_probs.gather(2, generated_samples.unsqueeze(-1).to(device)).squeeze(-1)
        
    #     # Calculate the Leave-One-Out (RLOO) baseline
    #     seq_len = token_valences.size(1)

        
    #     # Calculate the sum of rewards across the sequence
    #     reward_sum = token_valences.sum(dim=1, keepdim=True).to(device)

    #     token_valences = token_valences.to(device)

    #     rloo_baseline = (reward_sum - token_valences) / (seq_len - 1)
    
    #     # Subtract the RLOO baseline from the rewards and normalize
    #     adjusted_rewards = token_valences - rloo_baseline
    #     adjusted_rewards = (adjusted_rewards - adjusted_rewards.mean()) / (adjusted_rewards.std() + 1.0)

    #     # Compute the policy gradient loss
    #     loss = -(log_probs * adjusted_rewards).mean()
        
    #     return loss

    def forward(self, input_token_ids, target_ids = None):

        generated_samples = input_token_ids
        attention_mask = torch.ones_like(generated_samples).to(self.right_model_device_list[0])
        
        IDL_ids = {}
        IDL_iterr = 0
    
        while True:
          try:
            IDL_ids[IDL_iterr] = []
            # Forward pass right model on all samples (initially will be just the input)
            valence_mask_probs = self._forward_right(generated_samples, attention_mask) # only need last channel.. for reward
            valence_mask = valence_mask_probs[:,:,1]
            # if valence_mask.dim()==2:
            #     valence_mask = valence_mask.unsqueeze(0)
              
            if target_ids!=None: # if you provide sln.forward() with target_ids, then we will backprop
                ##########
                # Backprop for this IDL
                ##########
                
               if IDL_iterr==0:
                    # You should backward pass right model now. TODO
                    right_loss, right_classification_accuracy = self.learn_right_logits(generated_samples, 
                                                                                    valence_mask_probs, 
                                                                                    target_ids,
                                                                                    loss_fn=self.train_configs.right_loss_fn)
                
               else:
                    if self.right_classification_accuracy > 0.95:
                        self.left_loss, self.left_perplexity, self.left_accuracy = self.learn_left_logits(logits, 
                                                                                           target_ids, 
                                                                                           generated_samples=generated_samples,
                                                                                           valence_mask=valence_mask, 
                                                                                           loss_fn=self.train_configs.left_loss_fn)
                    else:
                        self.left_loss, self.left_perplexity, self.left_accuracy = -.01,-.01, -.01

                    
                    right_loss, right_classification_accuracy = self.learn_right_logits(generated_samples, 
                                                                                    valence_mask_probs, 
                                                                                    target_ids,
                                                                                    loss_fn=self.train_configs.right_loss_fn)
                   
                    
                    
                    # # We may want to do learn_right_logits and learn_left_logits using ThreadPoolExecutor to run the functions in parallel:
                    # with concurrent.futures.ThreadPoolExecutor() as executor:
                    #     # Submit tasks to the executor correctly
                    #     future_right = executor.submit(
                    #         self.learn_right_logits,  # Pass the function itself without calling it
                    #         generated_samples,
                    #         valence_mask_probs,
                    #         target_ids,
                    #         loss_fn=self.train_configs.right_loss_fn
                    #     )
                        
                    #     future_left = executor.submit(
                    #         self.learn_left_logits,  # Pass the function itself without calling it
                    #         logits,
                    #         target_ids,
                    #         generated_samples,  # Ensure arguments are in the correct order
                    #         valence_mask,
                    #         loss_fn=self.train_configs.left_loss_fn
                    #     )
                        
                    #     # Wait for both tasks to complete and get the results
                    #     left_result = future_left.result()
                    #     right_result = future_right.result()
                    
                    # # Unpack the results
                    # left_loss, left_perplexity, left_accuracy = left_result
                    # right_loss, right_classification_accuracy = right_result

                
            for i in range(valence_mask.shape[0]):
                
                total_valence = torch.sum(valence_mask[i,...]).item() 
                sample = generated_samples[i].to(self.left_device_map["embed_in"])
                valence_mask_fixed = valence_mask[i].to(self.right_device_map["embed_out"])
                
                # # remove the eos at the end and prepend a bos token to "restart the process"
                # if generated_samples[i,0].item() != self.tokenizer.bos_token_id:
                    
                #     sample = torch.cat((torch.tensor([self.tokenizer.bos_token_id]).to(self.left_device_map["embed_in"]), 
                #                         sample[:-1]),  dim=-1)
                    
                #     valence_mask_fixed = torch.cat((torch.tensor([1.0]).to(self.right_device_map["embed_out"]), 
                #                                     valence_mask_fixed[:-1]), dim=-1)
                
                
                IDL_ids[IDL_iterr].append( {"valence_mask":valence_mask_fixed.unsqueeze(0),
                                        "IDL_ids":sample.unsqueeze(0),
                                        "IDL_string":self.tokenizer._decode(sample.tolist()),
                                        "total_valence":total_valence
                                        })
                                        
                if self.verbose:
                    print(f"IDL count {IDL_iterr}, sample count {i}: total_valence: {total_valence} on : { valence_mask.shape[-1] } IDL: ", end='')
                    print_colored_text(valence_mask[i,...], sample.tolist(), self.tokenizer)
                    print("------------------------------------")

            # find best valence trajectory so far, and continue on from there
            
            best_sample = max(IDL_ids[IDL_iterr], key = lambda x: x["total_valence"])
            
            best_valence_mask = best_sample["valence_mask"] #tensor
            best_IDL_ids = best_sample["IDL_ids"]  #tensor
            best_IDL_string = best_sample["IDL_string"]
            best_total_valence = best_sample["total_valence"]
            
            if not self._stopping_criteria(IDL_iterr,best_valence_mask):
                # score is high enough or we have hit our limit
                break
            
            # Forward pass left model and generate samples from them
            logits = self._forward_left(best_IDL_ids, best_valence_mask, attention_mask, zero_out_bit_flag=None)
            generated_samples = self._generate_samples_from_logits(logits)
            
            IDL_iterr += 1
          except RuntimeError as e:
            print(f"Error in fpass updating weights: {e}")
            print("continuing....")
            raise Exception(e)
            
        self.IDL_ids = IDL_ids
        return self.IDL_ids

    def learn_left_logits(self, logits, target_ids, generated_samples=None, valence_mask=None, loss_fn="CE"): 

        
        batch_size, seq_len, vocab_size = logits.shape
        target_ids_expanded = target_ids.expand(batch_size, -1)
        
        shifted_targets = torch.zeros_like(logits[:,:,0])
        shifted_targets[..., :-1] = target_ids_expanded[..., 1:]
        shifted_targets[..., -1] = self.tokenizer.eos_token_id
        shifted_targets = shifted_targets.to(self.left_device_map["embed_out"])
        
        if loss_fn=="CE":
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1).long())
            
        elif loss_fn=="PG":
            probs = torch.softmax(logits / self.temperature, dim=-1)
            probs = probs.repeat(generated_samples.size(0), 1, 1)
            device = probs.device
            
            loss = self.compute_policy_gradients(probs, generated_samples.to(device), valence_mask.clone().detach().to(device))
            
        elif loss_fn=="RLOO":
            probs = torch.softmax(logits / self.temperature, dim=-1)
            probs = probs.repeat(generated_samples.size(0), 1, 1)
            device = probs.device
            valence_mask_cloned = (valence_mask.clone().detach().to(device) - self.valence_input_baseline)*self.valence_input_alpha
            
            loss = self.compute_policy_gradients_with_rloo(probs, generated_samples, valence_mask_cloned)
        elif loss_fn=="RLOO+CE":
            probs = torch.softmax(logits / self.temperature, dim=-1)
            probs = probs.repeat(generated_samples.size(0), 1, 1)
            device = probs.device
            valence_mask_cloned = (valence_mask.clone().detach().to(device) - self.valence_input_baseline)*self.valence_input_alpha

            criterion = nn.CrossEntropyLoss()
            loss1 = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1).long())

            loss2 = self.compute_policy_gradients_with_rloo(probs, generated_samples, valence_mask_cloned)

            loss = loss1 + loss2 * 10
            
        else:
            raise Exception(f"Invalid left loss_fn = {loss_fn}, please choose a correct one or implement this loss.")


        # TODO: IMPLEMENT Policy Gradient, RLOO, Focal loss, etc... 
        
        # Backward pass
        loss.backward()
        
        # Calculate metrics
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == shifted_targets).float()
        accuracy = (correct.sum() / correct.numel()).item()
        perplexity = torch.exp(loss).item()
        loss = loss.item()
        if self.verbose:# Logging
            print(f" LEFT Loss: {loss}", end='')
            print(f" LEFT Accuracy: {accuracy * 100:.2f}%", end='')
            print(f" LEFT Perplexity: {perplexity}", end='')
            print(f" LEFT Learning Rate: {self.left_scheduler.get_last_lr()[0]}")
           
            # for i in range(batch):
            #     print(f" Prediction: {self.tokenizer.decode(predictions[i,...], skip_special_tokens=False)}")
            #     print("")
            #     print("")
            print("-----"*50)

        self.left_loss = loss
        self.left_perplexity = perplexity
        self.left_accuracy = accuracy
        return loss, perplexity, accuracy


    
    def learn_left_with_forward(self, IDL_ids, target_ids): 

        input_ids = torch.cat([d["IDL_ids"].to(self.left_device_map["embed_in"]) for sub_list in IDL_ids.values() for d in sub_list])
        valence_masks = torch.cat([d["valence_mask"].to(self.left_device_map["embed_in"]) for sub_list in IDL_ids.values() for d in sub_list])
        attention_mask = torch.ones_like(input_ids).to(self.left_device_map["embed_in"])
        
        # fpass the model on all IDLs
        logits = self._forward_left(input_ids, valence_masks, attention_mask, zero_out_bit_flag=None)

        assert input_ids.shape[-1] == target_ids.shape[-1]

        loss, perplexity, accuracy = self.learn_left_logits(logits, 
                                                            target_ids, 
                                                            generated_samples=input_ids, 
                                                            valence_mask=valence_masks,  
                                                            loss_fn=self.train_configs.left_loss_fn)
        
        return loss, perplexity, accuracy


    
    def learn_right_logits(self, input_ids, valence_probs, target_ids, loss_fn="CE"):# loss_fn = [CE,Focal,RLOO,PG,etc..]):

        
        batch_size, seq_len, _ = valence_probs.shape 
        target_ids_expanded = target_ids.expand(batch_size, -1)

        input_ids = input_ids.to(self.right_device_map["valence_layer"])
        target_ids_expanded = target_ids_expanded.to(self.right_device_map["valence_layer"])
        valence_probs = valence_probs.to(self.right_device_map["valence_layer"])
        target_valences_expanded = (input_ids == target_ids_expanded).int()
        
        if self.train_configs.monotonic_negative_reward:
            target_valences_expanded *= target_valences_expanded.cumprod(dim=1)
            
        if self.verbose:
            print(input_ids)
            print(target_ids_expanded)
            print(target_valences_expanded)
            print(torch.sum(target_valences_expanded))
            print(target_valences_expanded.shape)
        
        assert target_valences_expanded.shape[0] == valence_probs.shape[0]
        assert target_valences_expanded.shape[1] == valence_probs.shape[1]

        only_take_first_two = True
        
        if only_take_first_two and valence_probs.shape[0]>2:
            valence_probs = valence_probs[2,...]
            target_valences_expanded = target_valences_expanded[2,...]
            
        valence_probs_flat = valence_probs.reshape(-1, valence_probs.size(-1))  # [batch*seq, 2]
        target_valences_expanded_flat = target_valences_expanded.reshape(-1).long()    # [batch*seq]
            
        if loss_fn=="Focal":
            # Calculate focal loss
            targets_one_hot = F.one_hot(target_valences_expanded_flat, num_classes=2).float()
            loss = -1 * sigmoid_focal_loss(valence_probs_flat, targets_one_hot, alpha=2.0, gamma=4.0, reduction='mean')                
        elif loss_fn=="CE":
            criterion = nn.CrossEntropyLoss()
            
            loss = criterion(valence_probs_flat, target_valences_expanded_flat)

        else:
            raise Exception(f"Invalid right loss function loss_fn {loss_fn}, please fix or implement.")
        
        # Backward pass
        loss.backward( )
        
        # Calculate metrics
        predictions = torch.argmax(valence_probs_flat, dim=-1)
        correct = (predictions == target_valences_expanded_flat).float()
        accuracy = (correct.sum() / correct.numel()).item()
        loss = loss.item()
        if self.verbose:# Logging
            print(f" RIGHT Loss: {loss}", end='')
            print(f" RIGHT Accuracy: {accuracy * 100:.2f}%", end='')
            print(f" RIGHT Learning Rate: {self.right_scheduler.get_last_lr()[0]}")
            
            # for i in range(batch):
            #     print(f" Prediction: {self.tokenizer.decode(predictions[i,...], skip_special_tokens=False)}")
            #     print("")
            #     print("")
            print("-----"*50)

        self.right_loss = loss
        self.right_classification_accuracy = accuracy
        
        return loss, accuracy


    
    def learn_right_with_forward(self, IDL_ids, target_ids):# loss_fn = [CE,Focal,RLOO,PG,etc..]):
        #	 - loop over all IDLs, calc loss, and grad accumulate for right model
        #	 - return metrics 
        #		 - loss
        #		 - classification_accuracy
        
        input_ids = torch.cat([d["IDL_ids"].to(self.right_device_map["embed_in"]) for sub_list in IDL_ids.values() for d in sub_list])
        
        attention_mask = torch.ones_like(input_ids).to(self.right_device_map["embed_in"])

        target_ids = target_ids.to(self.right_device_map["embed_in"])

        # fpass the model on all IDLs to get the valence_masks 
        valence_probs = self._forward_right(input_ids, attention_mask) # [batch, seq, 2]

        loss, accuracy = self.learn_right_logits(input_ids, valence_probs, target_ids, loss_fn=self.train_configs.right_loss_fn)
        
        return loss, accuracy


    
    def update_weights_left(self, total_loss):
        # optim.step for both left / right
        
        normed_grad = torch.nn.utils.clip_grad_norm_(self.current_left_model.parameters(), max_norm=1.).item()

        self.optimizer_left.step()
        self.left_scheduler.step()
        self.optimizer_left.zero_grad()
        return normed_grad
        
    def update_weights_right(self, total_loss):
        # optim.step for both left / right
        
        normed_grad = torch.nn.utils.clip_grad_norm_(self.current_right_model.parameters(), max_norm=1.).item()
        normed_grad_r = torch.nn.utils.clip_grad_norm_(self.valence_layer.parameters(), max_norm=1.).item()
        self.optimizer_right.step()
        self.right_scheduler.step()
        self.optimizer_right.zero_grad()
        
        return normed_grad + normed_grad_r
        
    def save_checkpoints(self,iterr,left_loss,right_loss, left_model_directory, right_model_directory, return_names = False):
        
        # save left checkpoint
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_filename = f"left_checkpoint_{timestamp}_iter_{iterr}_loss_{left_loss:.2f}.pth"
        left_checkpoint_filepath = os.path.join(left_model_directory, checkpoint_filename)
        torch.save({
            'model_state_dict': self.current_left_model.state_dict(),
            'optimizer_state_dict': self.optimizer_left.state_dict(),
            'loss': left_loss,
        }, left_checkpoint_filepath)
        print(f"Checkpoint saved at {left_checkpoint_filepath}")

        
        # save right checkpoint
        checkpoint_filename = f"right_checkpoint_{timestamp}_iter_{iterr}_loss_{right_loss:.2f}.pth"
        right_checkpoint_filepath = os.path.join(right_model_directory, checkpoint_filename)
        torch.save({
            'model_state_dict': self.current_right_model.state_dict(),
            'valence_layer_state_dict': self.valence_layer.state_dict(),
            'optimizer_state_dict': self.optimizer_right.state_dict(),
            'loss': right_loss,
        }, right_checkpoint_filepath)
        print(f"Checkpoint saved at {right_checkpoint_filepath}")
        if return_names:
            return left_checkpoint_filepath, right_checkpoint_filepath

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
    
