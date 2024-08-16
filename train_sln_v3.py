# sln_v3.py 

import torch
from transformers import GPTNeoXModel, GPTNeoXForCausalLM, GPTNeoXConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
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
import subprocess
from torchvision.ops import sigmoid_focal_loss
import sys
from torch.optim.lr_scheduler import LambdaLR
import math
from datasets import load_dataset
from types import SimpleNamespace
import concurrent
import wandb
import os
import traceback
sys.path.append('.')
sys.path.append(os.path.abspath(os.path.join('..', 'sentence_augs')))
from text_corrupter import text_corrupter_negative, generate_match_mask

from print_colored_text import print_colored_text
from sln_v3 import SLN

os.environ["WANDB_API_KEY"] = "cce47709d839921f0b13533529f31c8af7f3f4dc"

        


import torch
import random
import numpy as np

class ProgressiveTailMasking:
    def __init__(self, vocab_size, accuracy_threshold, cooldown, mask_token_id, random_replacement="RandomSample"):
        self.vocab_size = vocab_size
        self.accuracy_threshold = accuracy_threshold
        self.cooldown = cooldown
        self.n = 5
        self.mask_token_id = mask_token_id
        self.random_replacement = random_replacement  # could be "Mask", "RandomToken", or "RandomSample" methods
        self.accuracy_vector = []

    def mask_tokens(self, output_tokens, true_answer_len, replacement_target_response=None):
        input_tokens = output_tokens.clone()
        mask_amount = self.n

        if mask_amount > true_answer_len:
            mask_amount = true_answer_len

        if 0 < self.n:
            if self.random_replacement == "RandomToken":
                random_tokens = torch.tensor(random.choices(range(self.vocab_size), k=mask_amount)).to(output_tokens.device)
                input_tokens[..., -mask_amount:] = random_tokens  # Masking the rightmost n tokens

            elif self.random_replacement == "RandomSample":
                if replacement_target_response is None:
                    raise Exception("if using random_replacement=RandomSample strategy, replacement_target_response must be populated")
                replacement_tokens = replacement_target_response[..., -mask_amount:]
                input_tokens[..., -mask_amount:] = replacement_tokens  # Replace with tokens from another sample

            elif self.random_replacement == "Mask":  # Default "Mask" method
                input_tokens[..., -mask_amount:] = self.mask_token_id  # Masking the rightmost n tokens
            else:
                raise Exception("Not a valid self.random_replacement strategy: " + str(self.random_replacement))
        return input_tokens

    def update_n(self, current_accuracy):
        self.accuracy_vector.append(current_accuracy)
        if len(self.accuracy_vector) > self.cooldown:
            if np.mean(self.accuracy_vector[-self.cooldown:]) >= self.accuracy_threshold:
                self.n += 1
                self.accuracy_vector = []
                print(f"Updating ProgressiveTailMasking to {self.n}")

    def __call__(self, output_tokens, current_accuracy, true_answer_len, replacement_target_response=None):
        input_tokens = self.mask_tokens(output_tokens, true_answer_len, replacement_target_response=replacement_target_response)
        self.update_n(current_accuracy)
        
        return input_tokens




if __name__ == '__main__':
    
    #########
    # TRAINING CONFIG
    #########
    # Path to the JSON file
    training_interaction_file = './training_interaction_file.json'

    train_configs = {}
    train_configs["left_lr"] = 1e-8
    train_configs["right_lr"] = 1e-5
    train_configs["weight_decay"] = 0.0001
    train_configs["left_betas"] = (0.7, 0.9)
    train_configs["right_betas"] = (0.99, 0.999)
    

    # for scheduler
    train_configs["factor"]=0.5
    train_configs["patience"]=250
    train_configs["cooldown"]=250

    train_configs["macro_batch_size"] = 30
    train_configs["max_microbatch_size"] = 2 # IMPORTANT TO INCREASE IF YOU HAVE MORE GPU RAM
    train_configs["max_ctx_len"] = 3000 # IMPORTANT TO INCREASE IF YOU HAVE MORE GPU RAM
  
    train_configs["save_checkpoint_every_n_batches"] = 500

    # for handling the valence input signal
    train_configs["valence_input_baseline"] = 0.5
    train_configs["valence_input_alpha"] = 2.
    train_configs["reset_lr"] = True

    # 410m, 1b, 1.4b, 2.8b, 6.9b, 12b
    train_configs["model_id"] = "EleutherAI/pythia-410m" 
    wandb_project_name = "sln_training_pythia_RLOO_focal_"+train_configs["model_id"].replace("/","_") + "_"

    train_configs["start_from_raw"] = False # do not start from most recent checkpoints
    train_configs["trajectories_per_IDL"]=3
    train_configs["IDL_limit"]=1
    train_configs["temperature"]=1.0
    train_configs["monotonic_negative_reward"] = True

    train_configs["ptm_accuracy_threshold"] = 0.9  # Example accuracy threshold
    train_configs["ptm_cooldown"] = 100  # Example cooldown period
    train_configs["ptm_random_replacement"] = "RandomSample" #"RandomSample", "RandomToken", "Mask"
 

    train_configs['total_steps'] = 10000
    train_configs['warmup_steps'] = 100

    train_configs['left_loss_fn'] = "RLOO" # "RLOO+CE","RLOO", "PG", "CE"
    train_configs['right_loss_fn'] = "Focal" # "Focal", "CE"

    train_configs["checkpoint_every_n_iterrs"] = 3000
    train_configs["baseline_reward_for_rl_loss"] = 1
    
    decode_every_n_batches = 100
    include_wandb = True

    left_model_directory = "/left_checkpoints/" 
    right_model_directory = "/right_checkpoints/" 


    #########
    # SLN SETUP
    #########
    train_configs = SimpleNamespace(**train_configs)
    num_gpus = torch.cuda.device_count()
    
    if num_gpus<=4:
        raise(f"This script is not ready to handle only 4 GPUs, only {num_gpus} GPUs detected. Please use >4.")
        
    left_model_device_list = [f"cuda:{i}" for i in range(int(num_gpus/2))]
    right_model_device_list = [f"cuda:{i}" for i in range(int(num_gpus/2),num_gpus)]
    
    left_model_checkpoint_name = None
    right_model_checkpoint_name = None
    
    # - load from checkpoint if not from raw
    if not train_configs.start_from_raw:
        files = glob.glob(os.path.join(left_model_directory, "left_checkpoint_*.pth"))
        if len(files)==0:
            raise(f"No files in {left_model_directory}")
        left_model_checkpoint_name = max(files, key=os.path.getmtime)
        
        files = glob.glob(os.path.join(right_model_directory, "right_checkpoint_*.pth"))
        
        if len(files)==0:
            raise(f"No files in {right_model_directory}")

        right_model_checkpoint_name = max(files, key=os.path.getmtime)

    sln = SLN(train_configs.model_id,
         left_model_device_list,
         right_model_device_list,
         left_model_checkpoint_name=left_model_checkpoint_name, 
         right_model_checkpoint_name=right_model_checkpoint_name, 
         verbose=False, 
         IDL_limit = train_configs.IDL_limit,
         trajectories_per_IDL=train_configs.trajectories_per_IDL,
         temperature=train_configs.temperature,
         train_configs = train_configs
        )
    
    #------------------
    # DO ENVIRONMENT SETUP:
    #------------------
    
    
    ptm_masking_scheduler = ProgressiveTailMasking(sln.tokenizer.vocab_size, 
                                         train_configs.ptm_accuracy_threshold, 
                                         train_configs.ptm_cooldown, 
                                         sln.tokenizer.mask_token_id,
                                         random_replacement=train_configs.ptm_random_replacement)
    
    # Check if the left_model_directory exists
    if not os.path.exists(left_model_directory):
        # If the directory does not exist, create it
        os.mkdir(left_model_directory)
        print(f"Directory {left_model_directory} doesnt exit! Making it.")
    else:
        print(f"Directory {left_model_directory} exists.")


    # Check if the right_model_directory exists
    if not os.path.exists(right_model_directory):
        # If the directory does not exist, create it
        os.mkdir(right_model_directory)
        print(f"Directory {right_model_directory} doesnt exit! Making it.")
    else:
        print(f"Directory {right_model_directory} exists.")
    
    if include_wandb:
        wandb.init(project=wandb_project_name)

    print("Training with configs:")
    print(train_configs)

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset_size = len(dataset)
    
    iterr=-1
    accuracy = 0.0
    
    ####################
    # Start Training!
    ####################
    print("STARTING TRAINING")
    
    while True:
      try:
        iterr+=1
        # load a prompt and target from the dataset
        sample = dataset[iterr%dataset_size]
        # sample = dataset[4483]
        # sample = dataset[0]
        
        prompt = sln.special_tokens_dict["bos_token"] + sample["question"]
        target_response = sln.left_model_token + sample["answer"] + sln.special_tokens_dict["eos_token"] # WILL BE ADDED TO THE SHIFTED TARGET. ?? TODO
        
        full_target = prompt + target_response

        max_len_target_response_ids = sln.tokenizer(target_response, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True, 
                                        add_special_tokens=True).input_ids.shape[-1] - 1 # subtract one for the <left_model> token
        
        # target output of sln
        full_target_ids = sln.tokenizer(full_target, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True, 
                                        add_special_tokens=True).input_ids.to(sln.left_model_device_list[0])


        # input into sln
        if train_configs.ptm_random_replacement == "RandomSample":
            dataset_size = len(dataset)
            random_sample = random.choice([i for i in range(dataset_size) if i != iterr])
            
            # Get the replacement target response from the random sample
            replacement_sample = dataset[random_sample]
            replacement_target_response = sln.left_model_token + replacement_sample["answer"] + sln.special_tokens_dict["eos_token"]
            
            # Tokenize the replacement target response
            replacement_target_response_ids = sln.tokenizer(replacement_target_response, 
                                                        return_tensors='pt', 
                                                        padding=True, 
                                                        truncation=True, 
                                                        add_special_tokens=True).input_ids.to(sln.left_model_device_list[0])
        else:
            replacement_target_response_ids = None
        
        # Call the ProgressiveTailMasking scheduler
        progressive_tail_masked_input_ids = ptm_masking_scheduler(full_target_ids, 
                                                                  accuracy, 
                                                                  max_len_target_response_ids,  
                                                                  replacement_target_response=replacement_target_response_ids)
          
        assert full_target_ids.shape == progressive_tail_masked_input_ids.shape

        # fpass the sln, and get back all IDLs. Keep it in token form so we can easily calc loss
        IDL_ids = sln.forward(progressive_tail_masked_input_ids, target_ids=full_target_ids)
        
        # left_loss, left_perplexity, left_accuracy = sln.learn_left_with_forward(IDL_ids, full_target_ids)
        # right_loss, right_classification_accuracy = sln.learn_right_with_forward(IDL_ids, full_target_ids)
                    
        accuracy = sln.left_accuracy # I care more that we are improving vs. knowing that we are not.. 
        
        try:
            with open(training_interaction_file, 'r', encoding='utf-8') as file:
                content = file.read().strip()  # Strip any leading/trailing whitespace
                if content:  # Check if the file is not empty
                    data = json.loads(content)
                    # Update the verbose variable if the key exists and is a boolean
                    if 'verbose' in data and isinstance(data['verbose'], bool) and data['verbose'] != sln.verbose:
                        sln.verbose = data['verbose']
                        print(f"RESETTING VERBOSITY to : {data['verbose']}")
                        print("Training with configs:")
                        print(train_configs)

        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"An error occurred trying to update from training_interaction_file.json: {e}")
        
        if iterr % train_configs.macro_batch_size:
            if sln.right_classification_accuracy > 0.8:
                normed_grad_left = sln.update_weights_left(sln.left_loss)
            else:
                normed_grad_left = -1
            normed_grad_right = sln.update_weights_right(sln.right_loss)
                    
            message = {
                       "ptm_masking_scheduler_n": ptm_masking_scheduler.n,
                       "left_loss": round(float(sln.left_loss),5), 
                       "right_loss": round(float(sln.right_loss),5), 
                       "left_accuracy": round(float(sln.left_accuracy),5), 
                       "right_accuracy": round(float(sln.right_classification_accuracy),5), 
                       "left_perplexity": round(float(sln.left_perplexity),5), 
                       "left_lr": sln.optimizer_left.param_groups[0]['lr'],  
                       "right_lr": sln.optimizer_right.param_groups[0]['lr'],  
                       "left_normed_grad": round(float(normed_grad_left),5),  
                       "right_normed_grad": round(float(normed_grad_right),5),
                        "iterr":iterr,
                   }

            print(message)
            
            if include_wandb:
                wandb.log(message)
            
            
            
        if iterr % train_configs.checkpoint_every_n_iterrs == train_configs.checkpoint_every_n_iterrs-1:
            sln.save_checkpoints(iterr, sln.left_loss, sln.right_loss, left_model_directory, right_model_directory)
            print(train_configs)
        
        
      except Exception as e:
          
          print("ERROR!!!!!")
          print("---")
          traceback.print_exc()
          # print("---")
          # print(torch.cuda.memory_summary())
          # print("---")
          # result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
          # print(result.stdout)
          # print("---")
          # print(f"iterr:{iterr} full_target:{full_target}")

          # print("saving checkpoints")
          # left_model_checkpoint_name, right_model_checkpoint_name = sln.save_checkpoints(iterr,
          #                                                                               left_loss,right_loss, 
          #                                                                               left_model_directory, 
          #                                                                               right_model_directory,
          #                                                                               return_names = True)
          
          # print("saved! Reloading...")
          # sln = SLN(train_configs.model_id,
          #    left_model_device_list,
          #    right_model_device_list,
          #    left_model_checkpoint_name=left_model_checkpoint_name, 
          #    right_model_checkpoint_name=right_model_checkpoint_name, 
          #    verbose=False, 
          #    trajectories_per_IDL=train_configs.trajectories_per_IDL,
          #    temperature=train_configs.temperature,
          #    train_configs = train_configs
          #   )
          
          # print("reloaded!")
          pdb.set_trace()
          raise Exception(e)
          # continue
      
          






        
    
    
    
