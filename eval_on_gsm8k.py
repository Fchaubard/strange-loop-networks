
import os
import re
import json
import sys
sys.path.append('.')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sln_v3 import SLN
import glob
import torch
import pdb

num_gpus = torch.cuda.device_count()

if num_gpus<=4:
    raise(f"This script is not ready to handle only 4 GPUs, only {num_gpus} GPUs detected. Please use >4.")

left_model_device_list = [f"cuda:{i}" for i in range(int(num_gpus/2))]
right_model_device_list = [f"cuda:{i}" for i in range(int(num_gpus/2),num_gpus)]
    

# Load the dataset
dataset = load_dataset('openai/gsm8k', 'main')
dataset = dataset['test']

print("Booting consciousness... one sec.. :)")


# left_model_checkpoint = "/left_checkpoints/left_checkpoint_20240715173212_iter_800_loss_37.63.pth" #"<path to right model checkpoint>"
# right_model_checkpoint = "/right_checkpoints/right_checkpoint_20240715155144_iter_10_loss_6.31.pth"#"<path to left model checkpoint>"

# Directory containing the checkpoint files
right_directory = "/right_checkpoints/"

# Get list of all checkpoint files in the directory
files = glob.glob(os.path.join(right_directory, "right_checkpoint_*.pth"))

# Find the most recent file based on modification time
right_model_checkpoint = max(files, key=os.path.getmtime)


# Directory containing the checkpoint files
left_directory = "/left_checkpoints/"

# Get list of all checkpoint files in the directory
files = glob.glob(os.path.join(left_directory, "left_checkpoint_*.pth"))

# Find the most recent file based on modification time
left_model_checkpoint = max(files, key=os.path.getmtime)

print(left_model_checkpoint,right_model_checkpoint)

sln = SLN("EleutherAI/pythia-410m",
     left_model_device_list,
     right_model_device_list,
     left_model_checkpoint_name=left_model_checkpoint, 
     right_model_checkpoint_name=right_model_checkpoint, 
     verbose=False, 
     trajectories_per_IDL=10,
     temperature=1.0,
     train_configs = None
    )

# sln = SLN(right_model_checkpoint, 
#           left_model_checkpoint, 
#           return_type1_answer=False, 
#           return_highest_valence=True, 
#           return_all_IDLs=False,
#           round_valence=True,
#           left_model_device="cuda:2",
#           right_model_device="cuda:2",
#           verbose=True)

def extract_answer(text):
    """Extract the final numeric solution from the model's text output."""
    match = re.search(r'#### (\d+(\.\d+)?)', text)
    return match.group(1) if match else None

# Evaluate the model
correct = 0
total = len(dataset)
for i, sample in enumerate(dataset):
  
    print("="*50)
    
    prompt = sln.special_tokens_dict["bos_token"] + sample["question"]
    target_response = sln.left_model_token + sample["answer"] + sln.special_tokens_dict["eos_token"]
    
    full_target = prompt + target_response

    max_len_target_response_ids = sln.tokenizer(target_response, 
                                    return_tensors='pt', 
                                    padding=True, 
                                    truncation=True, 
                                    add_special_tokens=True).input_ids.shape[-1]
    
    # target output of sln
    full_target_ids = sln.tokenizer(full_target, 
                                    return_tensors='pt', 
                                    padding=True, 
                                    truncation=True, 
                                    add_special_tokens=True).input_ids.to(sln.left_model_device_list[0])


    # input into sln
    full_target_ids[-(max_len_target_response_ids+1):] = sln.tokenizer.mask_token_id
    

    # Extract the correct answer
    true_answer = extract_answer(target_response)
    print(f'true_answer {true_answer}')
    # Generate prediction
    IDL_ids = sln.forward(full_target_ids)
    flattened_list = [item for sublist in IDL_ids.values() for item in sublist]
    best_sample = max(flattened_list, key = lambda x: x["total_valence"])
            
    best_valence_mask = best_sample["valence_mask"] #tensor
    best_IDL_ids = best_sample["IDL_ids"]  #tensor
    best_IDL_string = best_sample["IDL_string"]
    best_total_valence = best_sample["total_valence"]
            
    predicted_answer = extract_answer(best_IDL_string)
    print(f'predicted_answer {predicted_answer}')
    if predicted_answer is None:
        print(best_IDL_string)
    # Evaluate the prediction
    if predicted_answer == true_answer:
        print('Got it correct!')
        correct += 1

    if (i + 1) % 1 == 0:
        print(f"Processed {i + 1.}/{total} samples, score so far: {correct*1.0/(i+1)}")

# Report the results
accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
