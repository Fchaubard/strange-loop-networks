
import os
import re
import json
import sys
sys.path.append('.')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sln_v2 import SLN
import glob




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

sln = SLN(right_model_checkpoint, 
          left_model_checkpoint, 
          return_type1_answer=False, 
          return_highest_valence=True, 
          return_all_IDLs=False,
          round_valence=True,
          left_model_device="cuda:2",
          right_model_device="cuda:2",
          verbose=True)

def extract_answer(text):
    """Extract the final numeric solution from the model's text output."""
    match = re.search(r'#### (\d+(\.\d+)?)', text)
    return match.group(1) if match else None

# Evaluate the model
correct = 0
total = len(dataset)
for i, sample in enumerate(dataset):
  
    print("="*50)
    question = sample['question']
    answer = sample['answer']

    # Extract the correct answer
    true_answer = extract_answer(answer)
    print(f'true_answer {true_answer}')
    # Generate prediction
    generated_text = sln.forward(question)
    predicted_answer = extract_answer(generated_text)
    print(f'predicted_answer {predicted_answer}')
    if predicted_answer is None:
        print(generated_text)
    # Evaluate the prediction
    if predicted_answer == true_answer:
        print('got it correct')
        correct += 1

    if (i + 1) % 1 == 0:
        print(f"Processed {i + 1}/{total} samples, score so far: {correct*1.0/(i+1)}")

# Report the results
accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
