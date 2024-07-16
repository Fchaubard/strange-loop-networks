# batch_generator.py 
# - creates batches in a while True loop, with positive / negative pairs and puts them in ./batches/ folder for train loops to pick up
# - if ./batches/ doesnt exist, create it. If it does exist, add more to it.
# - format: batch_timestamp_<left_model_checkpoint_name>.pckl
# -         ({'input_text':input_text, 'true_answer':true_answer, 'valence_mask':valence_mask})
# - will need to pickup most recent (best performing) left_model from checkpoint every once in a while.
import os
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import pdb
import datetime
sys.path.append('.')
from sln_v2 import SLN

# UPDATE THIS TO WHERE YOUR https://github.com/Fchaubard/sentence_augs.git is cloned to:
sys.path.append(os.path.abspath(os.path.join('..', 'sentence_augs')))
from text_corrupter import text_corrupter_negative, generate_match_mask
from timeout_decorator import timeout, TimeoutError


# @timeout(60*3)  # Set a timeout of n seconds for this function
def generate_random_valence_batch(list_of_programs, 
                                 sln,
                                 number_of_programs_to_sample, 
                                  samples_per_program=2, 
                                 num_corruptions=2, 
                                 max_length_left=100, 
                                 left_model_sep_tok="<left model>",
                                 batch_size = 5
                                ):
    # POS: Input - Plausible Augs of Truth Output
    #     Pick # programs = batch_size/runs_per_program random programs randomly, 
    #     generate 5 example from each. If we do this, hopefully it will not overfit to one, 
    #     but still get the ‘gist’ of what its supposed to do in each one.
    
    #     Input = question + “<left model>” + true answer
    #     Target = [1,1,1,1,1,1,1]
    # NEGS: Input  - Implausible Augs of Truth Output
    #     String_corrupted = text_corrupter_negative(true answer)
    #     Input = question + “<left model>” + string_corrupted
    #     Target = generate_match_mask(true answer, string_corrupted)
    # NEGS: Input - <left model> - Left Model Output 
    #     Then try the left model on each question, and see what it outputs. Append that to the example input of the model
    #     Input = question + “<left model>” + T0 from left model
    #     valence = [0,0,0,0,0,0,0] (unless T0 from left is correct at all, then valence = generate_match_mask(left model answer,true answer))
    # return batch = list of (input='blah blah',targets='[0,1,0,1,0,0,0,1,1,1 ... ]') with length b/2 of positives + b/2 of negatives

    batch = []
    sampled_programs = []
    failure_counter = 0
    while failure_counter<100 and len(sampled_programs) <= number_of_programs_to_sample:

            # choose a random 'program (math problem)' to sample from
            program = random.choice(list_of_programs)
            
            # for each sampled_program, generate samples_per_program samples from it to gen positive_samples with targets = [1,1,1,1..]
            try:
                exec(program, globals())
                positive_samples = [generate_math_problem() for _ in range(samples_per_program)]

                # Generate positives:
                for positive_sample in positive_samples:

                    question = positive_sample['question']
                    true_answer = positive_sample['answer']

                    input_text = question + " "+left_model_sep_tok+" " + true_answer
                    valence_mask = [1 for _ in range(len(sln.tokenizer.tokenize(input_text)))]

                    batch.append({'input_text':input_text, 'true_answer':true_answer, 'valence_mask':valence_mask})
                    all_IDLs = sln.forward(question)
                    for dd in all_IDLs:
                        
                        IDL_count = dd["IDL_count"]
                        left_model_response = dd["left_model_response"]
                        right_model_valence = dd["right_model_valence"]

                        left_model_input_text = question + " "+left_model_sep_tok+" " +left_model_response
                        
                        valence_mask = generate_match_mask(
                                                            sln.tokenizer, 
                                                            input_text, 
                                                            left_model_input_text
                                                         )
                        batch.append({'input_text':left_model_input_text, 'true_answer':true_answer, 'valence_mask':valence_mask})

                sampled_programs.append(program)
            except Exception as e:
                output = f"!!!!!!!!!!! Error executing the generated program: {e}"
                print(output)
                print(program)
                print(failure_counter)
                print("="*50)
                failure_counter+=1
                continue
                
    return batch

if __name__ == '__main__':
 
    #------------------
    # THINGS TO UPDATE:
    #------------------
    # Define the gsm8k file path... or if you want, create more programs somewhere else.
    gsm_file_path = '../sentence_augs/gsm8k_train_programs.txt'
    
    batches_directory = "./sln_batches/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    number_of_programs_to_sample = 1
    samples_per_program = 1
    
    left_model_checkpoint = "/left_checkpoints/left_checkpoint_20240715173212_iter_800_loss_37.63.pth" #"<path to right model checkpoint>"
    right_model_checkpoint = "/right_checkpoints/right_checkpoint_20240715155144_iter_10_loss_6.31.pth"
    #right_checkpoint_20240709113005_iter_2000_loss_0.52.pth" #"<path to left model checkpoint>"
    
    sln = SLN(right_model_checkpoint, left_model_checkpoint, verbose=False, return_all_IDLs=True)

    
    # Step 1: Check for ./batches/ and if it doesn't exist make the folder.
    # Check if the directory exists
    if not os.path.exists(batches_directory):
        # If the directory does not exist, create it
        os.makedirs(batches_directory)
        print(f"Directory {batches_directory} created.")
    else:
        print(f"Directory {batches_directory} already exists.")
    
    # Step 2: load the programs into RAM
    
    # Define the delimiter for splitting programs
    delimiter = "-----------<this will be used to split on later>----------------"
    
    # Initialize an empty list to store the programs
    list_of_programs = []
    
    # Read the file
    with open(gsm_file_path, 'r') as file:
        # Read the entire content of the file
        content = file.read()
        
        # Split the content based on the delimiter
        list_of_programs = content.split(delimiter)
    
    # Strip any leading or trailing whitespace from each program
    list_of_programs = [program.strip() for program in list_of_programs]
    print('# programs: '+str(len(list_of_programs)))

    # Step 4: create some batches and save them off
    while True:
        # Create the batch:
        batch = generate_random_valence_batch(list_of_programs, 
                                              sln,
                                             number_of_programs_to_sample, 
                                             samples_per_program=samples_per_program)

        # Save it to ./batches/batch_<timestamp>_<left_model_checkpoint_name>.pckl 
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Create the filename
        filename = f"batch_{timestamp}.json"

        # Path to the file
        file_path = os.path.join(batches_directory, filename)

        # Save the batch to the file
        with open(file_path, 'w') as file:
            json.dump(batch, file, indent=4)
        
        print(f"SUCCESS!!!!!! Wrote out file {file_path} of len {len(batch)}")
            
        # randomly output some to monitor whats going on
        if np.random.rand()>0.0:
            dd = random.choice(batch)
            input_text =  dd['input_text']
            true_answer =  dd['true_answer']
            valence_mask =  dd['valence_mask']
            print('='*50)
            print("Randomly sampling to show you whats being produced:")
            print(f"input_text: {input_text}")
            print(f"true_answer: {true_answer}")
            print(f"valence_mask: {valence_mask}")
            print('='*50)
