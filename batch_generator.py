# batch_generator.py 
# - creates batches in a while True loop, with positive / negative pairs and puts them in ./batches/ folder for train loops to pick up
# - if ./batches/ doesnt exist, create it. If it does exist, add more to it.
# - format: batch_timestamp_<left_model_checkpoint_name>.pckl
# -         ({'input_text':input_text, 'true_answer':true_answer, 'reward_mask':reward_mask})
# - will need to pickup most recent (best performing) left_model from checkpoint every once in a while.
import os
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import pdb
from datetime import datetime

# UPDATE THIS TO WHERE YOUR https://github.com/Fchaubard/sentence_augs.git is cloned to:
sys.path.append(os.path.abspath(os.path.join('..', 'sentence_augs')))
from text_corrupter import text_corrupter_negative, generate_match_mask

from timeout_decorator import timeout, TimeoutError

@timeout(60*3)  # Set a timeout of n seconds for this function
def generate_random_reward_batch(list_of_programs, 
                                 number_of_programs_to_sample, 
                                 tokenizer, 
                                 base_model, 
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
    #     reward = [0,0,0,0,0,0,0] (unless T0 from left is correct at all, then reward = generate_match_mask(left model answer,true answer))
    # return batch = list of (input='blah blah',targets='[0,1,0,1,0,0,0,1,1,1 ... ]') with length b/2 of positives + b/2 of negatives

    if samples_per_program < 2 or samples_per_program%2!=0:
        raise Exception('samples_per_program must be >=2 so you can sample enough to prevent overfitting and must be even numbered to ensure balanced mbs', samples_per_program)
    if number_of_programs_to_sample < 1:
        raise Exception('number_of_programs_to_sample must be >=1', number_of_programs_to_sample)
        
#     sampled_programs = random.sample(list_of_programs, number_of_programs_to_sample)
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
                    reward_mask = [1 for _ in range(len(tokenizer.tokenize(input_text)))]

                    batch.append({'input_text':input_text, 'true_answer':true_answer, 'reward_mask':reward_mask})

                # Gen Negatives w/ left_model: for first half of positive samples, 
                # generate left_model_samples by calling model.generate(positive_sample.question) 
                # and then compute targets = generate_match_mask(true answer, string_corrupted) to extend negative_samples
                starting_points_for_negs = positive_samples[:int(len(positive_samples)/2)]

                for i in range(0, len(starting_points_for_negs), batch_size):
                    batch_samples = starting_points_for_negs[i:i + batch_size]

                    # Tokenize the batch
                    input_texts_to_left_model = [sample['question'] for sample in batch_samples]
                    input_tokens = tokenizer(input_texts_to_left_model, return_tensors="pt", padding=True, truncation=True)

                    
                    # Generate output with attention mask
                    model_outputs = base_model.generate(
                        input_tokens.input_ids.to(base_model.device), 
                        attention_mask=input_tokens.attention_mask.to(base_model.device),
                        max_new_tokens=max_length_left,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    # Decode outputs in a batch
                    generated_texts = [tokenizer.decode(output, skip_special_tokens=False) for output in model_outputs]
                    
                    for j, sample in enumerate(batch_samples):
                        left_model_answers = generated_texts[j] #left_model_answers
                        input_text = sample['question'] + " "+left_model_sep_tok+" " + left_model_answers
                        # print("model_outputted negative example:" + input_text)
                        reward_mask = generate_match_mask(
                                                            tokenizer, 
                                                            sample['question'] + " "+left_model_sep_tok+" " +sample['answer'], 
                                                            sample['question'] + " "+left_model_sep_tok+" " +left_model_answers
                                                         )
                        batch.append({'input_text':input_text, 'true_answer':sample['answer'], 'reward_mask':reward_mask})

                # Gen Negatives w/ sentence augs: for first half of positive samples, generate text_corrupted_samples via text_corrupter_negative()
                # with targets = generate_match_mask(true answer, string_corrupted) 
                # to make negative_samples
                for sample in starting_points_for_negs:
                    question = sample['question']
                    true_answer = sample['answer']
                    corrupted_answer = true_answer
                    for _ in range(num_corruptions):
                        corrupted_answer = text_corrupter_negative(corrupted_answer)
                    input_text = question + " "+left_model_sep_tok+" " + corrupted_answer
                    reward_mask = generate_match_mask(tokenizer, true_answer, corrupted_answer)
                    reward_mask = generate_match_mask(
                                    tokenizer, 
                                    sample['question'] + " "+left_model_sep_tok+" " +true_answer, 
                                    sample['question'] + " "+left_model_sep_tok+" " +corrupted_answer
                                 )
                    batch.append({'input_text':input_text, 'true_answer':true_answer, 'reward_mask':reward_mask})
                    
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
    
    # update what device you want to use
    device = 'cuda:0'

    # update what model you want to use WARNING: IF YOU WANT TO START FROM A CHECKPOINT OF LEFT MODEL, THIS IS THE PLACE TO DO IT:
    model_id = "EleutherAI/pythia-70m-v0"
    update_left_model_every_n_batches = 1000 #None if do not ever update, otherwise put a number of batches.
    
    left_model_directory = "./left_checkpoints/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    
    batches_directory = "./batches/" # I WOULD KEEP THIS AS DEFAULT PATTERN FOR SLN TRAINING
    
    pad_tok = '[PAD]'
    left_model_sep_tok = '<left model>'
    # batch_size = number_of_programs_to_sample * samples_per_program * 2    
    number_of_programs_to_sample = 10 
    samples_per_program = 20 
    
    
    
    
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

    # Step 3: grab a tokenizer for reward_mask creation
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # create initial model 
    left_model_checkpoint_name = model_id
    
    current_left_model = AutoModelForCausalLM.from_pretrained(left_model_checkpoint_name)
    current_left_model = current_left_model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_tok})
        current_left_model.config.pad_token_id = tokenizer.pad_token_id
        current_left_model.resize_token_embeddings(len(tokenizer))
        print('setting tokenizer.pad_token_id to: ' + str(tokenizer.pad_token_id) + " with token: "+pad_tok)
        tokenizer.padding_side = "left"

    
    
    batches_created_since_last_model_update = 0 
    # Step 4: create some batches and save them off
    while True:
        # Create the batch:
        batch = generate_random_reward_batch(list_of_programs, 
                                             number_of_programs_to_sample, 
                                             tokenizer, 
                                             current_left_model, 
                                             left_model_sep_tok=left_model_sep_tok,
                                             samples_per_program=samples_per_program)

        # Save it to ./batches/batch_<timestamp>_<left_model_checkpoint_name>.pckl 
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Create the filename
        filename = f"batch_{timestamp}_{left_model_checkpoint_name.replace('/','_')}.json"

        # Path to the file
        file_path = os.path.join(batches_directory, filename)

        # Save the batch to the file
        with open(file_path, 'w') as file:
            json.dump(batch, file, indent=4)
            
        # randomly output some to monitor whats going on
        if np.random.rand()>0.0:
            dd = random.choice(batch)
            input_text =  dd['input_text']
            true_answer =  dd['true_answer']
            reward_mask =  dd['reward_mask']
            print('='*50)
            print("Randomly sampling to show you whats being produced:")
            print(f"input_text: {input_text}")
            print(f"true_answer: {true_answer}")
            print(f"reward_mask: {reward_mask}")
            print('batches_created_since_last_model_update: '+str(batches_created_since_last_model_update))
            print('='*50)

        # check if we should update our left_model and if so, update current_left_model from most recent checkpoint in left_model_directory
        batches_created_since_last_model_update+=1
        
        if batches_created_since_last_model_update > update_left_model_every_n_batches:
            batches_created_since_last_model_update = 0
            # 1. Check if left_model_directory exists
            if not os.path.exists(left_model_directory):
                print(f"Directory {left_model_directory} does not exist. Continuing with current_left_model.")
                continue
        
            # 2. If it exists and there are checkpoints, grab the most recent one
            checkpoint_files = glob.glob(os.path.join(left_model_directory, "*"))
            if not checkpoint_files:
                print(f"No checkpoints found in {left_model_directory}. Continuing with current_left_model.")
                continue
        
            # Sort files by creation time and get the most recent one
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
            # Load the model
            current_left_model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
            current_left_model = current_left_model.to(device)
        
            # 3. Output that you have updated the generating left model checkpoint
            print(f"Updated the generating left model checkpoint to latest checkpoint: {latest_checkpoint}")
            
            # I dont think you have to do this.. but you may??
            # if tokenizer.pad_token is None:
            #     current_left_model.config.pad_token_id = tokenizer.pad_token
            #     current_left_model.resize_token_embeddings(len(tokenizer))



            
