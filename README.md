Francois' SLN research:

The goal of this research is to test if SLNs can generalize better on the same train data.
- CustomModel_Francois.py (in lm_eval for generating results)
- train_sft.py (train code for sft)
- train_sln.py (train code for SLN)
- sln.py (SLN class definition)
- (be sure to 'git clone https://github.com/Fchaubard/sentence_augs.git')

# Step 1:
Choose a pythia model and a dataset (i.e. EleutherAI/pythia-14m) and update the train files.

# Step 2: (optional)
SFT it with python3 train_sft.py
That will produce a checkpoint with the best performing SFT model.

# Step 3: 
Generate batches with 'python3 generate_batches.py' in a screen to run forever.
# Step 4: 
Train the right model (reward model) with 'python3 train_right.py' in a screen to run forever.

# Step 5: 
Train the left model (trajectory model) with 'python3 train_left.py' in a screen to run forever.

# Step 6: 
Run Eval: Whenever you feel the models have converged and are not learning anymore.. you can run inference with lm_eval or otherwise (probably do not want to use lm_eval anymore..):
- First we must update CustomModel_Francois to pull from the checkpoints. 
- Then we must run something like this:
lm_eval --model CustomModel_Francois --model_args model_id=EleutherAI/pythia-2.8b --tasks hellaswag,lambada_openai,gsm8k,winogrande,piqa,sciq,logiqa --include_path ./ --device cuda:0 --batch_size 8 
or
python -m lm_eval --model CustomModel_Francois --model_args model_id=EleutherAI/pythia-2.8b --tasks hellaswag,lambada_openai,gsm8k,winogrande,piqa,sciq,logiqa --include_path ./ --device cuda:0 --batch_size 8

Now we will have eval results for DPO training procedure!

# Step 7:
Run lm_eval on the SLN model. 
lm_eval --model CustomSLN_Francois --model_args model_id=EleutherAI/pythia-2.8b --tasks hellaswag,lambada_openai,gsm8k,winogrande,piqa,sciq,logiqa --include_path ./ --device cuda:0 --batch_size 8

