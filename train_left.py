# train_left.py 
# - picks up batches from ./batches/*, sorts by time to grab n most recent, and then trains left model on them. 
# - pushes results to wandb, and saves checkpoints to ./checkpoint_left/ with format 'left_checkpoint_timestamp_iter_loss.h5'
# - train with Cross Entropy? or reward with policy gradient? or PPO? or DPO? ... TBD! IMPLEMENT THIS LAST.