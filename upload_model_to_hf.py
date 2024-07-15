from huggingface_hub import HfApi, HfFolder

# Define your variables
#repo_id = "fchaubard/right-strange-loop-network-410m"
#model_file_path = "/right_checkpoints/right_checkpoint_20240715173212_iter_800_loss_37.63.pth"

repo_id = "fchaubard/left-strange-loop-network-410m"
model_file_path = "/left_checkpoints/left_checkpoint_20240715173212_iter_800_loss_37.63.pth"
commit_message = "Uploading new checkpoint file."

# Get your Hugging Face token from the environment or directly
# Make sure you have logged in using `huggingface-cli login` or set your token manually here
hf_token = "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"

# Initialize the HfApi
api = HfApi()

# Upload the file
api.upload_file(
            path_or_fileobj=model_file_path,
            path_in_repo="left_checkpoint_20240715173212_iter_800_loss_37.63.pth",
            repo_id=repo_id,
            commit_message=commit_message,
            token=hf_token
        )

print("File uploaded successfully.")

