import os
import glob

# Directory containing the checkpoint files
directories = ["/right_checkpoints/","/left_checkpoints/","/left_checkpoints_masking/","/left_checkpoints_masking_random_replacement/" ]
for directory in directories:
    # Get list of all checkpoint files in the directory
    files = glob.glob(os.path.join(directory, "*_checkpoint_*.pth"))
    
    # Sort files by modification time (oldest first)
    files.sort(key=os.path.getmtime)
    
    if len(files)>3:
        # Calculate the number of files to delete (first half)
        num_files_to_delete = len(files) // 2
        
        # Delete the oldest half of the files
        for file in files[:num_files_to_delete]:
            os.remove(file)
            print(f"Deleted: {file}")
        
        print("Oldest half of the files have been deleted.")
