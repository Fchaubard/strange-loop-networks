import os
import json

def is_valid_json(file_path):
    """Check if a JSON file is valid."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False

def clean_invalid_json_files(json_folder):
    """Loop over all JSON files in a folder, deleting any that fail to load."""
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(json_folder, json_file)
        if not is_valid_json(file_path):
            print(f"Invalid JSON file detected and deleted: {file_path}")
            os.remove(file_path)
        else:
            print(f"Valid JSON file: {file_path}")

# Specify the path to your folder containing JSON files
json_folder = '/sln_batches/'

# Clean invalid JSON files
clean_invalid_json_files(json_folder)
