import boto3
import os

# Configure your Wasabi access and secret key
ACCESS_KEY = 'HIZEUW3ZPB32RZM68OUM'
SECRET_KEY = 'wcnUylYMF9frcUkaVd5yQOm52BqDdUG3mbMa8A6V'
REGION_NAME = 'us-west-1'  # Replace with your specific region if different

# Initialize the Wasabi S3 client
s3_client = boto3.client('s3',
                         aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_KEY,
                         endpoint_url='https://s3.us-west-1.wasabisys.com',
                         region_name=REGION_NAME)

# Directory containing your JSON files
directory = '/right_checkpoints/'    

# Bucket name
bucket_name = 'strange-loop-networks'

def upload_files(directory, bucket_name):
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Only upload .json files
        if filename.endswith('right_*.pth'):
            file_path = os.path.join(directory, filename)
            try:
                # Upload the file
                s3_client.upload_file(file_path, bucket_name, filename)
                print(f'Successfully uploaded {filename} to {bucket_name}')
            except Exception as e:
                print(f'Error uploading {filename}: {e}')


upload_files(directory, bucket_name)
