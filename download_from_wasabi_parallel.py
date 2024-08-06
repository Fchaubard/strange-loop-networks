import boto3
import os
import concurrent.futures

# Configure your Wasabi access and secret key
ACCESS_KEY = ''
SECRET_KEY = ''
REGION_NAME = 'us-west-1'  # Replace with your specific region if different

# Initialize the Wasabi S3 client
s3_client = boto3.client('s3',
                         aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_KEY,
                         endpoint_url='https://s3.us-west-1.wasabisys.com',
                         region_name=REGION_NAME)

# Bucket name
bucket_name = 'strange-loop-networks'

# Local directory to save the downloaded files
local_directory = './downloaded_files/'

# Ensure the local directory exists
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

def download_file(bucket_name, file_name, local_file_path):
    try:
        # Create local directories if they don't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the file
        s3_client.download_file(bucket_name, file_name, local_file_path)
        print(f'Successfully downloaded {file_name} to {local_file_path}')
    except Exception as e:
        print(f'Error downloading {file_name}: {e}')

def download_files(bucket_name, local_directory):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for obj in response['Contents']:
                    file_name = obj['Key']
                    local_file_path = os.path.join(local_directory, file_name)
                    futures.append(executor.submit(download_file, bucket_name, file_name, local_file_path))
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f'Error in future: {e}')
        else:
            print('No files found in the bucket.')
    except Exception as e:
        print(f'Error listing objects in the bucket: {e}')

if __name__ == '__main__':
    download_files(bucket_name, local_directory)