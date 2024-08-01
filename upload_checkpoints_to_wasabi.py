import boto3
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

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



def upload_files(directory, file_path_on_wasabi, bucket_name):
    

    # Iterate over files in the directory
    print(f'Uploading files from {directory} to {bucket_name} at {file_path_on_wasabi}')
    for filename in os.listdir(directory):
        # Only upload .pth files
        if filename.endswith('.pth'):
            file_path = os.path.join(directory, filename)
            s3_key = f"{file_path_on_wasabi}/{filename}"
            
            try:
                # Check if the file already exists in the bucket
                s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                print(f'File {s3_key} already exists in bucket {bucket_name}. Skipping upload.')
            except s3_client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    # File does not exist, proceed with upload
                    try:
                        s3_client.upload_file(file_path, bucket_name, s3_key)
                        print(f'Successfully uploaded {file_path} to {s3_key} in bucket {bucket_name}')
                    except (NoCredentialsError, PartialCredentialsError) as creds_error:
                        print(f'Failed to upload {file_path} due to credentials error: {creds_error}')
                    except Exception as upload_error:
                        print(f'Failed to upload {file_path} to {s3_key} in bucket {bucket_name}: {upload_error}')
                else:
                    # Other errors
                    print(f'Failed to check existence of {s3_key} in bucket {bucket_name}: {e}')
            print("----" * 10)

file_path_on_wasabi = 'pythia-2.8b'

directory = '/right_checkpoints/'    
bucket_name = 'right-checkpoints'

upload_files(directory, file_path_on_wasabi, bucket_name)

bucket_name = 'left-checkpoints'
directory = '/left_checkpoints/'    

upload_files(directory, file_path_on_wasabi, bucket_name)

bucket_name = 'left-checkpoints-masking'
directory = '/left_checkpoints_masking/'    

upload_files(directory, file_path_on_wasabi, bucket_name)

bucket_name = 'left-checkpoints-masking'
directory = '/left_checkpoints_masking_random_replacement/'    

upload_files(directory, file_path_on_wasabi, bucket_name)


