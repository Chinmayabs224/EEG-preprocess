import boto3
import os

def download_s3_folder(bucket_name, s3_folder, local_dir):
    """
    Downloads all files from an S3 folder to a local directory.
    """
    # Ask the user for their AWS credentials at runtime
    print("Please enter your AWS Credentials to access the bucket.")
    access_key = input("AWS Access Key ID: ").strip()
    secret_key = input("AWS Secret Access Key: ").strip()

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1'
    )
    s3 = session.client('s3')
    
    # Create a reusable paginator to handle folders with >1000 files
    paginator = s3.get_paginator('list_objects_v2')
    
    # Ensure the S3 folder path ends with a slash for correct prefix matching
    if not s3_folder.endswith('/'):
        s3_folder += '/'

    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        # 'Contents' contains the list of objects in the current page
        if 'Contents' not in page:
            print(f"No files found in {s3_folder}")
            return

        for obj in page['Contents']:
            key = obj['Key']
            
            # 1. Handle the local file path
            # We strip the s3_folder prefix to recreate the structure locally
            relative_path = os.path.relpath(key, s3_folder)
            local_file_path = os.path.join(local_dir, relative_path)
            
            # 2. Skip directory markers in S3 (objects ending in '/')
            if key.endswith('/'):
                continue

            # 3. Create local directories if they don't exist
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            # 4. Download the file
            print(f"Downloading: {key} -> {local_file_path}")
            s3.download_file(bucket_name, key, local_file_path)

# Usage
download_s3_folder(
    bucket_name='eegpreprocess', 
    s3_folder='plots/', 
    local_dir='./plots'
)