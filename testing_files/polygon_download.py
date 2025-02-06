import os
import boto3
from botocore.client import Config

#### We will start by creating a .txt file in /data/raw/ that lists all the files in the S3 bucket.

# Create an S3 client configured for the custom endpoint.
s3 = boto3.resource(
    's3',
    endpoint_url='https://files.polygon.io',
    aws_access_key_id='[KEY]',
    aws_secret_access_key='[KEY]',
    config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
)

bucket_name = 'flatfiles'
bucket = s3.Bucket(bucket_name)

# Define the relative output directory and file path.
output_dir = os.path.join("data", "raw")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 's3_files.txt')

# Write the file names and sizes to the text file.
with open(output_file, 'w') as f:
    for obj in bucket.objects.all():
        f.write(f"{obj.key} - {obj.size} bytes\n")

print(f"File listing has been written to {output_file}")


### Testing downloading of one file

s3_key = 'us_stocks_sip/day_aggs_v1/2024/03/2024-03-01.csv.gz'

destination_path = os.path.join('data', 'raw', *s3_key.split('/'))

os.makedirs(os.path.dirname(destination_path), exist_ok=True)

bucket.download_file(s3_key, destination_path)

print(f"Downloaded {s3_key} to {destination_path}")

## We get a 403 Forbidden error. This is because our account does not have download permissions.
