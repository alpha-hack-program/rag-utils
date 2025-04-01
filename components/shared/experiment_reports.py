import os
import yaml
import boto3
import botocore

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
region_name = os.environ.get('AWS_DEFAULT_REGION')
bucket_name = os.environ.get('AWS_S3_BUCKET')
experiment_reports_folder = os.environ.get('EXPERIMENT_REPORTS_FOLDER_S3_KEY')

# Download experiment report from S3 bucket and folder
def download_experiment_report(experiment_name: str) -> dict:
    # Experiment report s3 key to store the report
    experiment_report_file_name = f"{experiment_name}.yaml"
    experiment_report_s3_key = f"{experiment_reports_folder}/{experiment_report_file_name}"
    print(f"experiment_report_s3_key = {experiment_report_s3_key}")
        # Create a session using the provided credentials and region
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Create an S3 resource object using the session and region
    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name
    )

    # Get the bucket
    bucket = s3_resource.Bucket(bucket_name)

    # Create a temporary directory to store the report
    local_tmp_dir = '/tmp/experiment_report'
    print(f">>> local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Local file to store the report
    experiment_report_file_path = f'{local_tmp_dir}/{experiment_report_file_name}'

    print(f"Downloading {experiment_report_s3_key} to {experiment_report_file_path}")
    bucket.download_file(experiment_report_s3_key, experiment_report_file_path)
    print(f"Downloaded {experiment_report_s3_key}")

    # Read the content of the file and retun it
    experiment_report = None
    with open(experiment_report_file_path, 'r') as file:
        experiment_report = file.read()
    return load_yaml(experiment_report)

# Return a dict from a yaml string
def load_yaml(search_space: str) -> dict:
    return yaml.safe_load(search_space)