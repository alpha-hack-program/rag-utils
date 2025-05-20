import os
import boto3
import hashlib
import json

from kfp import dsl

NAMESPACE = os.environ.get("NAMESPACE", "default")
COMPONENT_NAME = os.getenv("COMPONENT_NAME", f"s3_sync")
BASE_IMAGE = os.getenv("BASE_IMAGE", "python:3.11-slim-bullseye")
REGISTRY = os.environ.get("REGISTRY", f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}")
TAG = os.environ.get("TAG", f"latest")
TARGET_IMAGE = f"{REGISTRY}/{COMPONENT_NAME}:{TAG}"

LOAD_DOTENV_PIP_VERSION = "0.1.0"
BOTOCORE_PIP_VERSION = "1.35.54"
BOTO3_PIP_VERSION = "1.35.54"

def compute_md5(file_path: str, chunk_size: int = 4096) -> str:
    """
    Computes the MD5 hash of a file.

    :param file_path: Path to the file.
    :param chunk_size: Size of the chunk to read at a time.
    :return: MD5 hex digest.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _s3_sync(
        endpoint_url: str,
        region_name: str,
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        folder: str,
        root_mount_path: str,
        local_folder: str,
        force: bool = False
) -> list[str]:
    s3 = boto3.client('s3',
                      endpoint_url=endpoint_url,
                      region_name=region_name,
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)

    # Ensure the destination base directory exists
    dest_base = os.path.join(root_mount_path, local_folder)
    if not os.path.exists(dest_base):
        os.makedirs(dest_base)

    # Paginate through the S3 objects in the specified bucket and folder
    list_of_files = []
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder)

    for page in page_iterator:
        if 'Contents' not in page:
            print("No objects found in remote folder:", folder)
            return []

        for obj in page['Contents']:
            remote_key = obj['Key']

            # Get the parts of the remote key
            # The expected structure is: <collection>/<x>/<y>/<z>/...
            parts = remote_key.split('/')
            if len(parts) <= 0:
                print(f"Skipping {remote_key}: we need at least a root folder 'collection'")
                continue

            # Join all the parts
            relative_path = os.path.join(*parts[0:])
            local_file_path = os.path.join(dest_base, relative_path)

            remote_size = obj.get('Size', 0)
            remote_etag = obj.get('ETag', '').strip('"')
            remote_last_modified = obj.get('LastModified')

            download = force
            reasons = ['forced'] if force else []

            if not force:
                if os.path.exists(local_file_path):
                    local_size = os.path.getsize(local_file_path)
                    if local_size != remote_size:
                        reasons.append(f"size mismatch (local: {local_size}, remote: {remote_size})")
                        download = True
                    else:
                        local_md5 = compute_md5(local_file_path)
                        if local_md5 != remote_etag:
                            reasons.append("hash mismatch")
                            download = True
                        else:
                            local_mod_time = os.path.getmtime(local_file_path)
                            remote_mod_time = remote_last_modified.timestamp() if remote_last_modified else None
                            if remote_mod_time and (remote_mod_time - local_mod_time) > 1:
                                reasons.append("local file is older than remote file")
                                download = True
                            else:
                                print(f"File is up-to-date: {local_file_path}")
                else:
                    reasons.append("file does not exist locally")
                    download = True

            if download:
                print(f"Downloading {remote_key} to {local_file_path} because: {', '.join(reasons)}")
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                try:
                    s3.download_file(bucket_name, remote_key, local_file_path)
                    print(f"Downloaded {remote_key} to {local_file_path}")
                    if remote_last_modified:
                        mod_time = remote_last_modified.timestamp()
                        os.utime(local_file_path, (mod_time, mod_time))
                    list_of_files.append(local_file_path)
                except Exception as e:
                    print(f"Error downloading {remote_key}: {e}")

    return list_of_files


@dsl.component(
    base_image=BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[
        f"load_dotenv=={LOAD_DOTENV_PIP_VERSION}",
        f"botocore=={BOTOCORE_PIP_VERSION}",
        f"boto3=={BOTO3_PIP_VERSION}"
    ]
)
def s3_sync(
        endpoint_url: str,
        region_name: str,
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        folder: str,
        root_mount_path: str,
        local_folder: str,
        force: bool = False,
) -> str:
    print(f"Syncing files from {bucket_name}/{folder} to {local_folder}")
    print(f"force = {force}")
    
    if not endpoint_url or not bucket_name or not aws_access_key_id or not aws_secret_access_key or not folder:
        raise ValueError("One or more required parameters for S3 interaction are not set")
    if not root_mount_path or not local_folder:
        raise ValueError("One or more required parameters for local interaction are not set")

    if not root_mount_path or not os.path.exists(root_mount_path):
        raise ValueError(f"Root mount path '{root_mount_path}' does not exist")

    local_folder_path = os.path.join(root_mount_path, local_folder)
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)

    list_of_files = _s3_sync(
        endpoint_url,
        region_name,
        bucket_name,
        aws_access_key_id,
        aws_secret_access_key,
        folder,
        root_mount_path,
        local_folder,
        force=force
    )

    # return the list as a json string
    return json.dumps(list_of_files)
    
if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    s3_sync.save_component_yaml(component_package_path)
