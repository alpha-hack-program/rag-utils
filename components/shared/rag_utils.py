import hashlib

from pathlib import Path

# Function to calculate the MD5 hash of a file
def calculate_md5(file: Path):
    """
    Calculate the MD5 hash of a file.
    Args:
        file (Path): Path to the file.
    Returns:
        hashlib.md5: MD5 hash object.
    """
    # Create an MD5 hash object
    md5_hash = hashlib.md5()
    # Open the file in binary mode and read it in chunks
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash

# Function to check if the file has processed
def is_processed(file: Path) -> bool:
    """
    Check if the file has not processed by calculating the MD5 hash of the file and checking if 
    there is a file with the same name + "{hash}.processed" in the same directory.
    This is used to avoid reprocessing files that have already been processed.
    Args:
        file (Path): Path to the file.
    Returns:
        bool: True if the file has been processed, False otherwise.
    """
    # Calculate the MD5 hash of the file
    md5_hash = calculate_md5(file)
    # Get the hash value as a hexadecimal string
    hash_value = md5_hash.hexdigest()
    # Check if there is a file with the same name + "{hash}" in the same directory
    processed_file = file.with_name(f"{file.stem}.{hash_value}.processed")
    if processed_file.exists():
        return True
    else:
        return False

def mark_file_processed(file: Path) -> None:
    """
    Mark the file as processed by creating a file with the same name + "{hash}" in the same directory.
    This is used to avoid reprocessing files that have already been processed.
    Args:
        file (Path): Path to the file.
    """
    # Calculate the MD5 hash of the file
    md5_hash = calculate_md5(file)
    # Get the hash value as a hexadecimal string
    hash_value = md5_hash.hexdigest()
    # Create a file with the same name + "{hash}" in the same directory
    processed_file = file.with_name(f"{file.stem}.{hash_value}.processed")
    with processed_file.open("w") as f:
        f.write("")
        f.write(f"File {file} has been processed.")
        f.write(f"\nHash value: {hash_value}")

def mark_file_chunked(file: Path) -> None:
    """
    Mark the file as chunked by creating a file with the same name + "{hash}.chunked" in the same directory.
    This is used to avoid reprocessing files that have already been chunked.
    Args:
        file (Path): Path to the file.
    """
    # Calculate the MD5 hash of the file
    md5_hash = calculate_md5(file)
    # Get the hash value as a hexadecimal string
    hash_value = md5_hash.hexdigest()
    # Create a file with the same name + "{hash}" in the same directory
    chunked_file = file.with_name(f"{file.stem}.{hash_value}.chunked")
    with chunked_file.open("w") as f:
        f.write("")
        f.write(f"File {file} has been chunked.")
        f.write(f"\nHash value: {hash_value}")

# Function to check if the file has been chunked
def is_chunked(file: Path) -> bool:
    """
    Check if the file has  been chunked by calculating the MD5 hash of the file and checking if 
    there is a file with the same name + "{hash}.chunked" in the same directory.
    This is used to avoid reprocessing files that have already been chunked.
    Args:
        file (Path): Path to the file.
    Returns:
        bool: True if the file has been chunked, False otherwise.
    """
    # Calculate the MD5 hash of the file
    md5_hash = calculate_md5(file)
    # Get the hash value as a hexadecimal string
    hash_value = md5_hash.hexdigest()
    # Check if there is a file with the same name + "{hash}" in the same directory
    processed_file = file.with_name(f"{file.stem}.{hash_value}.chunked")
    if processed_file.exists():
        return True
    else:
        return False