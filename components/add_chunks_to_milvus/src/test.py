import os
import logging
import time
import argparse
import requests

from pathlib import Path

from add_chunks_to_milvus import _add_chunks_to_milvus

# Allowed log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read from environment variable
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "WARNING").upper()

# Set the log level for the 'docling.utils' logger to ERROR
logging.getLogger('docling').setLevel(logging.ERROR)

# Create a logger for this module
_log = logging.getLogger(__name__)

def main():
    # Validate and set level
    if LOG_LEVEL_STR in VALID_LOG_LEVELS:
        log_level = getattr(logging, LOG_LEVEL_STR)
    else:
        print(f"Invalid LOG_LEVEL '{LOG_LEVEL_STR}' - defaulting to WARNING")
        log_level = logging.WARNING

    logging.basicConfig(level=log_level)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse ...")
    
    parser.add_argument(
        '--inputdir',
        required=True,
        help='Path to the input directory'
    )

    parser.add_argument(
        '--collection',
        required=True,
        help='Milvus Collection Name'
    )
    
    args = parser.parse_args()

    print(f"Input directory: {args.inputdir}")

    # Check environment variables for:
    # MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD
    # OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION
    MILVUS_DATABASE = os.getenv("MILVUS_DATABASE")
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")
    MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

    if not all([MILVUS_DATABASE, MILVUS_HOST, MILVUS_PORT, MILVUS_USERNAME, MILVUS_PASSWORD]):
        raise ValueError("Missing required environment variables for Milvus connection.")
    if not all([OPENAI_API_MODEL, OPENAI_API_BASE]):
        raise ValueError("Missing required environment variables for OpenAI connection to generate embeddings.")

    # Start the timer
    start_time = time.time()

    # Add chunks to Milvus
    success, failure = _add_chunks_to_milvus(
        input_dir=Path(args.inputdir),
        milvus_collection_name=args.collection,
    )

    # Log the conversion results
    _log.info(
        f"Successfully converted: {success}"
    )
    _log.info(
        f"Failed to convert: {failure}"
    )

    # Stop the timer
    end_time = time.time() - start_time

    _log.info(f"Chunks insertion complete in {end_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()