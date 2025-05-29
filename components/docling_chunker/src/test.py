import os
import logging
import time
import argparse

from pathlib import Path

from docling_chunker import _docling_chunker

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
    parser = argparse.ArgumentParser(description="Parse outputdir, basedir, and a list of files.")
    
    parser.add_argument(
        '--outputdir',
        required=True,
        help='Path to the output directory'
    )
    
    parser.add_argument(
        '--inputdir',
        required=True,
        help='Path to the input directory'
    )

    args = parser.parse_args()

    print(f"Input directory: {args.inputdir}")
    print(f"Output directory: {args.outputdir}")

    # Get TOKENIZER_EMBED_MODEL_ID from environment variable or use default
    tokenizer_embed_model_id = os.getenv("TOKENIZER_EMBED_MODEL_ID", "intfloat/multilingual-e5-large")
    print(f"Tokenizer embed model ID: {tokenizer_embed_model_id}")
    # Get TOKENIZER_MAX_TOKENS from environment variable or use default
    tokenizer_max_tokens = int(os.getenv("TOKENIZER_MAX_TOKENS", 476))  # Default to 476 if not set
    print(f"Tokenizer max tokens: {tokenizer_max_tokens}")
    # Get MERGE_PEERS from environment variable or use default
    merge_peers = os.getenv("MERGE_PEERS", "True").lower() in ("true", "1", "yes")
    print(f"Merge peers: {merge_peers}")

    # Start the timer
    start_time = time.time()

    # Chunk the documents
    success, failure = _docling_chunker(
        input_dir=Path(args.inputdir),
        output_dir=Path(args.outputdir),
        tokenizer_embed_model_id=tokenizer_embed_model_id,
        tokenizer_max_tokens=tokenizer_max_tokens,
        merge_peers=merge_peers
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

    _log.info(f"Document chunking complete in {end_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()