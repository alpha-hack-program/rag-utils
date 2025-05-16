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

    # Start the timer
    start_time = time.time()

    # Chunk the documents
    success, failure = _docling_chunker(
        input_dir=Path(args.inputdir),
        output_dir=Path(args.outputdir),
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