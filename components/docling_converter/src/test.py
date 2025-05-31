import os
import logging
import re
import time
import argparse

from pathlib import Path

from docling_converter import _docling_convert

# Allowed log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read from environment variable
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "WARNING").upper()

# Set the log level for the 'docling.utils' logger to ERROR
logging.getLogger('docling').setLevel(logging.ERROR)

# Create a logger for this module
_log = logging.getLogger(__name__)

def refresh_input_dir(input_dir: Path):
    """
    Deletes all files ending with '.processed' in the given directory.
    """
    for file in input_dir.rglob("*.processed"):
        try:
            file.unlink()
        except Exception as e:
            logging.warning(f"Could not delete {file}: {e}")

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
        '--inputdir',
        required=True,
        help='Path to the input directory'
    )

    parser.add_argument(
        '--outputdir',
        required=True,
        help='Path to the output directory'
    )
    
    # Add an optional boolean argument to start fresh called --refresh
    parser.add_argument(
        '--refresh',
        required=False,
        help='Delete all processed files in the input directory before starting'
    )
    
    args = parser.parse_args()

    print(f"Input directory: {args.inputdir}")
    print(f"Output directory: {args.outputdir}")
    print(f"Refresh input directory: {args.refresh}")

    # Validate the input directory
    input_dir = Path(args.inputdir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory '{args.inputdir}' does not exist or is not a directory.")

    # Delete all *.processed files in the input directory if --fresh is specified
    if args.refresh:
        print("Starting fresh, deleting all processed files...")
        refresh_input_dir(input_dir=input_dir)

    # Start the timer
    start_time = time.time()

    # Convert the documents
    success, partial_success, failure = _docling_convert(
        input_dir=input_dir,
        output_dir=Path(args.outputdir),
    )

    # Log the conversion results
    _log.info(
        f"Successfully converted: {success}"
    )
    _log.info(
        f"Partially converted: {partial_success}"
    )
    _log.info(
        f"Failed to convert: {failure}"
    )

    # Stop the timer
    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()