import os
import logging
import time
import argparse

from pathlib import Path

from generate_qa_per_chunk import _generate_qa_per_chunk

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
        '--questions',
        required=True,
        help='Number of questions to generate per chunk'
    )
    
    args = parser.parse_args()

    print(f"Input directory: {args.inputdir}")
    print(f"Number of questions: {args.questions}")

    # Check environment variables for:
    # OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

    if not all([OPENAI_API_MODEL, OPENAI_API_BASE]):
        raise ValueError("Missing required environment variables for OpenAI connection to generate embeddings.")

    # Start the timer
    start_time = time.time()

    # Generate QA per chunk
    success, failure = _generate_qa_per_chunk(
        input_dir=Path(args.inputdir),
        number_of_questions=int(args.questions),
    )

    # Log the conversion results
    _log.info(
        f"Successfully generated: {success}"
    )
    _log.info(
        f"Failed to generate: {failure}"
    )

    # Stop the timer
    end_time = time.time() - start_time

    _log.info(f"Chunks insertion complete in {end_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()