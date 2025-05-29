#!/bin/bash

# Set PYTHONPATH to the components directory (parent of both shared and docling_converter)
export PYTHONPATH=$(cd "$(dirname "$0")"/../../ && pwd)

export TOKENIZER_EMBED_MODEL_ID="intfloat/multilingual-e5-large"
export TOKENIZER_MAX_TOKENS=512

# Run the test script with DEBUG log level with all the arguments
LOG_LEVEL=DEBUG python test.py $@