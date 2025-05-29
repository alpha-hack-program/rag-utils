#!/bin/bash

# Check if .test.env exists
if [ ! -f ".test.env" ]; then
    echo "Error: .test.env file not found!"
    exit 1
fi

source .test.env

export MILVUS_DATABASE MILVUS_HOST MILVUS_PORT MILVUS_USERNAME MILVUS_PASSWORD
export OPENAI_API_KEY OPENAI_API_MODEL OPENAI_API_BASE
export OPENAI_API_KEY_COMPLETIONS OPENAI_API_MODEL_COMPLETIONS OPENAI_API_BASE_COMPLETIONS

# Set PYTHONPATH to the components directory (parent of both shared and docling_converter)
export PYTHONPATH=$(cd "$(dirname "$0")"/../../ && pwd)

# Run the test script with DEBUG log level with all the arguments
LOG_LEVEL=INFO python test.py $@