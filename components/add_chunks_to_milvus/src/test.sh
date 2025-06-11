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

export EMBEDDING_MAP_PATH=$(pwd)/scratch/embedding_map.json
export EMBEDDINGS_DEFAULT_MODEL

# Echo the environment variables for debugging
echo "Using the following environment variables:"
echo "MILVUS_DATABASE: $MILVUS_DATABASE"
echo "MILVUS_HOST: $MILVUS_HOST"
echo "MILVUS_PORT: $MILVUS_PORT"
echo "MILVUS_USERNAME: $MILVUS_USERNAME"
echo "MILVUS_PASSWORD: $MILVUS_PASSWORD"
echo "EMBEDDING_MAP_PATH: $EMBEDDING_MAP_PATH"
echo "EMBEDDINGS_DEFAULT_MODEL: $EMBEDDINGS_DEFAULT_MODEL"
echo "OPENAI_API_KEY_COMPLETIONS: $OPENAI_API_KEY_COMPLETIONS"
echo "OPENAI_API_MODEL_COMPLETIONS: $OPENAI_API_MODEL_COMPLETIONS"
echo "OPENAI_API_BASE_COMPLETIONS: $OPENAI_API_BASE_COMPLETIONS"

# Set PYTHONPATH to the components directory (parent of both shared and docling_converter)
export PYTHONPATH=$(cd "$(dirname "$0")"/../../ && pwd)

# Run the test script with DEBUG log level with all the arguments
LOG_LEVEL=INFO python test.py $@