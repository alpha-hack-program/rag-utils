#!/bin/bash

# Check if .test.env exists
if [ ! -f ".test.env" ]; then
    echo "Error: .test.env file not found!"
    exit 1
fi

source .test.env

export OPENAI_API_KEY OPENAI_API_MODEL OPENAI_API_BASE

# Echo the environment variables for debugging
echo "Using the following environment variables:"
echo "OPENAI_API_MODEL: $OPENAI_API_MODEL"
echo "OPENAI_API_BASE: $OPENAI_API_BASE"

# Set PYTHONPATH to the components directory (parent of both shared and docling_converter)
export PYTHONPATH=$(cd "$(dirname "$0")"/../../ && pwd)

# Run the test script with DEBUG log level with all the arguments
LOG_LEVEL=DEBUG python test.py $@