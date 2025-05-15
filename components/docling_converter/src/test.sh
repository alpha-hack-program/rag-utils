#!/bin/bash

# Set PYTHONPATH to the components directory (parent of both shared and docling_converter)
export PYTHONPATH=$(cd "$(dirname "$0")"/../../ && pwd)

# Run the test script with DEBUG log level with all the arguments
LOG_LEVEL=DEBUG python test.py $@