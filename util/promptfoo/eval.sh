#!/bin/bash
cd "$(dirname "$0")"

npx promptfoo eval \
  -c ./rag.yaml \
  -t ./scratch/merged_qa_test.csv
