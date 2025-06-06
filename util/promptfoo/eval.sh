#!/bin/bash
cd "$(dirname "$0")"

npx promptfoo eval \
  -c ./rag.yaml \
  -t ./scratch/merged_qa_granite_3_1_8b_instruct_w4a16_20250601t153820.csv
