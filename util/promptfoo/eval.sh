#!/bin/bash

# Set the working directory to the script's directory
cd "$(dirname "$0")"

# Source the .env file if it exists
if [ -f .env ]; then
  . .env
fi

# Check if environment variable API_BASE_URL is set
if [ -z "$API_BASE_URL" ]; then
  echo "API_BASE_URL environment variable is not set."
  exit 1
fi

# Check if the rag.yaml file exists
if [ ! -f rag.yaml ]; then
  echo "rag.yaml file not found in the current directory."
  exit 1
fi

# Read all CSV files in the scratch directory
if [ -d ./scratch ]; then
  csv_files=("./scratch/"*.csv)
  if [ ${#csv_files[@]} -eq 0 ]; then
    echo "No CSV files found in the scratch directory."
    exit 1
  fi
else
  echo "scratch directory does not exist."
  exit 1
fi

# Merge all CSV (except merged_qa_tmp) files into one temporary file
merged_file="./scratch/merged_qa_tmp.csv"
if [ -f "$merged_file" ]; then
  rm "$merged_file"
fi
for file in "${csv_files[@]}"; do
  if [[ "$file" != *"merged_qa_tmp.csv" ]]; then
    if [ -f "$file" ]; then
      if [ ! -f "$merged_file" ]; then
        cp "$file" "$merged_file"
      else
        tail -n +2 "$file" >> "$merged_file"
      fi
    else
      echo "File $file does not exist."
    fi
  fi
done

# Check if the merged file was created successfully
if [ ! -f "$merged_file" ]; then
  echo "Failed to create merged CSV file."
  exit 1
fi

export HF_API_TOKEN

# Run the promptfoo eval command with the merged CSV file
npx promptfoo@latest eval \
  --max-concurrency 4 \
  -c ./rag.yaml \
  -t ./scratch/merged_qa_tmp.csv
