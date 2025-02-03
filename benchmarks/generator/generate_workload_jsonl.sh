#!/bin/bash

SHAREGPT_FILE_PATH="dont_push/sharedgpt/ShareGPT_V3_unfiltered_cleaned_split.json"

python workload_generator.py \
    --prompt-file ${SHAREGPT_FILE_PATH} \
    --trace-type synthetic-from-csv-file \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --output-dir "workload" \
    --stats-csv $1