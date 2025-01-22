#!/bin/bash

SHAREGPT_FILE_PATH="sharedgpt/ShareGPT_V3_unfiltered_cleaned_split.json"

python workload_generator.py \
    --prompt-file ${SHAREGPT_FILE_PATH} \
    --trace-type synthetic-from-csv-file \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --output-dir "workload" \
    --stats-csv "input_trace/maas/window_0-20241017_2102-20241017_2110.csv"
    # --stats-csv "input_trace/maas/window_0-20241017_2102-20241017_2132.csv"