#!/bin/bash

api_key=""
if [ -z "$api_key" ]; then
    echo "API key is not set. Please set the API key in the script"
    exit 1
fi

input_workload_path=$1
if [ -z "$input_workload_path" ]; then
    echo "Usage: $0 <workload-path>"
    exit 1
fi

## split the workload path to get the workload name. split by '/' and '.' get the last element.
workload_name=$(echo $input_workload_path | tr '/' '\n' | grep .jsonl | cut -d '.' -f 1)
workload_dir=$(dirname "$input_workload_path") 
output_dir="${workload_dir}/output"
output_file_path="${workload_dir}/output/output-${workload_name}.jsonl"

if [ ! -d ${output_dir} ]; then
    echo "output directory does not exist. Create the output directory"
    mkdir -p ${output_dir}
fi

# if [ -f ${output_file_path} ]; then
#     echo "output file exists. Create a new output file with timestamp"
#     output_file_path="${output_file_path}-$(date +%s).jsonl"
# fi

echo "workload_name: $workload_name"
echo "workload_dir: $workload_dir"
echo "input_workload_path: $input_workload_path"
echo "output_dir: $output_dir"
echo "output_file_path: $output_file_path"




# exp 1: st1, et1
# exp 2: st2, et2

# prom

## run all the experiments

## apply config

## log the start time

## vke
python3 client.py --workload-path ${input_workload_path} \
    --endpoint "http://101.126.74.79:8000" \
    --model deepseek-coder-7b-instruct \
    --api-key ${api_key} \
    --output-file-path ${output_file_path}


##  log the end time

## send http request to the prometheus server with start/end time parameter and write to a specific path
# """
# curl -G 'http://your-prometheus-server:9090/api/v1/query_range' \
#     --data-urlencode 'query=<your_log_query>' \
#     --data-urlencode 'start=2024-01-16T00:00:00Z' \
#     --data-urlencode 'end=2024-01-17T00:00:00Z' \
#     --data-urlencode 'step=1h' > /path/to/your/log.json
# """

## plot


## local setup
# python3 client.py --workload-path ${input_workload_path} \
#     --endpoint "http://localhost:8000" \
#     --model llama2-7b \
#     --api-key sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi \
#     --output-file-path mock-output.jsonl

echo "----------------------------------------------"
python3 plot-output.py \
    --input_file ${output_file_path} \
    --output-dir "workload/${workload_name}/output"
