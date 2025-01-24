#!/bin/bash

api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
if [ -z "$api_key" ]; then
    echo "API key is not set. Please set the API key in the script"
    exit 1
fi

input_workload_path=$1
if [ -z "$input_workload_path" ]; then
    echo "Usage: $0 <workload-path>"
    exit 1
fi
workload_name=$(echo $input_workload_path | tr '/' '\n' | grep .jsonl | cut -d '.' -f 1)
experiment_result_dir="experiment_results/${workload_name}-$(date +%Y%m%d-%H%M%S)"

if [ ! -d ${experiment_result_dir} ]; then
    echo "output directory does not exist. Create the output directory"
    mkdir -p ${experiment_result_dir}
fi

echo "workload_name: $workload_name"
echo "input_workload_path: $input_workload_path"
echo "experiment_result_dir: $experiment_result_dir"

## port-forwarding
kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc  8888:80 &

k8s_config_dir="deepseek-llm-7b-chat-v100"
target_deployment="aibrix-model-deepseek-llm-7b-chat"
# echo "**********************"
# echo "* target_deployment: $target_deployment"
# echo "* will start in 3 seconds. check if this deployment is what you want."
# echo "**********************"
# sleep 3


kubectl delete -f ${k8s_config_dir}/kpa.yaml
kubectl apply -f ${k8s_config_dir}/kpa.yaml
python set_replicas_to_1.py --deployment ${target_deployment} --replicas 1
kubectl rollout restart deploy aibrix-controller-manager -n aibrix-system
kubectl rollout restart deploy ${target_deployment} -n default
echo "Sleep for 10 seconds after restarting deployment"
sleep 10
python check_k8s_is_ready.py ${target_deployment}

echo "Experiment starts in 1 seconds!!!"
sleep 1

cp ${input_workload_path} ${experiment_result_dir}

python count_num_pods.py ${target_deployment} ${experiment_result_dir} &
COUNT_NUM_POD_PID=$!

output_jsonl_path=${experiment_result_dir}/output.jsonl
python3 /Users/bytedance/projects/serverless/aibrix-experiment/benchmarks/generator/client.py \
    --workload-path ${input_workload_path} \
    --endpoint "localhost:8888" \
    --model deepseek-llm-7b-chat \
    --api-key ${api_key} \
    --output-dir ${experiment_result_dir} \
    --output-file-path ${output_jsonl_path} \
    --routing-strategy least-request

kill $COUNT_NUM_POD_PID

pod_log_dir="${experiment_result_dir}/pod_logs"
python dump_pod_log.py ${target_deployment} ${pod_log_dir} default vllm-openai
python dump_pod_log.py aibrix-controller-manager ${pod_log_dir} aibrix-system manager
python parse_application_pod_log.py ${pod_log_dir}
python3 plot/plot-output.py --output-dir ${experiment_result_dir}

cp output.txt ${experiment_result_dir}