#!/bin/bash

input_workload_path=$1
autoscaler=$2
# target_avg_rps=$3
routing=$3
aibrix_repo="/Users/bytedance/projects/aibrix"
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
k8s_config_dir="deepseek-llm-7b-chat-v100"
target_deployment="aibrix-model-deepseek-llm-7b-chat"
# Input validation
if [ -z "$api_key" ]; then
    echo "API key is not set. Please set the API key in the script"
    exit 1
fi
if [ -z "$input_workload_path" ]; then
    echo "input_workload_path is not given"
    echo "Usage: $0 <input_workload_path> <autoscaler-mechanism>"
    exit 1
fi
if [ -z "$autoscaler" ]; then
    echo "autoscaler is not given"
    echo "Usage: $0 <input_workload_path> <autoscaler-mechanism>"
    exit 1
fi
if [ -z "$routing" ]; then
    echo "routing is not given"
    echo "Usage: $0 <input_workload_path> <autoscaler> <routing>"
    exit 1
fi
# if [ -z "$target_avg_rps" ]; then
#     echo "${target_avg_rps} is not given. will be set to default value 5"
#     target_avg_rps=5
# fi

# Setup experiment directory
workload_name=$(echo $input_workload_path | tr '/' '\n' | grep .jsonl | cut -d '.' -f 1)
# experiment_result_dir="experiment_results/${workload_name}-${autoscaler}-${routing}-$(date +%Y%m%d-%H%M%S)"
experiment_result_dir="routing_experiment/${workload_name}-${autoscaler}-${routing}-$(date +%Y%m%d-%H%M%S)"

if [ ! -d ${experiment_result_dir} ]; then
    echo "output directory does not exist. Create the output directory"
    mkdir -p ${experiment_result_dir}
fi

echo "workload_name: $workload_name"
echo "target_deployment: $target_deployment"
echo "routing: $routing"
echo "autoscaler: $autoscaler"
echo "input_workload_path: $input_workload_path"
echo "experiment_result_dir: $experiment_result_dir"

# Start port-forwarding
kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc 8888:80 &
PORT_FORWARD_PID=$!
echo "started port-forwarding with PID: $PORT_FORWARD_PID"

# Clean up any existing autoscalers
kubectl delete podautoscaler --all --all-namespaces
kubectl delete hpa --all --all-namespaces

# Apply new autoscaler
if [ "$autoscaler" == "none" ]; then
    echo "No autoscaler is applied"
    kubectl apply -f ${k8s_config_dir}/8_replica_hpa.yaml
else
    kubectl apply -f ${k8s_config_dir}/${autoscaler}.yaml
    echo "kubectl apply -f ${k8s_config_dir}/${autoscaler}.yaml"
fi
# Reset deployment
python set_num_replicas.py --deployment ${target_deployment} --replicas 8
kubectl rollout restart deploy aibrix-controller-manager -n aibrix-system
kubectl rollout restart deploy ${target_deployment} -n default

# Wait for pods to be ready
sleep_before_pod_check=20
echo "Sleep for ${sleep_before_pod_check} seconds after restarting deployment"
sleep ${sleep_before_pod_check}
python check_k8s_is_ready.py ${target_deployment}

# Start pod log monitoring
pod_log_dir="${experiment_result_dir}/pod_logs"
mkdir -p ${pod_log_dir}

# Copy input workload
cp ${input_workload_path} ${experiment_result_dir}

# Start pod counter
python count_num_pods.py ${target_deployment} ${experiment_result_dir} &
COUNT_NUM_POD_PID=$!
echo "started count_num_pods.py with PID: $COUNT_NUM_POD_PID"

# python pod_log_monitor.py aibrix-controller-manager ${pod_log_dir} aibrix-system manager &
# CONTROLLER_LOG_MONITOR_PID=$!
# echo "started controller log monitor with PID: $CONTROLLER_LOG_MONITOR_PID"

# python pod_log_monitor.py ${target_deployment} ${pod_log_dir} default vllm-openai &
# POD_LOG_MONITOR_PID=$!
# echo "started pod log monitor with PID: $POD_LOG_MONITOR_PID"

# Run experiment
echo "Experiment starts in 1 second!!!"
sleep 1

echo routing: ${routing}

output_jsonl_path=${experiment_result_dir}/output.jsonl
python3 ${aibrix_repo}/benchmarks/generator/client.py \
    --workload-path ${input_workload_path} \
    --endpoint "localhost:8888" \
    --model deepseek-llm-7b-chat \
    --api-key ${api_key} \
    --output-dir ${experiment_result_dir} \
    --routing-strategy ${routing} \
    --output-file-path ${output_jsonl_path}

echo "Experiment is done. date: $(date)"

pod_log_dir="${experiment_result_dir}/pod_logs"
echo "started dumping pod logs for ${target_deployment} to ${pod_log_dir}"
python dump_pod_log.py ${target_deployment} ${pod_log_dir} default vllm-openai
echo "done dumping pod logs for ${target_deployment} to ${pod_log_dir}"

echo "started dumping pod logs for aibrix-controller-manager to ${pod_log_dir}"
python dump_pod_log.py aibrix-controller-manager ${pod_log_dir} aibrix-system manager
echo "done dumping pod logs for aibrix-controller-manager to ${pod_log_dir}"

# Cleanup
kubectl delete podautoscaler --all --all-namespaces

# set the number of replicas back to 1
# python set_num_replicas.py --deployment ${target_deployment} --replicas 1

# Stop monitoring processes
echo "Stopping monitoring processes..."
kill $COUNT_NUM_POD_PID
echo "killed count_num_pods.py with PID: $COUNT_NUM_POD_PID"

# kill -INT $POD_LOG_MONITOR_PID
# kill -INT $CONTROLLER_LOG_MONITOR_PID
# sleep 5  # Give monitors time to collect final logs
# echo "killed log monitors"

kill $PORT_FORWARD_PID
echo "killed port-forwarding with PID: $PORT_FORWARD_PID"

# # Parse logs and generate plots
# python3 plot/plot-output.py ${experiment_result_dir}
#     # --output-dir ${experiment_result_dir}
#     # --autoscaler ${autoscaler} \
#     # --workload ${workload_name} \

# Copy output file
cp output.txt ${experiment_result_dir}
echo "copied output.txt to ${experiment_result_dir}"

echo "Experiment completed successfully."

# Cleanup function for handling script interruption
cleanup() {
    echo "Cleaning up..."
    kill $PORT_FORWARD_PID 2>/dev/null
    kill $COUNT_NUM_POD_PID 2>/dev/null
    kill -INT $POD_LOG_MONITOR_PID 2>/dev/null
    kill -INT $CONTROLLER_LOG_MONITOR_PID 2>/dev/null
    kubectl delete podautoscaler --all --all-namespaces
    echo "Cleanup completed"
    exit
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM