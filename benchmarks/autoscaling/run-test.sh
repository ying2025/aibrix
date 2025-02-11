#!/bin/bash

input_workload_path=$1
autoscaler=$2
routing=$3
# target_avg_rps=$4 # will be added later
aibrix_repo="/Users/bytedance/projects/aibrix"
api_key="" # set your api key
k8s_yaml_dir="deepseek-llm-7b-chat-v100"
target_deployment="deepseek-llm-7b-chat-v100" # "aibrix-model-deepseek-llm-7b-chat"
ai_model=deepseek-llm-7b-chat

echo "Make sure ${target_deployment} is the right deployment."
sleep 3

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
if [${k8s_yaml_dir}/${autoscaler}.yaml file does not exist]; then
    echo "${k8s_yaml_dir}/${autoscaler}.yaml does not exist.. Check the ${autoscaler}"
    exit 1
fi

# Setup experiment directory
workload_name=$(echo $input_workload_path | tr '/' '\n' | grep .jsonl | cut -d '.' -f 1)
experiment_result_dir="experiment_results/${workload_name}-${autoscaler}-${routing}-$(date +%Y%m%d-%H%M%S)"
if [ ! -d ${experiment_result_dir} ]; then
    echo "output directory does not exist. Create the output directory (${experiment_result_dir})"
    mkdir -p ${experiment_result_dir}
fi

echo "----------------------------------------"
echo "workload_name: $workload_name"
echo "target_deployment: $target_deployment"
echo "routing: $routing"
echo "autoscaler: $autoscaler"
echo "input_workload_path: $input_workload_path"
echo "experiment_result_dir: $experiment_result_dir"
echo "----------------------------------------"

# Port-forwarding
# It is needed only when you run the client in the laptop not inside the K8S cluster.
kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc 8888:80 &
PORT_FORWARD_PID=$!
echo "started port-forwarding with PID: $PORT_FORWARD_PID"

# Clean up any existing autoscalers
kubectl delete podautoscaler --all --all-namespaces
kubectl delete hpa --all --all-namespaces

# Apply new autoscaler
if [ "$autoscaler" == "none" ]; then
    echo "No autoscaler should be applied."
    echo "Set the number of replicas to \"8\" (this is subject to your set up.)"
    python set_num_replicas.py --deployment ${target_deployment} --replicas 8
else
    kubectl apply -f ${k8s_yaml_dir}/${autoscaler}.yaml
    echo "kubectl apply -f ${k8s_yaml_dir}/${autoscaler}.yaml"
    python set_num_replicas.py --deployment ${target_deployment} --replicas 1
    echo "Set number of replicas to \"1\". Autoscaling experiment will start from 1 pod"
fi

echo "Restart aibrix-controller-manager deployment"
kubectl rollout restart deploy aibrix-controller-manager -n aibrix-system

echo "Restart aibrix-gateway-plugins deployment"
kubectl rollout restart deploy aibrix-gateway-plugins -n aibrix-system

echo "Restart ${target_deployment} deployment"
kubectl rollout restart deploy ${target_deployment} -n default

# Wait for pods to be ready
sleep_before_pod_check=20
echo "Sleep for ${sleep_before_pod_check} seconds after restarting deployment"
sleep ${sleep_before_pod_check}
python check_k8s_is_ready.py ${target_deployment}
python check_k8s_is_ready.py aibrix-controller-manager
python check_k8s_is_ready.py aibrix-gateway-plugins

# Start pod log monitoring
pod_log_dir="${experiment_result_dir}/pod_logs"
mkdir -p ${pod_log_dir}

# Copy the snapshot of input workload
cp ${input_workload_path} ${experiment_result_dir}

# Start pod counter. It will run on background until the end of the experiment.
python count_num_pods.py ${target_deployment} ${experiment_result_dir} &
COUNT_NUM_POD_PID=$!
echo "started count_num_pods.py with PID: $COUNT_NUM_POD_PID"

# Streaming pod logs to files on the background
python streaming_pod_log_to_file.py ${target_deployment} default ${pod_log_dir} & pid_1=$!
python streaming_pod_log_to_file.py aibrix-controller-manager aibrix-system ${pod_log_dir} & pid_2=$!
python streaming_pod_log_to_file.py aibrix-gateway-plugins aibrix-system ${pod_log_dir} & pid_3=$!

# Run experiment!!!
output_jsonl_path=${experiment_result_dir}/output.jsonl
python3 ${aibrix_repo}/benchmarks/generator/client.py \
    --workload-path ${input_workload_path} \
    --endpoint "localhost:8888" \
    --model ${ai_model} \
    --api-key ${api_key} \
    --output-dir ${experiment_result_dir} \
    --routing-strategy ${routing} \
    --output-file-path ${output_jsonl_path}

echo "Experiment is done. date: $(date)"

sleep 5
kill ${pid_1}
kill ${pid_2}
kill ${pid_3}
sleep 1

# Cleanup
kubectl delete podautoscaler --all --all-namespaces
python set_num_replicas.py --deployment ${target_deployment} --replicas 1

# Stop monitoring processes
echo "Stopping monitoring processes..."
kill $COUNT_NUM_POD_PID
echo "killed count_num_pods.py with PID: $COUNT_NUM_POD_PID"

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
echo "Experiment completed."

# Cleanup function for handling interruption
cleanup() {
    echo "Cleaning up..."
    kill $PORT_FORWARD_PID 2>/dev/null
    kill $COUNT_NUM_POD_PID 2>/dev/null
    kill $pid_1 2>/dev/null
    kill $pid_2 2>/dev/null
    kill $pid_3 2>/dev/null
    kubectl delete podautoscaler --all --all-namespaces
    echo "Cleanup completed"
    exit
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM
