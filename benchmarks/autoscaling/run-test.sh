#!/bin/bash

input_workload_path=$1
autoscaler_mechanism=$2
target_avg_rps=$3
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
if [ -z "$autoscaler_mechanism" ]; then
    echo "autoscaler_mechanism is not given"
    echo "Usage: $0 <input_workload_path> <autoscaler-mechanism>"
    exit 1
fi
if [ -z "$target_avg_rps" ]; then
    echo "${target_avg_rps} is not given. will be set to default value 5"
    target_avg_rps=5
fi

# Setup experiment directory
workload_name=$(echo $input_workload_path | tr '/' '\n' | grep .jsonl | cut -d '.' -f 1)
experiment_result_dir="experiment_results/${workload_name}-${autoscaler_mechanism}-$(date +%Y%m%d-%H%M%S)"

if [ ! -d ${experiment_result_dir} ]; then
    echo "output directory does not exist. Create the output directory"
    mkdir -p ${experiment_result_dir}
fi

echo "workload_name: $workload_name"
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
kubectl apply -f ${k8s_config_dir}/${autoscaler_mechanism}.yaml
echo "kubectl apply -f ${k8s_config_dir}/${autoscaler_mechanism}.yaml"

# Reset deployment
python set_replicas_to_1.py --deployment ${target_deployment} --replicas 1
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

python pod_log_monitor.py ${target_deployment} ${pod_log_dir} default vllm-openai &
POD_LOG_MONITOR_PID=$!
echo "started pod log monitor with PID: $POD_LOG_MONITOR_PID"

python pod_log_monitor.py aibrix-controller-manager ${pod_log_dir} aibrix-system manager &
CONTROLLER_LOG_MONITOR_PID=$!
echo "started controller log monitor with PID: $CONTROLLER_LOG_MONITOR_PID"

# Copy input workload
cp ${input_workload_path} ${experiment_result_dir}

# Start pod counter
python count_num_pods.py ${target_deployment} ${experiment_result_dir} &
COUNT_NUM_POD_PID=$!
echo "started count_num_pods.py with PID: $COUNT_NUM_POD_PID"

# Run experiment
echo "Experiment starts in 1 second!!!"
sleep 1

output_jsonl_path=${experiment_result_dir}/output.jsonl
python3 ${aibrix_repo}/benchmarks/generator/client.py \
    --workload-path ${input_workload_path} \
    --endpoint "localhost:8888" \
    --model deepseek-llm-7b-chat \
    --api-key ${api_key} \
    --output-dir ${experiment_result_dir} \
    --output-file-path ${output_jsonl_path} \
    --routing-strategy least-request

echo "Experiment is done. date: $(date)"

# Cleanup
kubectl delete podautoscaler --all --all-namespaces

# Stop monitoring processes
echo "Stopping monitoring processes..."
kill $COUNT_NUM_POD_PID
echo "killed count_num_pods.py with PID: $COUNT_NUM_POD_PID"

kill -INT $POD_LOG_MONITOR_PID
kill -INT $CONTROLLER_LOG_MONITOR_PID
sleep 5  # Give monitors time to collect final logs
echo "killed log monitors"

kill $PORT_FORWARD_PID
echo "killed port-forwarding with PID: $PORT_FORWARD_PID"

# Parse logs and generate plots
echo "started parsing logs and generating plots..."
python3 plot/plot-output.py ${experiment_result_dir}
    # --output-dir ${experiment_result_dir}
    # --autoscaler ${autoscaler_mechanism} \
    # --workload ${workload_name} \

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


#####################################################################
#####################################################################
#####################################################################


# input_workload_path=$1
# autoscaler_mechanism=$2
# target_avg_rps=$3
# aibrix_repo="/Users/bytedance/projects/aibrix"
# api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
# k8s_config_dir="deepseek-llm-7b-chat-v100"
# target_deployment="aibrix-model-deepseek-llm-7b-chat"

# if [ -z "$api_key" ]; then
#     echo "API key is not set. Please set the API key in the script"
#     exit 1
# fi
# if [ -z "$input_workload_path" ]; then
#     echo "input_workload_path is not given"
#     echo "Usage: $0 <input_workload_path> <autoscaler-mechanism>"
#     exit 1
# fi
# # autoscaler_mechanism should be one of the following: "hpa", "vpa", "aibrix"
# if [ -z "$autoscaler_mechanism" ]; then
#     echo "autoscaler_mechanism is not given"
#     echo "Usage: $0 <input_workload_path> <autoscaler-mechanism>"
#     exit 1
# fi
# if [ -z "$target_avg_rps" ]; then
#     echo "${target_avg_rps} is not given. will be set to default value 5"
#     echo target_avg_rps=5
# fi


# workload_name=$(echo $input_workload_path | tr '/' '\n' | grep .jsonl | cut -d '.' -f 1)
# experiment_result_dir="experiment_results/${workload_name}-${autoscaler_mechanism}-$(date +%Y%m%d-%H%M%S)"

# if [ ! -d ${experiment_result_dir} ]; then
#     echo "output directory does not exist. Create the output directory"
#     mkdir -p ${experiment_result_dir}
# fi

# echo "workload_name: $workload_name"
# echo "input_workload_path: $input_workload_path"
# echo "experiment_result_dir: $experiment_result_dir"

# ## port-forwarding
# kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc  8888:80 &

# # echo "**********************"
# # echo "* target_deployment: $target_deployment"
# # echo "* will start in 3 seconds. check if this deployment is what you want."
# # echo "**********************"
# # sleep 3

# kubectl delete podautoscaler --all --all-namespaces
# kubectl delete hpa --all --all-namespaces

# kubectl apply -f ${k8s_config_dir}/${autoscaler_mechanism}.yaml
# echo "kubectl apply -f ${k8s_config_dir}/${autoscaler_mechanism}.yaml"

# ###########################################################
# python set_replicas_to_1.py --deployment ${target_deployment} --replicas 1
# kubectl rollout restart deploy aibrix-controller-manager -n aibrix-system
# kubectl rollout restart deploy ${target_deployment} -n default
# sleep_before_pod_check=20
# echo "Sleep for ${sleep_before_pod_check} seconds after restarting deployment"
# sleep ${sleep_before_pod_check}
# python check_k8s_is_ready.py ${target_deployment}
# ###########################################################

# echo "Experiment starts in 1 seconds!!!"
# sleep 1

# cp ${input_workload_path} ${experiment_result_dir}


# python count_num_pods.py ${target_deployment} ${experiment_result_dir} &
# COUNT_NUM_POD_PID=$!
# echo "started count_num_pods.py with PID: $COUNT_NUM_POD_PID"

# output_jsonl_path=${experiment_result_dir}/output.jsonl
# python3 ${aibrix_repo}/benchmarks/generator/client.py \
#     --workload-path ${input_workload_path} \
#     --endpoint "localhost:8888" \
#     --model deepseek-llm-7b-chat \
#     --api-key ${api_key} \
#     --output-dir ${experiment_result_dir} \
#     --output-file-path ${output_jsonl_path} \
#     --routing-strategy least-request
#     # --target_avg_rps ${target_avg_rps}

# echo "Experiment is done. date: $(date)"

# kubectl delete podautoscaler --all --all-namespaces

# kill $COUNT_NUM_POD_PID
# echo "killed count_num_pods.py with PID: $COUNT_NUM_POD_PID"


# pod_log_dir="${experiment_result_dir}/pod_logs"
# echo "started dumping pod logs for ${target_deployment} to ${pod_log_dir}"
# python dump_pod_log.py ${target_deployment} ${pod_log_dir} default vllm-openai
# echo "done dumping pod logs for ${target_deployment} to ${pod_log_dir}"

# echo "started dumping pod logs for aibrix-controller-manager to ${pod_log_dir}"
# python dump_pod_log.py aibrix-controller-manager ${pod_log_dir} aibrix-system manager
# echo "done dumping pod logs for aibrix-controller-manager to ${pod_log_dir}"

# echo "started parsing application pod logs"
# python parse_application_pod_log.py ${pod_log_dir}
# echo "done parsing application pod logs"

# echo "started parsing aibrix-controller-manager pod logs"
# python3 plot/plot-output.py --autoscaling ${autoscaler_mechanism} --workload ${workload_name} --output-dir ${experiment_result_dir}
# echo "done parsing aibrix-controller-manager pod logs"

# cp output.txt ${experiment_result_dir}
# echo "copied output.txt to ${experiment_result_dir}"
