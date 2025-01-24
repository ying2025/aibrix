#!/bin/bash
set -x
AIBRIX_PATH=/root/aibrix
CLIENT_PATH=${AIBRIX_PATH}/benchmarks/client
WORKLOAD_PATH=${AIBRIX_PATH}/benchmarks/workload/constant.jsonl
MODEL="deepseek-llm-7b-chat"

export KUBECONFIG=~/.kube/config-vke
# kubectl get svc -A | grep envoy-aibrix-system-aibrix
# kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc 8888:80

## Running routing through client
## "random", "least-request", "throughput", "least-kv-cache", "least-busy-time", "least-latency" 
## https://github.com/aibrix/aibrix/blob/main/pkg/plugins/gateway/gateway.go

for STRATEGY in  "least-request" "throughput" "least-kv-cache" "least-busy-time" "least-latency" "random"
do
    python ${CLIENT_PATH}/client.py --workload-path ${WORKLOAD_PATH} --endpoint "http://localhost:8888" --model $MODEL --routing-strategy $STRATEGY --api-key sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi --output-file-path output-${STRATEGY}.jsonl
    python analyze.py output-${STRATEGY}.jsonl
done


#pip install --index https://pypi.tuna.tsinghua.edu.cn/simple requests
