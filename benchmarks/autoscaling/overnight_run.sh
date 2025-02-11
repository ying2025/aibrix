#!/bin/bash

workload_path=$1
if [ -z "${workload_path}" ]; then
    echo "workload path is not given"
    echo "Usage: $0 <workload_path>"
    exit 1
fi

# If you don't want to deploy any autoscaler use none. e.g., autoscalers="none"
autoscalers="hpa kpa apa optimizer-kpa"
routing_policies="random least-request least-kv-cache least-busy-time least-latency throughput"

for routing in ${routing_policies}; do
    for autoscaler in ${autoscalers}; do
        start_time=$(date +%s)
        echo "--------------------------------"
        echo "started experiment at $(date)"
        echo autoscaler: ${autoscaler}
        echo routing: ${routing}
        echo workload: ${workload_path} 
        echo "The stdout/stderr is being logged in ./output.txt"
        ./run-test.sh ${workload_path} ${autoscaler} ${routing} &> output.txt
        end_time=$(date +%s)
        echo "Done: Time taken: $((end_time-start_time)) seconds"
        echo "--------------------------------"
        sleep 10
    done
done