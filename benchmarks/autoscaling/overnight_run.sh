#!/bin/bash

if [ -z "$1" ]; then
    echo "workload path is not given"
    echo "Usage: $0 <workload_path>"
    exit 1
fi

for workload_path in $1; do
    # for routing in "random" "least-request" "least-kv-cache" "least-busy-time" "least-latency" "throughput"; do
    for routing in "random" "least-request" "least-latency" "throughput"; do
    # for routing in "least-kv-cache" "least-busy-time" "least-latency" "throughput"; do
    # for routing in "least-kv-cache" "least-busy-time"; do
    # for routing in "least-kv-cache"; do
        for autoscaling_mechanism in "none"; do
            start_time=$(date +%s)
            echo "started experiment at $(date)"
            echo autoscaler: ${autoscaling_mechanism}
            echo routing: ${routing}
            echo workload: ${workload_path} 
            ./run-test.sh ${workload_path} ${autoscaling_mechanism} ${routing} &> output.txt
            echo "done"
            end_time=$(date +%s)
            echo "Time taken: $((end_time-start_time)) seconds"
            sleep 10
        done
    done
done