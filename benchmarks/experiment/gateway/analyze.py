import json
import sys
import numpy as np

# Load the JSONL file
file_path = sys.argv[1]
data = []

with open(file_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

# Extract metrics
latencies = [item["latency"] for item in data]
throughputs = [item["throughput"] for item in data]
tokens_per_second = [item["total_tokens"] / item["latency"] for item in data]

# Helper function to calculate statistics
def calculate_statistics(values):
    values = sorted(values)
    avg = sum(values) / len(values)
    median = np.median(values)
    percentile_99 = np.percentile(values, 99)
    return avg, median, percentile_99

# Calculate statistics for each metric
latency_stats = calculate_statistics(latencies)
throughput_stats = calculate_statistics(throughputs)
tokens_per_second_stats = calculate_statistics(tokens_per_second)

# Print results
print("Latency Statistics (s): Average = {:.4f}, Median = {:.4f}, 99th Percentile = {:.4f}".format(*latency_stats))
print("Throughput Statistics: Average = {:.4f}, Median = {:.4f}, 99th Percentile = {:.4f}".format(*throughput_stats))
print("Tokens per Second Statistics: Average = {:.4f}, Median = {:.4f}, 99th Percentile = {:.4f}".format(*tokens_per_second_stats))
