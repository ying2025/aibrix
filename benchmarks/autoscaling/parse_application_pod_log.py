import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(filepath):
    pod_name = os.path.basename(filepath).replace('.log.txt', '')
    metrics_data = []
    with open(filepath, 'r') as f:
        content = f.read()
    pattern = r'INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Avg prompt throughput: ([\d.]+).*?Avg generation throughput: ([\d.]+).*?Running: (\d+).*?Swapped: (\d+).*?Pending: (\d+).*?GPU KV cache usage: ([\d.]+).*?CPU KV cache usage: ([\d.]+)'
    matches = re.finditer(pattern, content)
    for match in matches:
        timestamp = datetime.strptime(match.group(1), '%m-%d %H:%M:%S')
        metrics_data.append({
            'pod': pod_name,
            'timestamp': timestamp,
            'prompt_throughput': float(match.group(2)),
            'generation_throughput': float(match.group(3)),
            'running_reqs': int(match.group(4)),
            'swapped_reqs': int(match.group(5)),
            'pending_reqs': int(match.group(6)),
            'gpu_cache': float(match.group(7)),
            'cpu_cache': float(match.group(8))
        })
    return pd.DataFrame(metrics_data)

def plot_metrics(logs_dir):
    all_data = pd.DataFrame()
    all_pod_logs_files = os.listdir(logs_dir)
    print(f"Found {all_pod_logs_files} log files")
    for fn in all_pod_logs_files:
        if fn.endswith('.log'):
            temp = parse_log_file(os.path.join(logs_dir, fn))
            all_data = pd.concat([all_data, temp])
    if all_data.empty or all_data['pod'].nunique() == 0:
        print("No pod log files found")
        assert False
    metrics = {
        'Throughput': ['prompt_throughput', 'generation_throughput'],
        'Requests': ['running_reqs', 'swapped_reqs', 'pending_reqs'],
        'Cache Usage': ['gpu_cache', 'cpu_cache']
    }
    plt.style.use('bmh')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for title, metric_list in metrics.items():
        fig, ax = plt.subplots(figsize=(15, 8))
        for i, metric in enumerate(metric_list):
            for j, pod in enumerate(all_data['pod'].unique()):
                pod_data = all_data[all_data['pod'] == pod]
                ax.plot(pod_data['timestamp'], pod_data[metric], 
                       label=f'{pod} - {metric}', 
                       color=colors[i % len(colors)],
                       linestyle=['-', '--', '-.'][j % 3],
                       linewidth=2, marker='o', markersize=6)
        ax.set_title(f'{title} Over Time', fontsize=16, pad=20)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel(f'{title}', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, borderaxespad=0.)
        plt.tight_layout()
        fn = f'{logs_dir}/{title.lower().replace(" ", "_")}.pdf'
        plt.savefig(fn, bbox_inches='tight')
        print(f'** Saved plot: {fn}')
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logs_dir', help='Directory containing log files')
    args = parser.parse_args()
    plot_metrics(args.logs_dir)