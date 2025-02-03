#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import sys
from datetime import datetime

def read_experiment_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

def print_key_stats(stats, autoscaler, total_cost):
    print(f"\nKey Statistics:")
    print(f"Autoscaler: {autoscaler}")
    print(f"Number of samples: {stats['Sample Count']:,.0f}")
    print(f"Latency (p50/p90/p99): {stats['Latency P50']:.3f}/{stats['Latency P90']:.3f}/{stats['Latency P99']:.3f} seconds")
    print(f"Average RPS: {stats['Average RPS']:.2f}")
    print(f"Average total tokens: {stats['Total Tokens (avg)']:.1f}")
    print(f"Total cost: {total_cost:.2f}")

def parse_experiment_output(lines):
    results = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line.strip())
            required_fields = ['status_code', 'start_time', 'end_time', 'latency', 'throughput', 'prompt_tokens', 'output_tokens', 'total_tokens', 'input', 'output']
            if any(field not in data for field in required_fields):
                continue
            results.append(data)
        except json.JSONDecodeError:
            continue
    if not results:
        raise ValueError("plot-output.py, No valid data entries found")
    
    # Sort by start_time
    results.sort(key=lambda x: x['start_time'])
    
    df = pd.DataFrame(results)
    base_time = df['start_time'].iloc[0]
    df['start_time'] = df['start_time'] - base_time # asyncio time
    df['end_time'] = df['end_time'] - base_time
    df['second_bucket'] = df['start_time'].astype(int)
    rps_series = df.groupby('second_bucket').size()
    df['rps'] = df['second_bucket'].map(rps_series)
    df['timestamp'] = df['start_time']
    df['input_length'] = df['prompt_tokens']
    df['output_length'] = df['output_tokens']

    success_rps = df[df['status_code'] == 200].groupby('second_bucket').size()
    failed_rps = df[df['status_code'] != 200].groupby('second_bucket').size()
    df['success_rps'] = df['second_bucket'].map(success_rps).fillna(0)
    df['failed_rps'] = df['second_bucket'].map(failed_rps).fillna(0)
    
    return df, base_time


def analyze_performance(df):
    try:
        stats = {
            'Sample Count': len(df),
            'Average Latency': df['latency'].mean(),
            'Latency P50': df['latency'].quantile(0.50),
            'Latency P90': df['latency'].quantile(0.90),
            'Latency P95': df['latency'].quantile(0.95),
            'Latency P99': df['latency'].quantile(0.99),
            'Latency Std Dev': df['latency'].std(),
            'Min Latency': df['latency'].min(),
            'Max Latency': df['latency'].max(),
            'Average RPS': df['rps'].mean(),
            'Success RPS': df['success_rps'].mean(),
            'Failed RPS': df['failed_rps'].mean(),
            'Total Tokens (avg)': df['total_tokens'].mean(),
            'Prompt Tokens (avg)': df['prompt_tokens'].mean(),
            'Output Tokens (avg)': df['output_tokens'].mean(),
            # 'Total Num Requests': len(df),
            'Successful Num Requests': len(df[df['status_code'] == 200]),
            'Failed Num Requests': len(df[df['status_code'] != 200]),
        }
        return stats
    except Exception as e:
        raise Exception(f"Error analyzing performance metrics: {e}")
    
def plot_latency_cdf(df, output_dir):
    latencies_sorted = np.sort(df['latency'].values)
    p = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)
    ax = plt.figure().gca()
    ax.plot(latencies_sorted, p)
    ax.set_title('Latency CDF')
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Probability')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.01)
    ax.grid(True)

    output_path = f'{output_dir}/latency_cdf.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"** Saved: {output_path}")


def plot_latency_trend(df, output_dir, intended_rps, intended_traffic, pod_count_df, autoscaler):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 9))
    fig.tight_layout(pad=3.0)

    # ax1.plot(df['start_time'], df['latency'], label="Latency", color='tab:red')
    ax1.scatter(df['end_time'], df['latency'], label="Latency", color='tab:red', marker='.', s=5)
    ax1.set_title(autoscaler)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Latency (seconds)')
    ax1.grid(True)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    ax1_twin.spines['right'].set_position(('outward', 0))
    ax1_twin.plot(pod_count_df['asyncio_time'], pod_count_df['Running'], label="Running Pods", marker='x', color='tab:purple', markersize=5)
    ax1_twin.plot(pod_count_df['asyncio_time'], pod_count_df['Pending'], label="Pending Pods", marker='v', color='tab:orange', markersize=5)
    ax1_twin.plot(pod_count_df['asyncio_time'], pod_count_df['Init'], label="Init Pods", marker='P', color='tab:green', markersize=5)
    ax1_twin.set_ylabel('Pod Count')
    ax1_twin.legend(loc='upper right')
    ax1_twin.set_ylim(bottom=0)
    ax1_twin.set_xlim(left=0, right=pod_count_df['asyncio_time'].max())

    # third axis
    ax1_twin2 = ax1.twinx()
    ax1_twin2.spines['right'].set_position(('outward', 60))
    ax1_twin2.plot(pod_count_df['asyncio_time'], pod_count_df['accumulated_cost'], label="Cost", color='tab:blue', markersize=5)
    ax1_twin2.set_ylabel('Cost')
    ax1_twin2.legend(loc='lower right')
    ax1_twin2.set_ylim(bottom=0)
    ax1_twin2.set_xlim(left=0, right=pod_count_df['asyncio_time'].max())



    # ax2.plot(df['start_time'], df['rps'], label="Load (RPS)", color='tab:blue', linewidth=0.5)
    ax2.plot(df['start_time'], df['success_rps'], label="Goodput", color='tab:green', linewidth=0.5)
    ax2.plot(df['start_time'], df['failed_rps'], label="Failed RPS", color='tab:red', linewidth=0.5)
    ax2.set_ylabel('Client sending RPS', color='tab:blue')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper left')
    
    # ax2_twin = ax2.twinx()    
    # ax2_twin.plot([ts for ts, _ in intended_rps], [rps for _, rps in intended_rps], label="Intended RPS", color='tab:orange')
    # ax2_twin.plot([ts for ts, _ in intended_traffic], [rps for _, rps in intended_traffic], label="Intended traffic", color='tab:purple')
    # ax2_twin.set_ylabel('Intended RPS', color='tab:orange')
    # ax2_twin.grid(True)
    # ax2_twin.legend(loc='upper right')
    # ax2_twin.set_ylim(bottom=0)
    # ax2_twin.set_xlim(left=0)

    ax3.plot(df['start_time'], df['input_length'], label="Input Length (tokens)", color='tab:blue', linewidth=0.5)
    ax3.set_ylabel('Input Length (tokens)', color='tab:blue')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='upper left')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['start_time'], df['output_length'], label="Output Length (tokens)", color='tab:orange', linewidth=0.5)
    ax3_twin.set_ylabel('Output Length (tokens)', color='tab:orange')
    ax3_twin.set_xlabel('Time (seconds)')
    ax3_twin.legend(loc='upper right')
    ax3_twin.set_ylim(bottom=0)
    ax3_twin.set_xlim(left=0)

    ax4.plot(df['start_time'], df['total_tokens'], label="Total Tokens", color='tab:green', linewidth=0.5)
    ax4.set_ylabel('Total Tokens')
    ax4.set_xlabel('Time (seconds)')
    ax4.legend(loc='upper left')
    ax4.grid(True)
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)

    output_path = f'{output_dir}/time_series.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"** Saved: {output_path}")

def plot_correlation(df, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
    fig.tight_layout(pad=3.0)
    
    sns.scatterplot(data=df, x='input_length', y='latency', ax=ax1)
    sns.scatterplot(data=df, x='output_length', y='latency', ax=ax2)
    sns.scatterplot(data=df, x='total_tokens', y='latency', ax=ax3)
    
    # ax1.set_title('Input Length vs Latency')
    ax1.set_xlabel('Input Length (tokens)')
    ax1.set_ylabel('Latency (seconds)')

    # ax2.set_title('Output Length vs Latency')
    ax2.set_xlabel('Output Length (tokens)')
    ax2.set_ylabel('Latency (seconds)')

    # ax3.set_title('Total Length vs Latency')
    ax3.set_xlabel('Total Length (tokens)')
    ax3.set_ylabel('Latency (seconds)')

    output_path = f'{output_dir}/correlation.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"** Saved: {output_path}")


def save_statistics(stats, output_dir, autoscaler, total_cost, num_request_to_be_sent):
    try:
        output_path = f'{output_dir}/performance_stats.txt'
        print(f"** Saving statistics to: {output_path}")
        with open(output_path, 'w') as f:
            f.write(f"Performance Statistics, autoscaler: {autoscaler}\n")
            for metric, value in stats.items():
                if 'Count' in metric:
                    f.write(f"{metric}: {value:.0f}\n")
                elif 'Tokens' in metric:
                    f.write(f"{metric}: {value:.1f}\n")
                else:
                    f.write(f"{metric}: {value:.3f}\n")
            f.write(f"Total Cost: {total_cost:.2f}\n")
            f.write(f"Total Number of Requests: {num_request_to_be_sent:.0f}\n")
        return output_path
    except Exception as e:
        raise Exception(f"Error saving statistics: {e}")
    

def calc_cost(pod_count_df, autoscaler, output_dir):
    cost_per_pod_per_hour = 1
    total_cost = 0
    for idx, row in pod_count_df.iterrows():
        if idx == 0:
            prev_time = row['asyncio_time']
            pod_count_df.at[idx, 'accumulated_cost'] = 0
            continue
        cur_time = row['asyncio_time']
        duration = (cur_time - prev_time) / 3600
        total_cost += (row['Running'] + row['Pending'] + row['Init']) * duration * cost_per_pod_per_hour
        pod_count_df.at[idx, 'accumulated_cost'] = total_cost
        prev_time = cur_time
    return total_cost

def get_autoscaler(output_dir):
    if "apa" in output_dir or "APA" in output_dir:
        autoscaler = "APA"
    elif "hpa" in output_dir or "HPA" in output_dir:
        autoscaler = "HPA"
    elif "kpa" in output_dir or "KPA" in output_dir:
        autoscaler = "KPA"
    else:
        autoscaler = "Unknown"
        print(f"plot-output.py, Warning: Could not determine autoscaler from output directory name: {output_dir}")
        assert False
    return autoscaler



def plot_together(experiment_home_dir):
    all_dir = [d for d in os.listdir(experiment_home_dir) 
               if os.path.isdir(os.path.join(experiment_home_dir, d))]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(5, 8))
    fig.tight_layout(pad=4.0)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']
    
    for idx, subdir in enumerate(all_dir):
        output_dir = os.path.join(experiment_home_dir, subdir)
        autoscaler = get_autoscaler(output_dir)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Read and parse data
        experiment_output_file = os.path.join(output_dir, "output.jsonl")
        df, asyncio_base_time = parse_experiment_output(read_experiment_file(experiment_output_file))
        
        # Read pod count data
        pod_count_df = pd.read_csv(os.path.join(output_dir, "pod_count.csv"))
        pod_count_df['asyncio_time'] = pod_count_df['asyncio_time'] - asyncio_base_time
        
        # 1. Latency CDF
        latencies_sorted = np.sort(df['latency'].values)
        p = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)
        ax1.plot(latencies_sorted, p, label=f'{autoscaler}', color=color)
        
        # 2. Latency over time
        ax2.scatter(df['end_time'], df['latency'], 
                   label=f'{autoscaler}', color=color, 
                   marker='.', s=5, alpha=0.3)
        
        # 3. Running pods over time
        ax3.plot(pod_count_df['asyncio_time'], pod_count_df['Running'],
                label=f'{autoscaler}', color=color,
                marker=marker, markersize=4, markevery=0.1)
        
        # 4. Accumulated cost over time
        # Calculate accumulated cost if not present
        if 'accumulated_cost' not in pod_count_df.columns:
            cost_per_pod_per_hour = 1
            total_cost = 0
            pod_count_df['accumulated_cost'] = 0
            for i in range(1, len(pod_count_df)):
                duration = (pod_count_df.iloc[i]['asyncio_time'] - 
                          pod_count_df.iloc[i-1]['asyncio_time']) / 3600
                total_cost += (pod_count_df.iloc[i]['Running'] + 
                             pod_count_df.iloc[i]['Pending'] + 
                             pod_count_df.iloc[i]['Init']) * duration * cost_per_pod_per_hour
                pod_count_df.iloc[i, pod_count_df.columns.get_loc('accumulated_cost')] = total_cost
        
        ax4.plot(pod_count_df['asyncio_time'], pod_count_df['accumulated_cost'],
                label=f'{autoscaler}', color=color,
                marker=marker, markersize=4, markevery=0.1)
    
    # Configure plots
    ax1.set_title('Latency CDF')
    ax1.set_xlabel('Latency (seconds)')
    ax1.set_ylabel('Probability')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0, top=1.01)
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('Latency')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Latency (seconds)')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True)
    ax2.legend()
    
    ax3.set_title('Running Pods')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('# Running Pods')
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    ax3.grid(True)
    ax3.legend()
    
    ax4.set_title('Accumulated Cost')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Accumulated Cost')
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)
    ax4.grid(True)
    ax4.legend()
    
    # Save the figure
    output_path = os.path.join(experiment_home_dir, 'plot_together.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print("="*60)
    print(f"** Saved comparison plots to: {output_path}")
    print("="*60)


def main():
    experiment_home_dir = sys.argv[1]
    plot_together(experiment_home_dir)
    all_dir = os.listdir(experiment_home_dir)
    all_dir = [d for d in all_dir if os.path.isdir(f"{experiment_home_dir}/{d}")]
    for subdir in all_dir:
        print(f"subdir: {subdir}")
    for subdir in all_dir:
        output_dir = f"{experiment_home_dir}/{subdir}"
        autoscaler = get_autoscaler(output_dir)
        try:
            experiment_output_file = f"{output_dir}/output.jsonl"
            print("Generating report...")
            lines = read_experiment_file(experiment_output_file)
            df, asyncio_base_time = parse_experiment_output(lines)
            with open(f"{output_dir}/intended_rps.csv", 'r', encoding='utf-8') as f_:
                intended_rps = [tuple(map(int, line.strip().split(','))) for line in f_.readlines()]
            with open(f"{output_dir}/intended_traffic.csv", 'r', encoding='utf-8') as f_:
                intended_traffic = [tuple(map(float, line.strip().split(','))) for line in f_.readlines()]
            with open(f"{output_dir}/pod_count.csv", 'r', encoding='utf-8') as f_:
                pod_count_df = pd.read_csv(f_)
                pod_count_df['asyncio_time'] = pod_count_df['asyncio_time'] - asyncio_base_time
            total_cost = calc_cost(pod_count_df, autoscaler, output_dir)
            all_files = os.listdir(output_dir)
            num_request_to_be_sent = 0
            for fn in all_files:
                if fn.endswith(".jsonl") and fn != "output.jsonl":
                    with open(f"{output_dir}/{fn}", 'r', encoding='utf-8') as f_:
                        lines = read_experiment_file(f"{output_dir}/{fn}")
                        num_request_to_be_sent = len(lines)
            if num_request_to_be_sent == 0:
                print("plot-output.py, ERROR: No valid data entries found")
            stats = analyze_performance(df)
            stats_file = save_statistics(stats, output_dir, autoscaler, total_cost, num_request_to_be_sent)
            # print("\nGenerating plots...")
            plot_latency_cdf(df, output_dir)
            plot_latency_trend(df, output_dir, intended_rps, intended_traffic, pod_count_df, autoscaler)
            plot_correlation(df, output_dir)
            # print_key_stats(stats, autoscaler, total_cost)

        except Exception as e:
            print(f"plot-output.py, Error: {e}", file=sys.stderr)
            assert False
        print("="*60)
if __name__ == "__main__":
    exit(main())