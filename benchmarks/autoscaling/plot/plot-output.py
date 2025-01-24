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
    
def parse_experiment_output(lines):
    results = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line.strip())
            """
            status_code: response.status_code,
            "start_time": start_time,
            "end_time": None,
            "latency": None,
            "throughput": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "input": prompt,
            "output": None,
            """
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
            'Total Num Requests': len(df),
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


def plot_latency_trend(df, output_dir, intended_rps, intended_traffic, pod_count_df):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 9))
    fig.tight_layout(pad=3.0)

    ax1.plot(df['start_time'], df['latency'], label="Latency", color='tab:red')
    ax1.set_title('Latency over Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Latency (seconds)')
    ax1.grid(True)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(pod_count_df['asyncio_time'], pod_count_df['Running'], label="Running Pods", marker='o', color='tab:purple')
    ax1_twin.plot(pod_count_df['asyncio_time'], pod_count_df['Pending'], label="Pending Pods", marker='x', color='tab:orange')
    ax1_twin.plot(pod_count_df['asyncio_time'], pod_count_df['Init'], label="Init Pods", marker='P', color='tab:green')
    ax1_twin.set_ylabel('Pod Count')
    ax1_twin.legend(loc='upper right')
    ax1_twin.set_ylim(bottom=0)


    ax2.plot(df['start_time'], df['rps'], label="Client sending rate (RPS)", color='tab:blue')
    ax2.plot(df['start_time'], df['success_rps'], label="Successful RPS", color='tab:green')
    ax2.plot(df['start_time'], df['failed_rps'], label="Failed RPS", color='tab:red')
    ax2.set_ylabel('Client sending RPS', color='tab:blue')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    ax2_twin = ax2.twinx()    
    ax2_twin.plot([ts for ts, _ in intended_rps], [rps for _, rps in intended_rps], label="Intended RPS", marker='o', color='tab:orange')
    # ax2_twin.plot([ts for ts, _ in intended_traffic], [rps for _, rps in intended_traffic], label="Intended traffic", marker='o', color='tab:purple')
    ax2_twin.set_ylabel('Intended RPS', color='tab:orange')
    ax2_twin.grid(True)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2_twin.set_ylim(bottom=0)
    ax2_twin.set_xlim(left=0)

    ax3.plot(df['start_time'], df['input_length'], label="Input Length (tokens)", color='tab:blue')
    ax3.set_ylabel('Input Length (tokens)', color='tab:blue')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='upper left')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['start_time'], df['output_length'], label="Output Length (tokens)", color='tab:orange')
    ax3_twin.set_ylabel('Output Length (tokens)', color='tab:orange')
    ax3_twin.set_xlabel('Time (seconds)')
    ax3_twin.legend(loc='upper right')
    ax3_twin.set_ylim(bottom=0)
    ax3_twin.set_xlim(left=0)

    ax4.plot(df['start_time'], df['total_tokens'], label="Total Tokens", color='tab:green')
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


def save_statistics(stats, output_dir):
    try:
        output_path = f'{output_dir}/performance_stats.txt'
        print(f"** Saving statistics to: {output_path}")
        with open(output_path, 'w') as f:
            f.write("Performance Statistics:\n")
            f.write("=====================\n\n")
            for metric, value in stats.items():
                if 'Count' in metric:
                    f.write(f"{metric}: {value:,.0f}\n")
                elif 'Tokens' in metric:
                    f.write(f"{metric}: {value:,.1f}\n")
                else:
                    f.write(f"{metric}: {value:.3f}\n")
        return output_path
    except Exception as e:
        raise Exception(f"Error saving statistics: {e}")
    
def main():
    parser = argparse.ArgumentParser(description='Analyze experiment performance metrics from JSONL file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output-dir', type=str, help='Path to the experiment output file (JSONL format)')
    args = parser.parse_args()
    try:
        experiment_output_file = f"{args.output_dir}/output.jsonl"
        print("Generating report...")
        print(f"Reading data from {experiment_output_file}...")
        lines = read_experiment_file(experiment_output_file)
        df, asyncio_base_time = parse_experiment_output(lines)
        print(f"Successfully parsed {len(df)} valid entries")
        print("\nCalculating statistics...")
        stats = analyze_performance(df)
        stats_file = save_statistics(stats, args.output_dir)
        print(f"Statistics saved to: {stats_file}")

        with open(f"{args.output_dir}/intended_rps.csv", 'r', encoding='utf-8') as f_:
            intended_rps = [tuple(map(int, line.strip().split(','))) for line in f_.readlines()]
        with open(f"{args.output_dir}/intended_traffic.csv", 'r', encoding='utf-8') as f_:
            intended_traffic = [tuple(map(float, line.strip().split(','))) for line in f_.readlines()]
        with open(f"{args.output_dir}/pod_count.csv", 'r', encoding='utf-8') as f_:
            """
            pod_count.csv format
            Deployment,Running,Pending,Init,datetimestampe,unixtime,asyncio_time
            aibrix-model-deepseek-llm-7b-chat,6,0,0,2025-01-24_11-43-49,1737747829.320288,476011.363529541
            aibrix-model-deepseek-llm-7b-chat,6,0,0,2025-01-24_11-44-08,1737747848.648963,476030.69220275
            aibrix-model-deepseek-llm-7b-chat,6,0,0,2025-01-24_11-44-10,1737747850.5636191,476032.606575375
            aibrix-model-deepseek-llm-7b-chat,6,0,0,2025-01-24_11-44-12,1737747852.629598,476034.672520916
            """
            pod_count_df = pd.read_csv(f_)
            pod_count_df['asyncio_time'] = pod_count_df['asyncio_time'] - asyncio_base_time

        print("\nGenerating plots...")
        plot_latency_cdf(df, args.output_dir)
        plot_latency_trend(df, args.output_dir, intended_rps, intended_traffic, pod_count_df)
        plot_correlation(df, args.output_dir)

        print("\nKey Statistics:")
        print(f"Number of samples: {stats['Sample Count']:,.0f}")
        print(f"Latency (p50/p90/p99): {stats['Latency P50']:.3f}/{stats['Latency P90']:.3f}/{stats['Latency P99']:.3f} seconds")
        print(f"Average RPS: {stats['Average RPS']:.2f}")
        print(f"Average total tokens: {stats['Total Tokens (avg)']:.1f}")
    except Exception as e:
        print(f"plot-output.py, Error: {e}", file=sys.stderr)
        assert False
    return 0
if __name__ == "__main__":
    exit(main())