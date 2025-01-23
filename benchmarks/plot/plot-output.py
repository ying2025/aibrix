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
            required_fields = ['output', 'prompt_tokens', 'output_tokens', 'total_tokens', 
                             'start_time', 'end_time', 'latency', 'throughput']
            if any(field not in data for field in required_fields):
                continue
            results.append(data)
        except json.JSONDecodeError:
            continue
    
    if not results:
        raise ValueError("No valid data entries found")
    
    # Sort by start_time
    results.sort(key=lambda x: x['start_time'])
    
    df = pd.DataFrame(results)
    base_time = df['start_time'].iloc[0]
    df['start_time'] = df['start_time'] -base_time
    df['end_time'] = df['end_time'] - base_time
    df['second_bucket'] = df['start_time'].astype(int)
    rps_series = df.groupby('second_bucket').size()
    df['rps'] = df['second_bucket'].map(rps_series)
    df['timestamp'] = df['start_time']
    df['input_length'] = df['prompt_tokens']
    df['output_length'] = df['output_tokens']
    
    return df


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
            'Total Tokens (avg)': df['total_tokens'].mean(),
            'Prompt Tokens (avg)': df['prompt_tokens'].mean(),
            'Output Tokens (avg)': df['output_tokens'].mean(),
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


def plot_latency_trend(df, output_dir):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 15))
    fig.tight_layout(pad=3.0)

    ax1.plot(df['start_time'], df['latency'])
    ax1.set_title('Latency over Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Latency (seconds)')
    ax1.grid(True)

    ax2.plot(df['start_time'], df['rps'])
    ax2.set_ylabel('Requests per Second')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True)

    ax3.plot(df['start_time'], df['input_length'])
    ax3.set_ylabel('Input Length (tokens)')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True)

    ax4.plot(df['start_time'], df['output_length'])
    ax4.set_ylabel('Output Length (tokens)')
    ax4.set_xlabel('Time (seconds)')
    ax4.grid(True)

    ax5.plot(df['start_time'], df['total_tokens'])
    ax5.set_ylabel('Total Tokens')
    ax5.set_xlabel('Time (seconds)')
    ax5.grid(True)

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
    parser.add_argument('--input_file', type=str, help='Path to the experiment output file (JSONL format)')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save the analysis results')
    args = parser.parse_args()
    try:
        if args.output_dir != './output':
            output_dir = f"output/{args.output_dir}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        print("Generating report...")
        print(f"** Output directory: {output_dir}")
        print(f"Reading data from {args.input_file}...")
        lines = read_experiment_file(args.input_file)
        df = parse_experiment_output(lines)
        print(f"Successfully parsed {len(df)} valid entries")
        print("\nCalculating statistics...")
        stats = analyze_performance(df)
        stats_file = save_statistics(stats, output_dir)
        print(f"Statistics saved to: {stats_file}")

        print("\nGenerating plots...")
        plot_latency_cdf(df, output_dir)
        plot_latency_trend(df, output_dir)
        plot_correlation(df, output_dir)

        print("\nKey Statistics:")
        print(f"Number of samples: {stats['Sample Count']:,.0f}")
        print(f"Latency (p50/p90/p99): {stats['Latency P50']:.3f}/{stats['Latency P90']:.3f}/{stats['Latency P99']:.3f} seconds")
        print(f"Average RPS: {stats['Average RPS']:.2f}")
        print(f"Average total tokens: {stats['Total Tokens (avg)']:.1f}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0
if __name__ == "__main__":
    exit(main())