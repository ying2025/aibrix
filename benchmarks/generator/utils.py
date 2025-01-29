import logging
import json
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Union, Dict, Any, Optional
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from datetime import datetime

def convert_to_stat_df(qps_file: str, 
                       input_file: str, 
                       output_file: str,
                       internal_trace_type: str) -> pd.DataFrame:
    if internal_trace_type == "maas":
        # Load CSV files into DataFrames
        qps_df = pd.read_csv(qps_file)
        input_len_df = pd.read_csv(input_file)
        output_len_df = pd.read_csv(output_file)

        # Rename columns for merging and clarity
        input_len_df.rename(columns={"P50": "input_len_p50", "P70": "input_len_p70", "P90": "input_len_p90", "P99": "input_len_p99"}, inplace=True)
        output_len_df.rename(columns={"P50": "output_len_p50", "P70": "output_len_p70", "P95": "output_len_p90", "P99": "output_len_p99"}, inplace=True)
        qps_df.rename(columns={"Success": "qps_success"}, inplace=True)

        # Merge DataFrames on the 'Time' column (now renamed to 'timestamp')
        merged_df = pd.merge(input_len_df, output_len_df, on="Time")
        merged_df = pd.merge(merged_df, qps_df, on="Time")

        # Drop unwanted columns (if needed)
        merged_df.drop(columns=["Total", "5xx Error", "4xx Error"], inplace=True)

        # Rename the 'Time' column to 'timestamp'
        merged_df.rename(columns={"Time": "timestamp"}, inplace=True)

        # Rearrange columns to match the desired order
        merged_df = merged_df[[
            "timestamp",
            "input_len_p50", "input_len_p70", "input_len_p90", "input_len_p99",
            "output_len_p50", "output_len_p70", "output_len_p90", "output_len_p99",
            "qps_success"
        ]]
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    elif internal_trace_type == "cloudide":
        if input_file != output_file:
            logging.error(f"input file {input_file} does not match output_file {output_file}")
        df = pd.read_csv(input_file, parse_dates=['Time'])
        df = df.replace("undefined", 0)
        df['Time'] = pd.to_datetime(df['Time'], unit = 'ms')  # Ensure timestamp is a datetime object
        df = df.set_index('Time')  # Set 'Time' as index for rolling window calculation
        df_rate = pd.read_csv(qps_file, parse_dates=['Time'])
        df_rate.columns.values[1] = "Rate"
        df_rate = df_rate.replace("undefined", 0)
        df_rate['Time'] = pd.to_datetime(df_rate['Time'], unit = 'ms') 
        df_rate = df_rate.set_index('Time')
        
        
        sent_columns = df.filter(regex = r'^sent_bytes.rate@')
        sent_columns = sent_columns.apply(pd.to_numeric, errors='coerce').fillna(0)
        df['sent'] = sent_columns.sum(axis = 1)
        
        recv_columns = df.filter(regex = r'^recv_bytes.rate@')
        recv_columns = recv_columns.apply(pd.to_numeric, errors='coerce').fillna(0)
        df['recv'] = recv_columns.sum(axis = 1)
        
        df_merged = pd.merge(df, df_rate, left_index=True, right_index=True, how='outer')
        df_merged = df_merged.fillna(0)
        df_merged = df_merged.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        df_merged['sent_rate'] = df_merged.apply(lambda row : 0 if row['Rate'] == 0 else row['sent'] / row['Rate'], axis=1)
        df_merged['recv_rate'] = df_merged.apply(lambda row : 0 if row['Rate'] == 0 else row['recv'] / row['Rate'], axis=1)
        
        df_merged = df_merged.reset_index()
        merged_df = pd.DataFrame({
            "timestamp": df_merged['Time'],
            "input_len_p50": df_merged['recv_rate'], 
            "input_len_p70": df_merged['recv_rate'],
            "input_len_p90": df_merged['recv_rate'],
            "input_len_p99": df_merged['recv_rate'],
            "output_len_p50": df_merged['sent_rate'], 
            "output_len_p70": df_merged['sent_rate'], 
            "output_len_p90": df_merged['sent_rate'], 
            "output_len_p99": df_merged['sent_rate'], 
            "qps_success":df_merged['Rate'], 
        })
        df_merged.to_csv("df_merged.csv")
    return merged_df

def get_sample_interval_ms(file_path):
    # Initialize variables
    timestamps = []

    # Read the file and extract the first two timestamps
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if 'Time' in row and row['Time']:
                # Parse the timestamp
                timestamps.append(datetime.strptime(row['Time'], "%Y-%m-%d %H:%M:%S"))
            # Stop after reading the first two timestamps
            if len(timestamps) == 2:
                break

    # Calculate the interval in milliseconds
    interval = None
    if len(timestamps) == 2:
        interval = int((timestamps[1] - timestamps[0]).total_seconds() * 1000)
        logging.info(f"Sampling interval: {interval} milliseconds")
    else:
        logging.error("Insufficient data to calculate the sampling interval.")
    return interval


def make_serializable(data):
    """Recursively convert data into JSON serializable types."""
    if isinstance(data, list):
        return [make_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(make_serializable(item) for item in data)
    elif isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}
    elif isinstance(data, (np.integer, np.int64)):  # Convert NumPy int types to int
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):  # Convert NumPy float types to float
        return float(data)
    else:
        return data


def get_tokenizer(
        pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                         trust_remote_code=trust_remote_code)

def plot_rps_workload(workload_dict: Dict[str, List[Dict[str, Any]]], output_file: str = None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    percentiles = [50, 70, 95, 99]
    window_size = 30000  # 10 seconds in milliseconds
    
    for workload_name, workload in workload_dict.items():
        # Extract data in single pass
        timestamps, input_lens, output_lens = zip(*[
            (req['timestamp'], req['requests'][0].input_token_len, req['requests'][0].output_token_len) 
            for req in workload
        ])
        timestamps = np.array(timestamps)
        input_lens = np.array(input_lens)
        output_lens = np.array(output_lens)

        # Calculate RPS using 1-second windows
        min_time = timestamps.min()
        max_time = timestamps.max()
        second_bins = np.arange(min_time, max_time + 1000, 1000)  # 1s bins for RPS
        hist, _ = np.histogram(timestamps, bins=second_bins)
        bin_centers_sec = (second_bins[:-1] + second_bins[1:]) / 2000  # convert to seconds

        # Plot simple RPS
        ax1.plot(bin_centers_sec, hist, label=f"{workload_name}", alpha=0.7)

        # Create 10-second bins for token statistics
        bins = np.arange(min_time, max_time + window_size, window_size)  # 10s bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_centers_10sec = bin_centers / 1000  # convert to seconds
        n_bins = len(bins) - 1
        bin_indices = np.digitize(timestamps, bins)

        # Initialize arrays for token statistics
        input_stats = np.zeros((n_bins, len(percentiles) + 1))
        output_stats = np.zeros((n_bins, len(percentiles) + 1))
        
        # Calculate token statistics for each 10-second window
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.any(mask):
                # Input token statistics
                bin_input = input_lens[mask]
                input_stats[i-1, 0] = np.mean(bin_input)
                input_stats[i-1, 1:] = np.percentile(bin_input, percentiles)
                
                # Output token statistics
                bin_output = output_lens[mask]
                output_stats[i-1, 0] = np.mean(bin_output)
                output_stats[i-1, 1:] = np.percentile(bin_output, percentiles)
        
        # Plot token statistics
        labels = ['avg', 'p50', 'p70', 'p95', 'p99']
        
        # Plot input token statistics
        for j, label in enumerate(labels):
            ax2.plot(bin_centers_10sec, input_stats[:, j],
                    label=f"{workload_name} {label}", alpha=0.7)
        
        # Plot output token statistics
        for j, label in enumerate(labels):
            ax3.plot(bin_centers_10sec, output_stats[:, j],
                    label=f"{workload_name} {label}", alpha=0.7)
    
    # Format plots
    ax1.set_title('Request Rate (per second)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Requests per Second')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    ax2.set_title('Input Token Length Statistics (10s windows)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Input Token Length')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(bottom=0)
    
    ax3.set_title('Output Token Length Statistics (10s windows)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Output Token Length')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        logging.info(f'Saved workload plot to {output_file}')
    else:
        plt.show()
    plt.close()

def plot_workload(workload_dict, interval_ms, output_file: str = None):
    """
    Plots the concurrency (item length) of the generated workload.

    Args:
        workload_dict (dict): A dictionary where the keys are workload names (labels) and the values are lists of lists representing the workload.
        interval_ms (int): Interval in milliseconds. 
    """
    fig, ax = plt.subplots()
    for workload_name, workload in workload_dict.items():
        concurrency_values = [len(item["requests"]) for item in workload]
        ax.plot(np.arange(len(concurrency_values)) * interval_ms, concurrency_values, label=workload_name)

    ax.set_ylim(0, )
    plt.xlabel('Time (ms)')
    plt.ylabel('Concurrency')
    plt.title('Workload Concurrency')
    plt.legend()
    if output_file is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(f"{output_file}-traffic.pdf")
        logging.info(f'Saved traffic plot to {output_file}-traffic.pdf')
        
        
    fig, ax = plt.subplots()
    for workload_name, workload in workload_dict.items():
        input_lengths = [item["requests"][0]['prompt_length'] for item in workload]
        output_lengths = [item["requests"][0]['output_length'] for item in workload]
        ax.plot(np.arange(len(concurrency_values)) * interval_ms, input_lengths, label=f"{workload_name} prompt_length")
        ax.plot(np.arange(len(concurrency_values)) * interval_ms, output_lengths, label=f"{workload_name} output_length")

    ax.set_ylim(0, )
    plt.xlabel('Time (ms)')
    plt.ylabel('Lengths')
    plt.title('Request Sizes')
    plt.legend()
    if output_file is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(f"{output_file}-requests.pdf")
        logging.info(f'Saved traffic plot to {output_file}-requests.pdf')


def save_workload(load_struct: List[Any],
                  output_path: str,
                  use_jsonl: Optional[bool] = False):
    # create the path if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if use_jsonl:
        with open(output_path + ".jsonl", "w") as file:
            for row in load_struct:
                json_line = json.dumps(row)  # Convert list to JSON string
                file.write(json_line + "\n")
            logging.warn(f'Saved workload file to {output_path + ".jsonl"}')
    else:
        with open(output_path + ".json", 'w') as file:
            json.dump(load_struct, file, indent=4)
        logging.warn(f'Saved workload file to {output_path + ".json"}')




def load_workload(input_path: str) -> List[Any]:
    try:
        if input_path.endswith(".jsonl"):
            with open(input_path, "r") as file:
                load_struct = [json.loads(line) for line in file]
        else:
            with open(input_path, "r") as file:
                load_struct = json.load(file)
        return load_struct
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_path}: {str(e)}")
    except IOError as e:
        raise ValueError(f"Error reading file {input_path}: {str(e)}")


# Function to wrap the prompt into OpenAI's chat completion message format.
def wrap_prompt_as_chat_message(prompt: str):
    """
    Wrap the prompt into OpenAI's chat completion message format.

    :param prompt: The user prompt to be converted.
    :return: A list containing chat completion messages.
    """
    return prompt
    # user_message = {"role": "user", "content": prompt}
    # return user_message
    # return [user_message] # original