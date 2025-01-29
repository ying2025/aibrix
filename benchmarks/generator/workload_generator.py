import logging
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import csv
import json
import time
import numpy as np
import os
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import List, Dict
from pandas import Timedelta
from typing import List, Tuple, Dict, Any
from transformers import PreTrainedTokenizerBase
from datetime import timedelta
from sample_request import (load_requests,  sample_requests_len_range, sample_requests_all)
from sample_request import (load_sharegpt_requests,  sample_sharegpt_requests_len_range, Request, RequestEncoder, RANDOM_SEED)
from utils import (convert_to_stat_df, get_tokenizer, plot_workload, make_serializable, save_workload, plot_rps_workload, get_sample_interval_ms)
from collections import defaultdict

# Set up logging to print only warning and above level messages
logging.basicConfig(level=logging.INFO)

np.random.seed(RANDOM_SEED)


def generate_from_internal_csv(file_path: str,
                            prompt_file_path: str, 
                            duration_ms: int,
                            summary_interval_ms: int,
                            tokenizer: PreTrainedTokenizerBase,
                            interval_ms: int = 1000,
                            output_file: str = 'output/output',
                            input_trace: str = None,
                            output_trace: str = None,
                            to_jsonl: bool = False,
                            qps_scale: float = 1.0,
                            input_scale: float = 1.0,
                            output_scale: float = 1.0,
                            ) -> Dict[str, Any]:
   traffic = []
   input_lengths = []
   output_lengths = [] 
   
   # Read traffic file
   with open(file_path, 'r') as file:
       reader = csv.DictReader(file)
       for row in reader:
           if 'Total' in row:
               total_value = row['Total']
               if total_value:
                   traffic.append(float(total_value))

   # Read input lengths if provided
   if input_trace:
       with open(input_trace, 'r') as file:
           reader = csv.DictReader(file)
           for row in reader:
               if 'P50' in row:
                   length = row['P50']
                   if length:
                       input_lengths.append(round(float(length)))

   # Read output lengths if provided                    
   if output_trace:
       with open(output_trace, 'r') as file:
           reader = csv.DictReader(file)
           for row in reader:
               if 'P50' in row:
                   length = row['P50']
                   if length:
                       output_lengths.append(round(float(length)))
   
   print(f"input_lengths size {len(input_lengths)} output_lengths size {len(output_lengths)}")

   # Generate workload
   sharegpt_df = load_requests(dataset_path=prompt_file_path, tokenizer=tokenizer)
   workload = []
   total_ms = 0
   print(traffic)
   for i, interval_requests in enumerate(traffic):
       interval_requests = interval_requests / qps_scale
       mean_rate = round(interval_requests) #round(interval_requests / (summary_interval_ms / interval_ms))
       input_length = input_lengths[i] / input_scale if len(input_lengths)>0 else None
       output_length = output_lengths[i] / output_scale if len(output_lengths)>0 else None
       print(f"current total ms {total_ms} summary_interval_ms {summary_interval_ms} interval_ms {interval_ms}  mean_rate {mean_rate} input_length {input_length} output_length {output_length} interval_requests {interval_requests} ")
       concurrent_sampled_reqs = sample_requests_len_range(
           df=sharegpt_df,
           num_requests=mean_rate,
           input_lens=[input_length] * mean_rate,
           output_lens=[output_length] * mean_rate,
           initial_err_perc=0.5,
           err_step=0.05
           )
       
       for _ in range (0, int(summary_interval_ms/interval_ms)):
           if concurrent_sampled_reqs:
               workload.append({"timestamp": total_ms, "requests": concurrent_sampled_reqs})
               total_ms += interval_ms
               if total_ms >= duration_ms:
                   print(f"total ms {total_ms} ")
                   break
       if total_ms >= duration_ms:
           print(f"total ms {total_ms}")
           break
          
   workload = make_serializable(workload)
   save_workload(workload, output_file, use_jsonl=to_jsonl)
   return workload


def generate_constant(prompt_file_path: str,
                       qps: int, 
                       duration_ms: int = None,
                       interval_ms: int = None,
                       output_file: str = 'output/output',
                       to_jsonl: bool = False,
                       ) -> List[List[Any]]:
    workload = []
    ts = 0
    sharegpt_df = load_requests(dataset_path=prompt_file_path, tokenizer=tokenizer)
    while ts < duration_ms:
        concurrent_reqs = sample_requests_len_range(
            df=sharegpt_df,
            num_requests=qps,
            input_lens=[None] * qps, 
            output_lens=[None] * qps, 
            initial_err_perc=0.5,
            err_step=0.05
        )
        if concurrent_reqs:  # Only add non-empty groups
            workload.append({"timestamp": ts, "requests": concurrent_reqs})  
        else:
            logging.error(f"sampled return {concurrent_reqs}")
        ts += interval_ms
    ### Generate constant load for all requests
    # idx = 0
    # while idx < len(sharegpt_df):
    #     concurrent_reqs = sample_requests_all(df=sharegpt_df, start_idx=idx, qps=qps)
    #     workload.append({"timestamp": ts, "requests": concurrent_reqs})  
    #     idx += qps
    #     ts += interval_ms
   
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload

def generate_synthetic(prompt_file_path: str,
                       A=1, B=1,
                       sigma=0.1,
                       only_rise: bool = False,
                       omega: float = None,
                       period=0.25,
                       length: int = None,
                       duration_ms: int = None,
                       interval_ms: int = None,
                       output_file: str = 'output/output',
                       to_jsonl: bool = False,
                       ) -> List[List[Any]]:
    """
    Generates a workload based on a given list of input requests and a concurrency function.

    The concurrency function is defined as:
        concurrency(t) = trend(t) + noise
        trend(t) = A * sin(omega * t) + B
        noise ~ N(0, sigma^2)

    Args:
        input_requests (list): The list of all requests to be sent.
        A (float, optional): The amplitude of the sine wave in the concurrency function. Defaults to 1.
        B (float, optional): The vertical shift of the sine wave in the concurrency function. Defaults to 1.
        sigma (float, optional): The standard deviation of the normal distribution for the noise. Defaults to 0.1.
        omega (float, optional): if None, omega = pi / (2 * length / period)
        period (float, optional): See omega. Defaults to 0.25.
        only_rise: if True, the concurrency will monotonically increase
        length (int, optional): if None, length = duration_ms / interval_ms
        duration_ms (int, optional): See param: length
        interval_ms (int, optional): See param: length

    Returns:
        list: A list of items, where each item is a list of requests to be sent concurrently.
    """

    def math_function(t):
        """
        Calculates the concurrency value based on the given concurrency function.

        The concurrency function is defined as:
        concurrency(t) = trend(t) + noise
        trend(t) = A * sin(omega * t) + B
        noise ~ N(0, sigma^2)

        Args:
            t (int): The discrete integer value of t, starting from 0.

        Returns:
            int: The concurrency value rounded to the nearest integer.
        """
        trend = A * math.sin(omega * t) + B
        noise = random.gauss(0, sigma)
        return round(trend + noise)

    assert length is not None or (duration_ms is not None and interval_ms is not None), \
        "duration_ms and interval_ms must be specified if length is not None"
    if length is None:
        length = int(duration_ms // interval_ms) + 1
    assert omega is not None or period is not None, "period must be specified if length is not None"
    if omega is None:
        omega = 2 * math.pi / (length / period)
    workload = []
    t = 0
    previous_concurrency = -1
    end_index = 0
    ts = 0
    base_req_id = 0
    
    sharegpt_df = load_requests(dataset_path=prompt_file_path, tokenizer=tokenizer)
    while t < length:
        current_concurrency = math_function(t)
        if only_rise:
            current_concurrency = max(previous_concurrency, current_concurrency)
            previous_concurrency = current_concurrency

        # start from last end index
        end_index += current_concurrency
        concurrent_reqs = sample_requests_len_range(
            df=sharegpt_df,
            num_requests=current_concurrency,
            input_lens=[None] * current_concurrency, 
            output_lens=[None] * current_concurrency, 
            initial_err_perc=0.5,
            err_step=0.05
        )
        workload.append({"timestamp": ts, "requests": concurrent_reqs})  
        base_req_id += current_concurrency
        ts += interval_ms
        t += 1
   
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload


def generate_synthetic_rps(
        prompt_file_path: str,
        duration_ms,
        rps_dist: List[int],
        input_token_len_dist: List[int],
        output_token_len_dist: List[int],
        qps_scale: float,
        input_scale: float,
        output_scale: float,
    ) -> List[Dict[str, Any]]:
    
    if not (len(rps_dist) == len(input_token_len_dist) == len(output_token_len_dist)):
        raise ValueError(f"All distributions must have the same length, len(rps_dist): {len(rps_dist)}, len(input_token_len_dist): {len(input_token_len_dist)}, len(output_token_len_dist): {len(output_token_len_dist)}")
    workload = []
    current_time = 0
    req_id = 0
    total_seconds = len(rps_dist)
    while current_time < total_seconds * 1000:
        time_idx = int(current_time / 1000)
        if time_idx >= total_seconds:
            time_idx = total_seconds - 1
        current_rate = rps_dist[time_idx] / qps_scale
        current_input_len = input_token_len_dist[time_idx] / input_scale
        current_output_len = output_token_len_dist[time_idx] / output_scale
        inter_arrival_time = 1000 if current_rate == 0 else np.random.exponential(scale=1000/current_rate) 
        current_time += inter_arrival_time
        if current_time < total_seconds * 1000:
            request = Request(int(current_time), req_id, current_input_len,current_output_len)
            workload.append(request)
            req_id += 1
            if current_time > duration_ms:
                break
    
    ts = time.time()
    if os.path.exists("sharegpt_df.csv"):
        print(f"Reading sharedgpt_df.csv")
        sharegpt_df = pd.read_csv("sharegpt_df.csv")
    else:
        sharegpt_df = load_requests(dataset_path=prompt_file_path, tokenizer=tokenizer)
        sharegpt_df.to_csv("sharegpt_df.csv")
        print(f"Saved sharegpt_df {len(sharegpt_df)}")
    print(f"load_sharedgpt took {int(time.time() - ts)}s")
    
    prompt_workload = []
    for request in workload:
        prompt_request = sample_sharegpt_requests_len_range(
            df=sharegpt_df,
            request=request,
            initial_err_perc=0.5,
            err_step=0.05
        )
        temp = {"timestamp": prompt_request.timestamp, "requests": [prompt_request]}
        prompt_workload.append(temp)
    return prompt_workload



def generate_sine_rps_dist(mean_rps: int, amplitude: float, period: int, total_seconds: int) -> List[int]:
    t = np.arange(total_seconds)
    mean_rps = np.full_like(t, mean_rps, dtype=float)
    mean_rps += amplitude * np.sin(2 * np.pi * t / period)
    return [max(1, int(rate)) for rate in mean_rps]

def generate_poisson_rps_dist(mean_rps:int,total_seconds:int)->List[int]:
    return np.random.poisson(lam=mean_rps,size=total_seconds).tolist()


def generate_token_len_dist(mean_len: int, amplitude: float, std: float, period: int, total_seconds: int) -> List[int]:
    t = np.arange(total_seconds)
    token_len_list = np.full_like(t, mean_len, dtype=float)
    token_len_list += amplitude * np.sin(2 * np.pi * t / period)
    normal_values = np.random.normal(1, std, size=len(t))
    token_len_list *= normal_values
    token_len_list = [int(max(1, x)) for x in token_len_list]
    return token_len_list


def generate_from_azure_csv(file_path: str,
                            prompt_file_path: str,
                            duration_ms: int,
                            tokenizer: PreTrainedTokenizerBase,
                            interval_ms: int,
                            output_file: str = 'output/output',
                            to_jsonl: bool = False,
                            ) -> List[List[Any]]:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Ensure TIMESTAMP is a datetime object
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # Define the grouping time range (e.g., 1 second)
    time_range = timedelta(milliseconds=interval_ms)

    # Initialize a list to hold the grouped requests
    grouped_requests = []

    # Group requests by the time range
    df.set_index('TIMESTAMP', inplace=True)
    current_time = df.index.min()
    tracing_file_end_time = df.index.max()
    end_time = current_time + Timedelta(milliseconds=duration_ms)
    if tracing_file_end_time < end_time:
        logging.warning(f"{tracing_file_end_time} can not cover duration {duration_ms}, cap to end time of tracing file")
        end_time = tracing_file_end_time

    logging.info(f"Start generation from time {current_time} to {end_time}")
    sharegpt_df = load_requests(dataset_path=prompt_file_path, tokenizer=tokenizer)

    ts = 0
    while current_time <= end_time:
        # Select requests within the current time range
        mask = (df.index >= current_time) & (df.index < current_time + time_range)
        group = df.loc[mask]
        input_lens = []
        output_lens = []
        for _, row in group.iterrows():
            input_lens.append(int(row['ContextTokens']))
            output_lens.append(int(row['GeneratedTokens']))
        sampled_requests = sample_requests_len_range(
            df=sharegpt_df,
            num_requests=len(input_lens),
            input_lens=input_lens,
            output_lens=output_lens,
            initial_err_perc=0.5,
            err_step=0.05
        )

        if sampled_requests:  # Only add non-empty groups
            grouped_requests.append({"timestamp": ts, "requests": sampled_requests})
        ts += interval_ms
        if ts > duration_ms:
            break
        # Move to the next time range
        current_time += time_range

    # Save to file
    grouped_requests = make_serializable(grouped_requests)
    save_workload(grouped_requests, output_file, use_jsonl=to_jsonl)

    return grouped_requests


def pair_requests_with_prompts_round_robin(workload: List[List[Any]],
                                           prompts: List[Tuple[str, int, int, None]],
                                           output_file: str = 'output/output',
                                           to_jsonl: bool = False
                                           ) -> List[List[Tuple[Any, str]]]:
    paired_workload = []
    prompt_count = len(prompts)
    for ts, requests in workload:
        requests_with_prompts = [
            prompts[request % prompt_count] for request in requests
        ]
        paired_workload.append({"timestamp": ts, "requests": requests_with_prompts})

    # Save to file
    save_workload(paired_workload, output_file, use_jsonl = to_jsonl)

    return paired_workload


def read_distribution_stats(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    time_diffs = df['timestamp'].diff().dt.total_seconds()
    section_in_seconds = int(time_diffs.mean())  # Use average time difference
    input_len_configs = []
    output_len_configs = []
    rps_configs = []
    for _, row in df.iterrows():
        input_len_configs.append({
            "p50": float(row['input_len_p50']),
            "p70": float(row['input_len_p70']),
            "p90": float(row['input_len_p90']),
            "p99": float(row['input_len_p99']),
            "period": section_in_seconds,
            "total_seconds": section_in_seconds
        })
        output_len_configs.append({
            "p50": float(row['output_len_p50']),
            "p70": float(row['output_len_p70']),
            "p90": float(row['output_len_p90']),
            "p99": float(row['output_len_p99']),
            "period": section_in_seconds,
            "total_seconds": section_in_seconds
        })
        rps_configs.append({
            "mean_rps": float(row['qps_success']),
            "amplitude": float(row['qps_success']) * 0.2,  # 20% variation
            "period": section_in_seconds,
            "total_seconds": section_in_seconds
        })
    return input_len_configs, output_len_configs, rps_configs



## New with grouping
def generate_workload_from_stats(
    qps_file: str,
    input_file: str,
    output_file: str,
    prompt_file: str,
    output_dir: str,
    duration_ms: int,
    qps_scale: float,
    input_scale: float,
    output_scale: float,
    workload_subname: str,
    internal_trace_type: str,
) -> Dict:
    merged_df = convert_to_stat_df(qps_file, input_file, output_file, internal_trace_type)
    input_len_configs, output_len_configs, rps_configs = read_distribution_stats(merged_df)
    input_len_dist = []
    output_len_dist = []
    rps_dist = []
    for config in rps_configs:
        rps_segment = generate_poisson_rps_dist(config['mean_rps'], config['total_seconds'])
        rps_dist.extend(rps_segment)
    if internal_trace_type == "maas":
        for config in input_len_configs:
            config['scale'] = input_scale
            input_segment = generate_token_len_dist_with_percentiles(**config)
            input_len_dist.extend(input_segment)
        for config in output_len_configs:
            config['scale'] = output_scale
            output_segment = generate_token_len_dist_with_percentiles(**config)
            output_len_dist.extend(output_segment)
    elif internal_trace_type == "cloudide":
        for config in input_len_configs:
            config['scale'] = input_scale
            input_segment = generate_token_len_dist_with_constant(**config)
            input_len_dist.extend(input_segment)
            output_segment = generate_token_len_dist_with_constant(**config)
            output_len_dist.extend(output_segment)
    print(f"Generated rps_dist {rps_dist} output_len_dist {output_len_dist} input_len_dist: {input_len_dist}")
    scenarios = {
        'csv-based-scenario': {
            'rps_dist': rps_dist,
            'input_token_len_dist': input_len_dist,
            'output_token_len_dist': output_len_dist,
        }
    }
    
    workload_dict = {}
    for scenario_name, scenario in scenarios.items():
        ts = time.time()
        generated_workload = generate_synthetic_rps(
            prompt_file,
            duration_ms,
            scenario['rps_dist'],
            scenario['input_token_len_dist'],
            scenario['output_token_len_dist'],
            qps_scale,
            input_scale,
            output_scale
        )
        print(f"Generated workload for {scenario_name} took {int(time.time() - ts)} seconds")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = f"{output_dir}/workload-{workload_subname}.jsonl"

        # Group requests by timestamp in dictionary
        timestamp_groups = defaultdict(list)
        for item in generated_workload:
            request = item["requests"][0]  # Each item has a single request
            timestamp = item["timestamp"]
            timestamp_groups[timestamp].append({
                "Prompt Length": request.input_token_len,
                "Output Length": request.output_token_len,
                "prompt": request.prompt
            })

        # Create merged workload with sorted timestamps
        flattened_workload = [
            {
                "timestamp": timestamp,
                "requests": requests
            }
            for timestamp, requests in sorted(timestamp_groups.items())
        ]

        with open(output_path, 'w') as f:
            for req in flattened_workload:
                json.dump(req, f, cls=RequestEncoder)
                f.write('\n')
                
        print(f"** Saved generated workload to {output_path}")
        workload_dict[scenario_name] = generated_workload
    
    return workload_dict

def generate_token_len_dist_with_percentiles(
    p50: int,
    p70: int,
    p90: int,
    p99: int,
    period: int,
    total_seconds: int,
    scale: float,
    amplitude_factor: float = 0.2  # Controls the amplitude of sinusoidal variation
) -> List[int]:
    if not (p50 < p70 < p90 < p99):
        raise ValueError("Percentiles must be strictly increasing: p50 < p70 < p90 < p99")
    if p50 <= 0:
        raise ValueError("Token lengths must be positive")
    percentiles = [0.50, 0.70, 0.90, 0.99]
    token_lengths = [p50, p70, p90, p99]
    token_lengths = [x / scale for x in token_lengths]
    log_lengths = np.log(token_lengths)
    def objective(params, percs, lengths):
        mu, sigma = params
        expected = stats.norm.ppf(percs, mu, sigma)
        return np.sum((expected - lengths) ** 2)
    result = minimize(
        objective,
        x0=[np.mean(log_lengths), np.std(log_lengths)],
        args=(percentiles, log_lengths),
        method='Nelder-Mead'
    )
    mu, sigma = result.x
    t = np.arange(total_seconds)
    amplitude = p50 * amplitude_factor
    sinusoidal_variation = amplitude * np.sin(2 * np.pi * t / period)
    base_samples = np.random.lognormal(mu, sigma, size=total_seconds)
    scale_factor = p50 / np.median(base_samples)
    token_len_list = base_samples * scale_factor + sinusoidal_variation
    token_len_list = [int(max(1, x)) for x in token_len_list]
    return token_len_list

def generate_token_len_dist_with_constant(
    p50: int,
    p70: int,
    p90: int,
    p99: int,
    period: int,
    total_seconds: int,
    scale: float,
    amplitude_factor: float = 0.2
) -> List[int]:
    rate = p50
    t = np.arange(total_seconds)
    rate = rate / scale
    amplitude = rate * amplitude_factor
    sinusoidal_variation = amplitude * np.sin(2 * np.pi * t / period)
    base_samples = np.repeat(rate, total_seconds)
    scale_factor = rate / np.median(base_samples)
    token_len_list = base_samples * scale_factor + sinusoidal_variation
    token_len_list = [int(max(1, x)) for x in token_len_list]
    return token_len_list

def calculate_percentiles(data,window_size, timestamps):
    percentiles=[]
    timestamps_aggregated=[]
    for i in range(0,len(data),window_size):
        window_data=data[i:i+window_size]
        if len(window_data)>0:
            p50=np.percentile(window_data,50)
            p70=np.percentile(window_data,70)
            p90=np.percentile(window_data,90)
            p99=np.percentile(window_data,99)
            percentiles.append([p50,p70,p90,p99])
            timestamps_aggregated.append(timestamps[i])
    return np.array(percentiles),timestamps_aggregated

def plot_distribution_comparison(
    csv_path: str,
    generated_data: Dict[str, List],
    output_path: str = "distribution_comparison.png"
):
    df=pd.read_csv(csv_path,parse_dates=['timestamp'])
    fig,axs=plt.subplots(3,1,figsize=(15,20))
    plt.subplots_adjust(hspace=0.3)
    window=len(generated_data['input_token_len_dist'])//len(df)
    timestamps=[df['timestamp'].iloc[0]+timedelta(seconds=i) for i in range(len(generated_data['input_token_len_dist']))]
    
    ax=axs[0]
    ax_twin = ax.twinx()
    ax.plot(df['timestamp'],df['input_len_p50'],'bo-',label='Original P50',alpha=0.7)
    ax_twin.plot(df['timestamp'],df['input_len_p70'],'go-',label='Original P70',alpha=0.7)
    ax_twin.plot(df['timestamp'],df['input_len_p90'],'ro-',label='Original P90',alpha=0.7)
    ax_twin.plot(df['timestamp'],df['input_len_p99'],'mo-',label='Original P99',alpha=0.7)
    gen_input=np.array(generated_data['input_token_len_dist'])
    ax_twin.plot(timestamps,gen_input,'k-',label='Generated',alpha=0.3)
    input_percentiles,input_timestamps=calculate_percentiles(gen_input,window, timestamps)
    if len(input_percentiles)>0:
        ax.plot(input_timestamps,input_percentiles[:,0],'bx-.',label='Generated P50',alpha=0.7)
        ax_twin.plot(input_timestamps,input_percentiles[:,1],'gx-.',label='Generated P70',alpha=0.7)
        ax_twin.plot(input_timestamps,input_percentiles[:,2],'rx-.',label='Generated P90',alpha=0.7)
        ax_twin.plot(input_timestamps,input_percentiles[:,3],'mx-.',label='Generated P99',alpha=0.7)
    ax.set_title('Input Length Distribution Comparison')
    ax.set_ylabel('Token Length')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax_twin.set_ylim(bottom=0)

    ax=axs[1]
    ax_twin = ax.twinx()
    ax.plot(df['timestamp'],df['output_len_p50'],'bo-',label='Original P50',alpha=0.7)
    ax_twin.plot(df['timestamp'],df['output_len_p70'],'go-',label='Original P70',alpha=0.7)
    ax_twin.plot(df['timestamp'],df['output_len_p90'],'ro-',label='Original P90',alpha=0.7)
    ax_twin.plot(df['timestamp'],df['output_len_p99'],'mo-',label='Original P99',alpha=0.7)
    gen_output=np.array(generated_data['output_token_len_dist'])
    ax_twin.plot(timestamps,gen_output,'k-',label='Generated',alpha=0.3)
    output_percentiles,output_timestamps=calculate_percentiles(gen_output,window, timestamps)
    if len(output_percentiles)>0:
        ax.plot(output_timestamps,output_percentiles[:,0],'bx-.',label='Generated P50',alpha=0.7)
        ax_twin.plot(output_timestamps,output_percentiles[:,1],'gx-.',label='Generated P70',alpha=0.7)
        ax_twin.plot(output_timestamps,output_percentiles[:,2],'rx-.',label='Generated P90',alpha=0.7)
        ax_twin.plot(output_timestamps,output_percentiles[:,3],'mx-.',label='Generated P99',alpha=0.7)
    ax.set_title('Output Length Distribution Comparison')
    ax.set_ylabel('Token Length')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax_twin.set_ylim(bottom=0)

    ax=axs[2]
    ax.plot(df['timestamp'],df['qps_success'],'bo-',label='Original QPS',alpha=0.7)
    gen_rps=np.array(generated_data['rps_dist'])
    ax.plot(timestamps,gen_rps,'k-',label='Generated RPS',alpha=0.3)
    rps_percentiles,rps_timestamps=calculate_percentiles(gen_rps,window, timestamps)
    if len(rps_percentiles)>0:
        ax.plot(rps_timestamps,rps_percentiles[:,0],'bx',label='Generated P50',alpha=0.7)
    ax.set_title('RPS/QPS Distribution Comparison')
    ax.set_ylabel('Requests per Second')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax_twin.set_ylim(bottom=0)

    for ax in axs:
        ax.set_xlabel('Time')
        plt.setp(ax.xaxis.get_majorticklabels(),rotation=45)
        ax.margins(x=0.01)
    plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.close()
    print(f"** Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Workload Generator')
    parser.add_argument('--prompt-file', type=str, required=True, help='File containing prompts.')
    parser.add_argument('--num-prompts', type=int, default=100, help='Number of prompts to sample.')
    parser.add_argument('--trace-type', type=str, required=True, choices=['constant', 'synthetic', 'internal', 'azure', 'synthetic-rps', 'synthetic-from-csv-file'],
                        help='Type of trace consumed. Choose among: synthetic, internal, azure')
    parser.add_argument('--traffic-file', type=str, required=False, default=None,
                        help='Traffic file containing times of arrival, which workload generator depends upon to '
                             'convert to traffic used in workload. This is only needed for for internal and azure trace type. ')
    parser.add_argument('--prompt-len-file', type=str, required=False, default=None,
                        help='File containing request input lengths varied by time, which workload generator depends upon to '
                             'select input prompt. This is only needed for for internal trace type. ')
    parser.add_argument('--completion-len-file', type=str, required=False, default=None,
                        help='File containing request output lengths varied by time, which workload generator depends upon to '
                             'select input prompt. This is only needed for for internal trace type. ')
    parser.add_argument('--model', type=str, required=False, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help='Target model tokenizer.')
    parser.add_argument('--group-interval-seconds', type=int, default=1, help='Grouping interval seconds.')
    parser.add_argument('--interval-ms', type=int, required=False, default=1000,
                        help='Granularity of request injection interval in milliseconds.')
    parser.add_argument('--duration-ms', type=int, default=60000, help='Duration of the trace generated.')
    parser.add_argument('--target-qps', type=int, required=False, default=1, help='Target QPS for the workload.')
    parser.add_argument('--output-dir', type=str, required=False, default="output", help='Output directory to save ')
    parser.add_argument('--qps-scale', type=float, required=False, default=1.0, help='QPS scaling factor')
    parser.add_argument('--input-scale', type=float, required=False, default=1.0, help='Input length scaling factor')
    parser.add_argument('--output-scale', type=float, required=False, default=1.0, help='Output length scaling factor')
    parser.add_argument('--summary-interval-ms', type=float, required=False, default=1.0, help='Trace aggregated intervals in milliseconds')
    parser.add_argument('--internal-trace-type', type=str, choices=['maas', 'cloudide'], default="maas", help='Type of internal traces')
    parser.add_argument('--output-format', type=str, choices=['json', 'jsonl'], default='json',
                        help='Set output data format to either .json or .jsonl (default is .json).')
    args = parser.parse_args()

    # Generate workloads and pair with prompts
    workload_dict = {}
    tokenizer = get_tokenizer(pretrained_model_name_or_path=args.model, trust_remote_code=True)
    workload_subname = None
    if args.trace_type == "constant":
        generated_workload = generate_constant(prompt_file_path=args.prompt_file, 
                                                qps=args.target_qps,
                                                duration_ms=args.duration_ms, 
                                                interval_ms=args.interval_ms,
                                                output_file=f"{args.output_dir}/{args.trace_type}",
                                                to_jsonl=(args.output_format == "jsonl"),
                                                )
    elif args.trace_type == "synthetic":
        # Define scenarios specific to synthetic type
        scenarios = {
            'quick_rising': {'duration_ms': args.duration_ms, 'interval_ms': args.interval_ms, 'A': 5, 'period': 5,
                             'only_rise': True},
            'slow_rising': {'duration_ms': args.duration_ms, 'interval_ms': args.interval_ms, 'A': 5, 'period': 0.25,
                            'only_rise': True},
            'slight_fluctuation': {'duration_ms': args.duration_ms, 'interval_ms': args.interval_ms, 'A': 5, 'B': 5,
                                   'period': 1, 'only_rise': False},
            'severe_fluctuation': {'duration_ms': args.duration_ms, 'interval_ms': args.interval_ms, 'A': 5, 'B': 10,
                                   'period': 12, 'only_rise': False},
        }
        for scenario_name, params in scenarios.items():
            params['prompt_file_path'] = args.prompt_file
            params['output_file'] = f"{args.output_dir}/{scenario_name}"
            params['to_jsonl'] = (args.output_format == "jsonl")
        for scenario_name, params in scenarios.items():
            generated_workload = generate_synthetic(**params)
            workload_dict[scenario_name] = generated_workload
    
    
    
    elif args.trace_type == "synthetic-from-csv-file":
        workload_subname = f"{args.trace_type}"
        print(f"Generating workload for {workload_subname}")
        generated_workload = generate_workload_from_stats(
            qps_file=args.traffic_file,
            input_file=args.prompt_len_file, 
            output_file=args.completion_len_file,
            prompt_file=args.prompt_file,
            output_dir=args.output_dir,
            workload_subname=workload_subname,
            qps_scale=args.qps_scale,
            input_scale=args.input_scale,
            output_scale=args.output_scale,
            duration_ms=args.duration_ms,
            internal_trace_type=args.internal_trace_type
        )

    elif args.trace_type == "synthetic-rps":
        workload_subname = f"{args.trace_type}-{args.stats_csv.split('/')[-1].split('.')[0]}"
        print(f"Generating workload for {workload_subname}")
        section_in_seconds = 300
        rps_configs = [
            {"mean_rps": 30, "amplitude": 10, "period": 60, "total_seconds": section_in_seconds},
            {"mean_rps": 50, "amplitude": 10, "period": 60, "total_seconds": section_in_seconds},
        ]
        input_len_configs = [
            {"mean_len": 5000, "amplitude": 200, "std": 0.5, "period": 60, "total_seconds": section_in_seconds},
            {"mean_len": 4000, "amplitude": 200, "std": 0.5, "period": 60, "total_seconds": section_in_seconds},
        ]
        output_len_configs = [
            {"mean_len": 500, "amplitude": 200, "std": 0.5, "period": 60, "total_seconds": section_in_seconds},
            {"mean_len": 1000, "amplitude": 200, "std": 1, "period": 60, "total_seconds": section_in_seconds},
        ]
        input_len_dist = []
        output_len_dist = []
        rps_dist = []
        for config in input_len_configs:
            input_segment = generate_token_len_dist(**config)
            input_len_dist.extend(input_segment)
        for config in output_len_configs:
            output_segment = generate_token_len_dist(**config)
            output_len_dist.extend(output_segment)
        for config in rps_configs:
            rps_segment = generate_sine_rps_dist(**config)
            rps_dist.extend(rps_segment)
            
        scenarios = {
            'test-scenario-1': {
                'rps_dist': rps_dist,
                'input_token_len_dist': input_len_dist,
                'output_token_len_dist': output_len_dist,
            }
        }
        
        for scenario_name, scenario in scenarios.items():
            generated_workload = generate_synthetic_rps(
                args.prompt_file,
                scenario['rps_dist'],
                scenario['input_token_len_dist'],
                scenario['output_token_len_dist'],
            )
            if not os.path.exists(f"{args.output_dir}"):
                os.makedirs(f"{args.output_dir}")
            generated_workload_fn = f"{args.output_dir}/generated_workload-{workload_subname}.jsonl"
            with open(generated_workload_fn, 'w') as f:
                for req in generated_workload:
                    json.dump(req, f, cls=RequestEncoder)
                    f.write('\n')
                print(f"Saved generated workload to {generated_workload_fn}")
                
            workload_dict[scenario_name] = generated_workload
    elif args.trace_type == "internal":
            generated_workload = generate_from_internal_csv(file_path=args.traffic_file, 
                                                            prompt_file_path=args.prompt_file, 
                                                            summary_interval_ms=args.summary_interval_ms,
                                                            duration_ms=args.duration_ms, 
                                                            tokenizer=tokenizer,
                                                            interval_ms=args.interval_ms,
                                                            output_file=f"{args.output_dir}/{args.trace_type}",
                                                            input_trace=args.prompt_len_file, 
                                                            output_trace=args.completion_len_file,
                                                            to_jsonl=(args.output_format == "jsonl"),
                                                            qps_scale=args.qps_scale,
                                                            input_scale=args.input_scale,
                                                            output_scale=args.output_scale,
                                                            )

    elif args.trace_type == "azure":
        generated_workload = generate_from_azure_csv(file_path=args.traffic_file, 
                                                        prompt_file_path=args.prompt_file,
                                                        duration_ms=args.duration_ms, 
                                                        tokenizer=tokenizer,
                                                        interval_ms=args.interval_ms, 
                                                        output_file=f"{args.output_dir}/{args.trace_type}",
                                                        to_jsonl=(args.output_format == "jsonl"),
                                                        )

    workload_dict[args.trace_type] = generated_workload

    # if workload_dict:
    #     # Plot the workloads
    #     # plot_workload(workload_dict, interval_ms=args.interval_ms, output_file=f"plot/{args.trace_type}.pdf")
    #     if workload_subname:
    #         plot_rps_workload(workload_dict, output_file=f"{args.output_dir}/plot-{workload_subname}.pdf")
    #     else:
    #         plot_workload(workload_dict, interval_ms=args.interval_ms, output_file=f"plot/{args.trace_type}")
