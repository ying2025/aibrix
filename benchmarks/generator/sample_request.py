import logging
import json
import sys
import random 
import time

import pandas as pd
import numpy as np

from typing import Tuple, Optional, List, Dict, Any
from transformers import PreTrainedTokenizerBase

RANDOM_SEED = 1111  # Define at top level
np.random.seed(RANDOM_SEED)

def load_requests(
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
) -> pd.DataFrame:
    if "ShareGPT" in dataset_path:
        return load_sharegpt_requests(dataset_path, tokenizer)
    else:
        return load_generated_dataset(dataset_path, tokenizer)

class Request:
    def __init__(self, timestamp, request_id, input_token_len, output_token_len):
        self.timestamp = int(timestamp)
        self.request_id = request_id
        self.input_token_len = int(input_token_len)
        self.output_token_len = int(output_token_len)
        self.prompt = None
        
    def __str__(self):
        ret_str = ""
        ret_str += f"Request ID: {self.request_id}\n"
        ret_str += f"Timestamp: {self.timestamp}\n"
        # ret_str += f"Input Token Length: {self.input_token_len}\n"
        ret_str += f"Prompt Length: {self.input_token_len}\n"
        ret_str += f"Output Length: {self.output_token_len}\n"
        ret_str += f"Prompt: {self.prompt}\n"
        return ret_str
        
    def to_dict(self):
        return {
            "Prompt Length": self.input_token_len,
            "Output Length": self.output_token_len,
            "prompt": self.prompt
        }


class RequestEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Request):
            return {
                "timestamp": int(obj.timestamp),  # Convert to native int
                "requests": [obj.to_dict()]
            }
        if isinstance(obj, np.integer):  # Handle numpy integers
            return int(obj)
        if isinstance(obj, np.floating):  # Handle numpy floats
            return float(obj)
        return super().default(obj)

    
def load_sharegpt_requests(
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
) -> pd.DataFrame:
    # Load the dataset into a DataFrame
    logging.warn(f"...Start dataframe transformation")
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset if len(data["conversations"]) >= 2
    ]
    df = pd.DataFrame(dataset, columns=["prompt", "completion"])
    # Tokenize and calculate lengths
    ###############
    ## TODO: It takes 120 seconds. needs to be optimized
    print("prompt len calc start")
    ts = time.time()
    df["prompt_len"] = df["prompt"].apply(lambda x: len(tokenizer(x).input_ids))
    print(f"prompt len calc took {time.time() - ts}")
    ###############
    
    ###############
    ## TODO: It takes 75 seconds. needs to be optimized
    print("completion len calc start")
    ts = time.time()
    df["completion_len"] = df["completion"].apply(lambda x: len(tokenizer(x).input_ids))
    print(f"completion len calc took {time.time() - ts}")
    ###############
    
    logging.warn(f"...Complete dataframe transformation")
    return df

def load_generated_dataset(
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
) -> pd.DataFrame:
    # Load the dataset into a DataFrame
    with open(dataset_path, encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    # Create a DataFrame with the desired columns
    logging.warn(f"...Start dataframe transformation")
    df = pd.DataFrame({
        'prompt': [entry['input'][0]['content'] for entry in dataset],
        'completion': [entry['output'] for entry in dataset],
        'prompt_len': [entry['prompt_tokens'] for entry in dataset],
        'completion_len': [entry['output_tokens'] for entry in dataset]
    })
    logging.warn(f"...Complete dataframe transformation")
    return df

def sample_sharegpt_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]

    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break
        prompt = dataset[i][0]
        if tokenizer is not None:
            prompt_token_ids = tokenizer(prompt).input_ids
            completion = dataset[i][1]
            completion_token_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_token_ids)
            output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
            if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                continue
            filtered_dataset.append({"prompt": prompt,
                                     "prompt_length": prompt_len,
                                     "output_length": output_len})
        else:
            filtered_dataset.append({"prompt": prompt,
                                     "prompt_length": -1,
                                     "output_length": -1})

    return filtered_dataset


def sample_requests_len_range(
        df: pd.DataFrame,
        num_requests: int,
        input_lens: List[int],
        output_lens: List[int],
        initial_err_perc: Optional[float] = 0.1,
        err_step: float = 0.05
) -> List[Tuple[str, int, int, None]]:
    filtered_results = []
    # Relaxation mechanism
    for i in range(num_requests):
        input_len = input_lens[i]
        output_len = output_lens[i]
        err_perc = initial_err_perc

        while err_perc < 1:
            input_range = range(0, sys.maxsize)
            output_range = range(0, sys.maxsize)
            if input_len is not None:
                input_range = (int(input_len * (1 - err_perc)), int(input_len * (1 + err_perc)))
            else:
                input_range = (0, sys.maxsize)
            if output_len is not None:
                output_range = (int(output_len * (1 - err_perc)), int(output_len * (1 + err_perc))) 
            else:
                output_range = (0, sys.maxsize)
            filtered = df[
                (df["prompt_len"] >= input_range[0]) &
                (df["prompt_len"] <= input_range[1]) &
                (df["completion_len"] >= output_range[0]) &
                (df["completion_len"] <= output_range[1])
                ]

            if not filtered.empty:
                # Select the first match or random sample
                total_rows = len(filtered)
                sample = filtered.iloc[random.randint(0, total_rows - 1)] 
                filtered_results.append({"prompt": sample["prompt"],
                                         "prompt_length": sample["prompt_len"],
                                         "output_length": sample["completion_len"]})
                break  # Stop relaxing for this request once a match is found

            # Reduce err_perc for next iteration
            logging.debug(f"Relax err_perc {err_perc} by {err_step} new err_perc {err_perc + err_step} input_range {input_range} output_range {output_range}")
            err_perc += err_step

        if err_perc >= 1:
            raise Exception(f"No match found for request {i + 1} even after relaxing err_perc to 0")

    return filtered_results


def sample_requests_all(
        df: pd.DataFrame,
        start_idx: int,
        qps: int
) -> List[Tuple[str, int, int, None]]:
    results = []

    # Relaxation mechanism
    end_idx = min(start_idx + qps, len(df))
    for i in  range(start_idx, end_idx):
        print(f"start_idx {start_idx} end_idx {end_idx} i {i} len {len(df)} ")
        row = df.iloc[i]
        results.append({"prompt": row["prompt"],
                        "prompt_length": row["prompt_len"],
                        "output_length": row["completion_len"]})

    return results

def sample_sharegpt_requests_len_range(df:pd.DataFrame,request,initial_err_perc:float=0.5,err_step:float=0.05):
    target_input_len=request.input_token_len
    target_output_len=request.output_token_len
    total_lens=df['prompt_len']+df['completion_len']
    length_ratios=df['prompt_len']/(df['completion_len'].replace(0,1))
    err_perc=initial_err_perc
    while err_perc<1:
        input_range=[int(target_input_len*(1-err_perc)),int(target_input_len*(1+err_perc))]
        output_range=[int(target_output_len*(1-err_perc)),int(target_output_len*(1+err_perc))]
        mask=(df['prompt_len']>=input_range[0])&(df['prompt_len']<=input_range[1])&(df['completion_len']>=output_range[0])&(df['completion_len']<=output_range[1])
        filtered=df[mask]
        if not filtered.empty:
            logging.debug(f"Found {len(filtered)} matches with err_perc={err_perc}")
            sample=filtered.sample(n=1).iloc[0]
            request.prompt=sample["prompt"]
            request.input_token_len=sample["prompt_len"]
            request.output_token_len=sample["completion_len"]
            return request
        logging.debug(f"No matches with err_perc={err_perc}, trying bigger range")
        err_perc+=err_step
    
    # logging.warning(f"No matches found with direct range matching, trying closest match")
    target_total=target_input_len+target_output_len
    target_ratio=target_input_len/target_output_len if target_output_len!=0 else float('inf')
    
    try:
        scores=np.abs(total_lens-target_total)/target_total*0.6+np.abs(length_ratios-target_ratio)/target_ratio*0.4
        num_candidates=max(1,len(df)//10)
        best_indices=np.argpartition(scores,num_candidates)[:num_candidates]
        best_sample=df.iloc[np.random.choice(best_indices)]
        
        # logging.info(f"Found closest match: input_len={best_sample['prompt_len']}, output_len={best_sample['completion_len']}")
        request.prompt=best_sample["prompt"]
        request.input_token_len=best_sample["prompt_len"]
        request.output_token_len=best_sample["completion_len"]
        if request.prompt is None:
            raise ValueError("Selected sample has null prompt")
        return request
        
    except Exception as e:
        error_msg=f"Failed to find match for request {request.request_id}. Error: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Request details: {request}")
        raise Exception(error_msg) from e

# def sample_sharegpt_requests_len_range(
#         df: pd.DataFrame,
#         request,
#         initial_err_perc: Optional[float] = 0.5,
#         err_step: float = 0.05
# ):
#     err_perc = initial_err_perc
#     target_input_len = request.input_token_len
#     target_output_len = request.output_token_len
    
#     # First try with regular error percentage relaxation
#     while err_perc >= 0:
#         input_range = [int(target_input_len * (1 - err_perc)), int(target_input_len * (1 + err_perc))]
#         output_range = [int(target_output_len * (1 - err_perc)), int(target_output_len * (1 + err_perc))]
        
#         filtered = df[
#             (df["prompt_len"] >= input_range[0]) &
#             (df["prompt_len"] <= input_range[1]) &
#             (df["completion_len"] >= output_range[0]) &
#             (df["completion_len"] <= output_range[1])
#         ]
        
#         if not filtered.empty:
#             sample = filtered.sample(n=1).iloc[0]
#             request.prompt = sample["prompt"]
#             request.input_token_len = sample["prompt_len"]
#             request.output_token_len = sample["completion_len"]
#             return request
        
#         logging.debug(f"Relax err_perc {err_perc} by {err_step}")
#         err_perc -= err_step

#     # If no match found, try a different approach: find closest match by total token length
#     logging.debug("No match found with error percentage relaxation, trying closest match approach")
    
#     # Calculate total token length for request and dataset
#     target_total = target_input_len + target_output_len
#     df['total_len'] = df['prompt_len'] + df['completion_len']
    
#     # Calculate relative ratio between input and output lengths
#     target_ratio = target_input_len / target_output_len if target_output_len != 0 else float('inf')
#     df['length_ratio'] = df['prompt_len'] / df['completion_len'].replace(0, 1)
    
#     # Score each row based on how close it is to target lengths
#     # Lower score is better
#     df['score'] = (
#         np.abs(df['total_len'] - target_total) / target_total * 0.6 +  # 60% weight on total length
#         np.abs(df['length_ratio'] - target_ratio) / target_ratio * 0.4  # 40% weight on ratio
#     )
    
#     # Get the best matching rows (top 10%)
#     num_candidates = max(1, len(df) // 10)
#     best_matches = df.nsmallest(num_candidates, 'score')
    
#     if not best_matches.empty:
#         sample = best_matches.sample(n=1).iloc[0]
#         request.prompt = sample["prompt"]
#         request.input_token_len = sample["prompt_len"]
#         request.output_token_len = sample["completion_len"]
        
#         # Clean up temporary columns
#         df.drop(['total_len', 'length_ratio', 'score'], axis=1, inplace=True)
#         return request
    
#     raise Exception(f"No suitable match found for request {request.request_id} even after trying all relaxation methods.\nOriginal request: {request}")
