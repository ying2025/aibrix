import argparse
import logging
import time
import asyncio
import openai
import json
import io
import traceback
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


from typing import List
from utils import (load_workload, wrap_prompt_as_chat_message)

logging.basicConfig(level=logging.INFO)
task_queue = Queue()
thread_pool_size = 8

def worker(client, model, send_request_func, output_file):
    """Worker function to run an asyncio event loop in a separate thread."""
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    while True:
        task = task_queue.get()
        if task is None:  # Stop signal
            break
        loop.run_until_complete(send_request_func(client, model, *task))


def start_worker_threads(num_threads, client, model, send_request_func, output_file):
    """Start multiple threads, each running an event loop for handling tasks."""
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker, args=(client, model, send_request_func, output_file), daemon=True)
        thread.start()
        threads.append(thread)
    return threads



async def send_request_streaming(client: openai.AsyncOpenAI,
                             model: str,
                             prompt: str,
                             output_file: str,
                             request_id: int):
    start_time = asyncio.get_event_loop().time()
    first_response_time = None
    target_pod = ""
    try:
        logging.info(f"Request {request_id}: Starting streaming request")
        response_stream = await client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=2048,
            stream=True,
            stream_options={"include_usage": True},
        )
        if hasattr(response_stream, 'response') and hasattr(response_stream.response, 'headers'):
            target_pod = response_stream.response.headers.get('target-pod')

        text_chunks = []
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0

        try:
            async for chunk in response_stream:
                if chunk.choices:
                    if chunk.choices[0].delta.content is not None:
                        if not first_response_time:
                            first_response_time = asyncio.get_event_loop().time()
                        output_text = chunk.choices[0].delta.content
                        text_chunks.append(output_text)
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    # For OpenAI, we expect to get complete usage stats, not partial ones to accumulate
                    # So we can safely overwrite previous values if they exist
                    if chunk.usage.prompt_tokens is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
                    if chunk.usage.completion_tokens is not None:
                        output_tokens = chunk.usage.completion_tokens
                    if chunk.usage.total_tokens is not None:
                        total_tokens = chunk.usage.total_tokens
        except Exception as stream_error:
            # Handle errors during streaming
            logging.error(f"Request {request_id}: Stream interrupted: {type(stream_error).__name__}: {str(stream_error)}")

        response_text = "".join(text_chunks)
        response_time = asyncio.get_event_loop().time()
        latency = response_time - start_time
        throughput = output_tokens / latency if output_tokens > 0 else 0
        ttft = first_response_time - start_time if first_response_time else None
        tpot = (response_time - first_response_time) / output_tokens if first_response_time and output_tokens > 0 else None

        result = {
            "request_id": request_id,
            "status": "success",
            "input": prompt,
            "output": response_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
            "throughput": throughput,
            "start_time": start_time,
            "end_time": response_time,
            "ttft": ttft,
            "tpot": tpot,
            "target_pod": target_pod,
        }

        # Write result to JSONL file
        logging.info(f"Request {request_id}: Completed successfully. Tokens: {total_tokens}, Latency: {latency:.2f}s")
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        # Determine error type based on exception class
        error_type = type(e).__name__
        error_result = {
            "request_id": request_id,
            "status": "error",
            "error_type": error_type,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "input": prompt,
            "latency": error_time - start_time,
            "start_time": start_time,
            "end_time": error_time,
            "target_pod": target_pod,
        }
        logging.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
        output_file.write(json.dumps(error_result) + "\n")
        output_file.flush()
        return error_result

async def benchmark_streaming(api_key: str,
                              endpoint: str,
                              max_retries: int,
                              timeout: float,
                              routing_strategy: str,
                              load_struct: List,
                              output_file: io.TextIOWrapper,
                              model: str,
                              ):
    request_id = 0
    batch_tasks = []
    base_time = time.time()
    num_requests = 0
    threads = []
    for _ in range(0, thread_pool_size):
        client = create_client(api_key, endpoint, max_retries, timeout, routing_strategy)
        threads.extend(start_worker_threads(1, client, model, send_request_streaming, output_file))
    for requests_dict in load_struct:
        ts = int(requests_dict["timestamp"])
        requests = requests_dict["requests"]
        cur_time = time.time()
        target_time = base_time + ts / 1000.0
        logging.warning(f"Prepare to launch {len(requests)} streaming tasks after {target_time - cur_time}")
        if target_time > cur_time:
            await asyncio.sleep(target_time - cur_time)
        formatted_prompts = [wrap_prompt_as_chat_message(request["prompt"]) for request in requests]
        for i in range(len(requests)):
            task_queue.put((formatted_prompts[i], output_file, request_id))
            request_id += 1
        num_requests += len(requests)
    task_queue.join()
    # Stop all worker threads
    for _ in threads:
        task_queue.put(None)

    for thread in threads:
        thread.join()
    logging.warning(f"All {num_requests} requests completed for deployment.")

# Asynchronous request handler
async def send_request_batch(client: openai.AsyncOpenAI,
                             model: str,
                             prompt: str,
                             output_file: str,
                             request_id: int):
    start_time = asyncio.get_event_loop().time()
    target_pod = ""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=2048
        )
        if hasattr(response, 'response') and hasattr(response.response, 'headers'):
            target_pod = response.response.headers.get('target-pod')

        response_time = asyncio.get_event_loop().time()
        latency = response_time - start_time
        prompt_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        throughput = output_tokens / latency
        output_text = response.choices[0].message.content

        result = {
            "request_id": request_id,
            "status": "success",
            "input": prompt,
            "output": output_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
            "throughput": throughput,
            "start_time": start_time,
            "end_time": response_time,
            "ttft": "Unknown",
            "tpot": "Unknown",
            "target_pod": target_pod,
        }
        logging.info(result)
        # Write result to JSONL file
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        error_type = type(e).__name__
        error_result = {
            "request_id": request_id,
            "status": "error",
            "error_type": error_type,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "input": prompt,
            "latency": error_time - start_time,
            "start_time": start_time,
            "end_time": error_time,
            "target_pod": target_pod
        }
        logging.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
        output_file.write(json.dumps(error_result) + "\n")
        output_file.flush()
        return error_result


async def benchmark_batch(api_key: str,
                          endpoint: str,
                          max_retries: int,
                          timeout: float,
                          routing_strategy: str,
                          load_struct: List,
                          output_file: io.TextIOWrapper,
                          model: str,
                          ):
    request_id = 0
    batch_tasks = []
    base_time = time.time()
    num_requests = 0
    threads = []
    for _ in range(0, thread_pool_size):
        client = create_client(api_key, endpoint, max_retries, timeout, routing_strategy)
        threads.extend(start_worker_threads(1, client, model, send_request_batch, output_file))
    for requests_dict in load_struct:
        ts = int(requests_dict["timestamp"])
        requests = requests_dict["requests"]
        cur_time = time.time()
        target_time = base_time + ts / 1000.0
        logging.warning(f"Prepare to launch {len(requests)} batched tasks after {target_time - cur_time}")
        if target_time > cur_time:
            await asyncio.sleep((target_time - cur_time)/4)
        formatted_prompts = [wrap_prompt_as_chat_message(request["prompt"]) for request in requests]
        for i in range(len(requests)):
            task_queue.put((formatted_prompts[i], output_file, request_id))
        num_requests += len(requests)
    await asyncio.gather(*batch_tasks)
    logging.warning(f"All {num_requests} requests completed for deployment.")

def create_client(api_key: str,
                  endpoint: str,
                  max_retries: int,
                  timeout: float,
                  routing_strategy: str,
                  ):
    if args.api_key is None:
        client = openai.AsyncOpenAI(
            base_url=args.endpoint + "/v1",
            max_retries=max_retries,
            timeout=timeout,
        )
    else:
        client = openai.AsyncOpenAI(
            api_key=args.api_key,
            base_url=args.endpoint + "/v1",
            max_retries=max_retries,
            timeout=timeout,
        )
    if routing_strategy is not None:
        client = client.with_options(
            default_headers={"routing-strategy": routing_strategy}
        )
    return client

def main(args):
    logging.info(f"Starting benchmark on endpoint {args.endpoint}")
    with open(args.output_file_path, 'w', encoding='utf-8') as output_file:
        load_struct = load_workload(args.workload_path)
        if not args.streaming:
            logging.info("Using batch client")
            start_time = time.time()
            asyncio.run(benchmark_batch(
                api_key = args.api_key,
                endpoint = args.endpoint,
                max_retries = 0,
                timeout = 60.0,
                routing_strategy = args.routing_strategy,
                load_struct=load_struct,
                output_file=output_file,
                model=args.model,
            ))
            end_time = time.time()
            logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")
        else:
            logging.info("Using streaming client")
            start_time = time.time()
            asyncio.run(benchmark_streaming(
                api_key = args.api_key,
                endpoint = args.endpoint,
                max_retries = 0,
                timeout = 60.0,
                routing_strategy = args.routing_strategy,
                load_struct=load_struct,
                output_file=output_file,
                model=args.model,
            ))
            end_time = time.time()
            logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload Generator')
    parser.add_argument("--workload-path", type=str, default=None, help="File path to the workload file.")
    parser.add_argument("--model", type=str, default=None, help="Default target model (if workload does not contains target model).")
    parser.add_argument('--endpoint', type=str, required=True)
    parser.add_argument("--api-key", type=str, default=None, help="API key to the service. ")
    parser.add_argument('--output-file-path', type=str, default="output.jsonl")
    parser.add_argument("--streaming", action="store_true", help="Use streaming client.")
    parser.add_argument("--routing-strategy", type=str, required=False, default="random", help="Routing strategy to use.")

    args = parser.parse_args()
    main(args)

