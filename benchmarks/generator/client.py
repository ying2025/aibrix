import argparse
import logging
import time
import asyncio
import openai
import json
import os
import httpx
import matplotlib.pyplot as plt
from utils import (load_workload, wrap_prompt_as_chat_message)
logging.basicConfig(level=logging.INFO)

async def send_request_with_httpx(args, client, prompt, output_file, completion_map, batch_id=-1, request_id=-1):
   start_time = asyncio.get_event_loop().time()
   if not args.endpoint.startswith(('http://', 'https://')):
       args.endpoint = 'http://' + args.endpoint
   try:
        response = await client.post(
            f"{args.endpoint}/v1/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.api_key}", 
                "routing-strategy": args.routing_strategy
            },
            json={
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 2048
            },
        )
        try:
            data = response.json()
            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time
            result = {
                "status_code": response.status_code,
                "start_time": start_time,
                "end_time": end_time,
                "latency": latency,
                "throughput": data['usage']['completion_tokens'] / latency,
                "prompt_tokens": data['usage']['prompt_tokens'],
                "output_tokens": data['usage']['completion_tokens'],
                "total_tokens": data['usage']['total_tokens'],
                "input": prompt,
                "output": data['choices'][0]['text'],
            }
        except Exception as e:
            logging.error(f"Status: {response.status_code}, Raw response: {response.text}")
            logging.error(f"Error parsing response from {args.endpoint}: {str(e)}")
            result = {
                "status_code": response.status_code,
                "start_time": start_time,
                "end_time": None,
                "latency": None,
                "throughput": None,
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "input": prompt,
                "output": None,
            }
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()
        if completion_map[request_id] != 0:
            logging.error(f"Request {request_id} already completed")
            assert False
        completion_map[request_id] = 1
        logging.warning(f"Batch {batch_id}, Request {request_id}, completed in {latency:.2f} seconds with throughput {result['throughput']:.2f} tokens/s")
        logging.warning(f"Total sent requests so far: {len(completion_map)}, completed requests: {sum(completion_map.values())}, completion_ratio: {sum(completion_map.values()) / len(completion_map)*100:.2f}%")
        return result
   except Exception as e:
       logging.error(f"Error sending request to at {args.endpoint}: {str(e)}\nFull error: {repr(e)}")
       return None

# Asynchronous request handler
async def send_request(api_key, client, model, endpoint, prompt, output_file, completion_map, batch_id=-1, request_id=-1):
    start_time = asyncio.get_event_loop().time()
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 2048
    }

    logging.warning("-"*40)
    logging.warning(f"curl -X POST {endpoint}/v1/completions \\"
            f"-H 'Content-Type: application/json' \\"
            f"-H 'Authorization: Bearer {api_key}' \\"
            f"-H 'routing-strategy: least-request' \\"
            f"-d '{json.dumps(data)}'")
    logging.warning("-"*40)
    try:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=2048
        )
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time
        prompt_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        throughput = output_tokens / latency
        output_text = response.choices[0].message.content

        result = {
            "input": prompt,
            "output": output_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "start_time": start_time,
            "end_time": end_time,
            "latency": latency,
            "throughput": throughput
        }

        # Write result to JSONL file
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file
        logging.warning(f"Batch {batch_id}, Request {request_id}, completed in {latency:.2f} seconds with throughput {throughput:.2f} tokens/s, request {prompt} response {response}")
        if completion_map[request_id] != 0:
            logging.error(f"Request {request_id} already completed")
            assert False
        completion_map[request_id] = 1
        logging.info(f"Total sent requests so far: {len(completion_map)}, completed requests: {sum(completion_map.values())}, completion_ratio: {sum(completion_map.values()) / len(completion_map)*100:.2f}%")
        return result
    except Exception as e:
        logging.error(f"Error sending request to at {endpoint}: {str(e)}")
        return None

def collapse_workload(load_struct, minimum_time_unit=500, write=False):
    # Collapse requests
    collapsed_requests = {}
    for requests_dict in load_struct:
        original_ts = int(requests_dict["timestamp"])
        collapsed_ts = (original_ts // minimum_time_unit) * minimum_time_unit
        if collapsed_ts not in collapsed_requests:
            collapsed_requests[collapsed_ts] = []
        collapsed_requests[collapsed_ts].extend([{
            "Prompt Length": request.get("Prompt Length"),
            "Output Length": request.get("Output Length"),
            "prompt": request["prompt"]
        } for request in requests_dict["requests"]])
    # Write collapsed workload
    if write:
        collapsed_workload_path = workload_path.rsplit('.', 1)[0] + '_collapsed.jsonl'
        with open(collapsed_workload_path, 'w', encoding='utf-8') as f:
            for ts in sorted(collapsed_requests.keys()):
                entry = {
                    "timestamp": ts,
                    "requests": collapsed_requests[ts]
                }
                f.write(json.dumps(entry) + '\n')
        logging.info(f"Written collapsed workload to {collapsed_workload_path}")
    return collapsed_requests

async def new_benchmark_prescheduling2(args):
    # # openai client
    # client = openai.AsyncOpenAI(
    #     api_key=api_key,
    #     base_url=args.endpoint + "/v1",
    #     default_headers={"routing-strategy": "least-request"},
    # )

    # httpx client
    client = httpx.AsyncClient(timeout=300.0)

    load_struct = load_workload(args.workload_path)
    collapsed_wrk = collapse_workload(load_struct, minimum_time_unit=1)
    rps_dict = {}
    for ts, requests in collapsed_wrk.items():
        second = ts//1000
        if second not in rps_dict:
            rps_dict[second] = 0
        rps_dict[second] += len(requests)
    rps_list = sorted(rps_dict.items())
    with open(f"{args.output_dir}/intended_rps.csv", 'w', encoding='utf-8') as f:
        for ts, rps in rps_list:
            f.write(f"{ts},{rps}\n")
    with open(f"{args.output_dir}/intended_traffic.csv", 'w', encoding='utf-8') as f:
        for ts, requests in collapsed_wrk.items():
            f.write(f"{ts/1000},{len(requests)}\n")
    logging.info(f"expected load: {rps_list}")
    base_time = time.time()
    num_requests = 0
    all_tasks = []
    num_requests_sent = 0
    request_id = 0
    completion_map = {}
    with open(args.output_file_path, 'w', encoding='utf-8') as f_out:
        sorted_timestamps = sorted(collapsed_wrk.keys())
        num_requests = sum(len(collapsed_wrk[ts]) for ts in sorted_timestamps)
        logging.info(f"Starting benchmark with {len(sorted_timestamps)} batches, total {num_requests} requests")
        try:
            for batch_num, ts in enumerate(sorted_timestamps):
                formatted_prompts = [request["prompt"] for request in collapsed_wrk[ts]]
                target_time = base_time + (ts / 1000.0)
                current_time = time.time()
                sleep_duration = target_time - current_time
                if sleep_duration > 0:
                    logging.info(f"Waiting {sleep_duration:.2f}s before sending batch {batch_num} with {len(formatted_prompts)} requests")
                    await asyncio.sleep(sleep_duration)
                    logging.info(f"Sending batch {batch_num} with {len(formatted_prompts)} requests at {(time.time()-base_time):.2f}s (on schedule)")
                else:
                    logging.info(f"Sending batch {batch_num} with {len(formatted_prompts)} requests at {(time.time()-base_time):.2f}s (behind by {-sleep_duration:.2f}s)")
                # Create tasks for each request in the batch but don't await them
                batch_tasks = []
                for prompt in formatted_prompts:
                    completion_map[request_id] = 0
                    # batch_tasks.append(asyncio.create_task(send_request(api_key, client, model, endpoint, wrap_prompt_as_chat_message(prompt), f_out, completion_map, batch_num, request_id)))
                    batch_tasks.append(asyncio.create_task(send_request_with_httpx(args, client, wrap_prompt_as_chat_message(prompt), f_out, completion_map, batch_num, request_id)))
                    request_id += 1
                all_tasks.extend(batch_tasks)
                num_requests_sent += len(formatted_prompts)
            # Wait for all requests to complete after all batches have been sent
            await asyncio.gather(*all_tasks)
            total_time = time.time() - base_time
            actual_qps = num_requests / total_time
            logging.info(f"Completed {num_requests} requests in {total_time:.2f}s (actual QPS: {actual_qps:.2f})")
            logging.info(f"num of requests sent: {num_requests_sent}")
            logging.info(f"num of requests completed: {sum(completion_map.values())}")
            logging.info(f"Completion ratio: {sum(completion_map.values()) / num_requests_sent * 100:.2f}%")
            logging.info(f"Output file: {args.output_file_path}")
        except Exception as e:
            logging.error(f"Benchmark failed: {str(e)}")
            raise

async def new_benchmark_prescheduling(endpoint, model, api_key, workload_path, output_file_path, minimum_time_unit=500):
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint + "/v1",
    )
    logging.info(f"Writing output to {output_file_path}")
    
    # Load workload
    load_struct = load_workload(workload_path)
    
    # Collapse requests
    collapsed_requests = {}
    for requests_dict in load_struct:
        original_ts = int(requests_dict["timestamp"])
        collapsed_ts = (original_ts // minimum_time_unit) * minimum_time_unit
        if collapsed_ts not in collapsed_requests:
            collapsed_requests[collapsed_ts] = []
        collapsed_requests[collapsed_ts].extend([{
            "Prompt Length": request.get("Prompt Length"),
            "Output Length": request.get("Output Length"),
            "prompt": request["prompt"]
        } for request in requests_dict["requests"]])

    # Write collapsed workload
    collapsed_workload_path = workload_path.rsplit('.', 1)[0] + '_collapsed.jsonl'
    with open(collapsed_workload_path, 'w', encoding='utf-8') as f:
        for ts in sorted(collapsed_requests.keys()):
            entry = {
                "timestamp": ts,
                "requests": collapsed_requests[ts]
            }
            f.write(json.dumps(entry) + '\n')
    logging.info(f"Written collapsed workload to {collapsed_workload_path}")

    # Pre-create all tasks
    base_time = time.time()
    all_tasks = []
    num_requests = 0
    
    async def execute_batch_requests(prompts, output_file):
        """Execute all requests in a batch concurrently"""
        return await asyncio.gather(*[
            send_request(client, model, endpoint, wrap_prompt_as_chat_message(prompt), output_file)
            for prompt in prompts
        ])

    # Schedule all batches
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        for batch_num, ts in enumerate(sorted(collapsed_requests.keys())):
            formatted_prompts = [request["prompt"] for request in collapsed_requests[ts]]
            target_time = base_time + (ts / 1000.0)
            num_requests += len(formatted_prompts)
            
            async def execute_timed_batch(prompts, scheduled_time, batch_number):
                current_time = time.time()
                sleep_duration = scheduled_time - current_time
                if sleep_duration > 0:
                    logging.warning(f"Waiting {sleep_duration:.2f}s before sending batch {batch_number} with {len(prompts)} requests")
                    await asyncio.sleep(sleep_duration)
                    logging.warning(f"Sending batch {batch_number} with {len(prompts)} requests at {(time.time()-base_time):.2f}s (on schedule)")
                else:
                    logging.warning(f"Sending batch {batch_number} with {len(prompts)} requests at {(time.time()-base_time):.2f}s (behind by {-sleep_duration:.2f}s)")
                
                return await execute_batch_requests(prompts, output_file)
            
            batch_task = asyncio.create_task(
                execute_timed_batch(formatted_prompts, target_time, batch_num)
            )
            all_tasks.append(batch_task)

        logging.warning(f"Created {len(all_tasks)} batches with total {num_requests} requests")

        # Execute all tasks
        try:
            await asyncio.gather(*all_tasks)
            total_time = time.time() - base_time
            actual_qps = num_requests / total_time
            logging.warning(f"Completed {num_requests} requests in {total_time:.2f}s (actual QPS: {actual_qps:.2f})")
        except Exception as e:
            logging.error(f"Benchmark failed: {str(e)}")
            raise


## NOTE (gangmuk): This modified original benchmark function to minimize the time drifting but it is essentially impossible to do correctly it in this way. This is highly inefficient as well as it is utilizing the single thread only.
## The current update added more correct way to track time including time drifting due to request dispatching overhead. It is still best effort not the best by design as long as it is not hihgly effficient. (multi-thread, sharing tcp connection, etc.)
async def new_benchmark(endpoint, model, api_key, workload_path, output_file_path, minimum_time_unit=500):
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint + "/v1",
    )
    logging.info(f"Writing output to {output_file_path}")
    load_struct = load_workload(workload_path)
    collapsed_requests = {}
    for requests_dict in load_struct:
        original_ts = int(requests_dict["timestamp"])
        collapsed_ts = (original_ts // minimum_time_unit) * minimum_time_unit
        if collapsed_ts not in collapsed_requests:
            collapsed_requests[collapsed_ts] = []
        collapsed_requests[collapsed_ts].extend([{
            "Prompt Length": request.get("Prompt Length"),
            "Output Length": request.get("Output Length"),
            "prompt": request["prompt"]
        } for request in requests_dict["requests"]])
    collapsed_workload_path = workload_path.rsplit('.', 1)[0] + '_collapsed.jsonl'
    with open(collapsed_workload_path, 'w', encoding='utf-8') as f:
        for ts in sorted(collapsed_requests.keys()):
            entry = {
                "timestamp": ts,
                "requests": collapsed_requests[ts]
            }
            f.write(json.dumps(entry) + '\n')
    logging.info(f"Written collapsed workload to {collapsed_workload_path}")
    prepared_batches = []
    for ts in sorted(collapsed_requests.keys()):
        formatted_prompts = [wrap_prompt_as_chat_message(request["prompt"]) for request in collapsed_requests[ts]]
        prepared_batches.append({
            "timestamp": ts,
            "prompts": formatted_prompts
        })
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        base_time = time.time()
        num_requests = 0
        all_tasks = []  # Keep track of all tasks
        for batch in prepared_batches:
            ts = batch["timestamp"]
            formatted_prompts = batch["prompts"]
            target_time = base_time + (ts / 1000.0)
            current_time = time.time()
            sleep_duration = target_time - current_time
            if sleep_duration > 0:
                logging.warning(f"Launching {len(formatted_prompts)} tasks after {sleep_duration:.2f}s at {(time.time()-base_time):.2f}s")
                await asyncio.sleep(sleep_duration)
            else:
                behind_schedule = -sleep_duration
                logging.warning(f"Behind schedule by {behind_schedule:.2f}s, launching {len(formatted_prompts)} tasks at {(time.time()-base_time):.2f}s")
            batch_tasks = [
                asyncio.create_task(
                    send_request(client, model, endpoint, prompt, output_file)
                )
                for prompt in formatted_prompts
            ]
            all_tasks.extend(batch_tasks)
            num_requests += len(formatted_prompts)
        await asyncio.gather(*all_tasks)
        total_time = time.time() - base_time
        actual_qps = num_requests / total_time
        logging.warning(f"Completed {num_requests} requests in {total_time:.2f}s (actual QPS: {actual_qps:.2f})")

## Old
async def benchmark(endpoint, model, api_key, workload_path, output_file_path):
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint + "/v1/completions",
    )
    logging.info(f"Writing output to {output_file_path}")
    load_struct = load_workload(workload_path)
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        base_time = time.time()
        num_requests = 0
        batch_tasks = []
        idx = 0
        for requests_dict in load_struct:
            idx += 1
            ts = int(requests_dict["timestamp"])
            requests = requests_dict["requests"]
            cur_time = time.time()
            target_time = base_time + ts / 1000.0
            logging.warning(f"Prepare to launch {len(requests)} tasks after {target_time - cur_time}")
            if target_time > cur_time:
                await asyncio.sleep(target_time - cur_time)
                logging.info(f"batch idx: {idx}, sleeping for {target_time - cur_time}s")
            formatted_prompts = [wrap_prompt_as_chat_message(request["prompt"]) for request in requests]
            logging.info(f"batch idx: {idx}, num_requests: {len(formatted_prompts)}, time: {time.time()-base_time:.2f}s")
            for formatted_prompt in formatted_prompts:
                task = asyncio.create_task(
                    send_request(client, model, endpoint, formatted_prompt, output_file)
                )
                batch_tasks.append(task)
            num_requests += len(requests)
        await asyncio.gather(*batch_tasks)
        logging.warning(f"All {num_requests} requests completed for deployment.")


def main(args):
    logging.info(f"Starting benchmark on endpoint {args.endpoint}")
    start_time = time.time()
    # asyncio.run(benchmark(args.endpoint, args.model, args.api_key, args.workload_path, args.output_file_path))
    # asyncio.run(new_benchmark(args.endpoint, args.model, args.api_key, args.workload_path, args.output_file_path))
    # asyncio.run(new_benchmark_prescheduling(args.endpoint, args.model, args.api_key, args.workload_path, args.output_file_path))
    asyncio.run(new_benchmark_prescheduling2(args))
    end_time = time.time()
    logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload Generator')
    parser.add_argument("--workload-path", type=str, default=None, help="File path to the workload file.")
    parser.add_argument('--endpoint', type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Name of the model.")
    parser.add_argument("--api-key", type=str, required=True, help="API key to the service. ")
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--output-file-path', type=str, default="output.jsonl")
    parser.add_argument('--routing-strategy', type=str, default="least-request")

    args = parser.parse_args()
    main(args)
