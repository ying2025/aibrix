import argparse
import logging
import time
import asyncio
import random
import httpx
import openai
import json

from utils import (load_workload, wrap_prompt_as_chat_message)


# Asynchronous request handler
async def send_request(client, model, endpoint, prompt, output_file):
    start_time = asyncio.get_event_loop().time()
    # lora_num = random.randint(1, 10)
    # model = f"lora-{lora_num}"
    api_key = "sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt[0]["content"],
        "temperature": 0,
        "top_p": 1
    }

    async with httpx.AsyncClient() as aclient:
        try:
            response = await aclient.post(endpoint+"/v1/completions", headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP error codes
            response_json = response.json()
        except Exception as e:
            logging.error(f"Error sending request to at {endpoint}: {str(e)}")
            return None
        latency = asyncio.get_event_loop().time() - start_time

        # Extracting usage information
        usage = response_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        throughput = output_tokens / latency if latency > 0 else 0

        # Extracting the generated text
        choices = response_json.get("choices", [])
        if choices:
            output_text = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
        else:
            output_text = ""

        result = {
            "output": output_text.strip(),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
            "throughput": throughput
        }
        # Write result to JSONL file
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file

        logging.warning(
            f"Request completed in {latency:.2f} seconds with throughput {throughput:.2f} tokens/s, response {result}")
        return result
        


async def benchmark(endpoint, model, api_key, workload_path, output_file_path):
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint + "/v1",
    )
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        load_struct = load_workload(workload_path)
        batch_tasks = []
        base_time = time.time()
        num_requests = 0
        for requests_dict in load_struct:
            ts = int(requests_dict["timestamp"])
            requests = requests_dict["requests"]
            cur_time = time.time()
            target_time = base_time + ts / 1000.0
            logging.warning(f"Prepare to launch {len(requests)} tasks after {target_time - cur_time}")
            if target_time > cur_time:
                await asyncio.sleep(target_time - cur_time)
            formatted_prompts = [wrap_prompt_as_chat_message(request["prompt"]) for request in requests]
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
    asyncio.run(benchmark(args.endpoint, args.model, args.api_key, args.workload_path, args.output_file_path))
    end_time = time.time()
    logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload Generator')
    parser.add_argument("--workload-path", type=str, default=None, help="File path to the workload file.")
    parser.add_argument('--endpoint', type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Name of the model.")
    parser.add_argument("--api-key", type=str, required=True, help="API key to the service. ")
    parser.add_argument('--output-file-path', type=str, default="output.jsonl")

    args = parser.parse_args()
    main(args)
