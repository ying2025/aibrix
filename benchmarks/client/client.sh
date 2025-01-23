#!/bin/bash

 rm output.jsonl
 python client.py --workload-path "output/constant.jsonl" --endpoint "http://localhost:8000" --model deepseek-llm-7b-chat --api-key sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi --output-file-path output.jsonl