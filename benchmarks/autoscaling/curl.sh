#!/bin/bash

curl -X POST -v http://localhost:8888/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "routing-strategy: least-request" \
    -d '{
        "model": "deepseek-llm-7b-chat",
        "prompt": "Where is Beijing",
        "temperature": 0.5,
        "max_tokens": 4000}'


# 33204