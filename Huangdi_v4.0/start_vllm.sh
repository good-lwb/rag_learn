#!/bin/bash
MODEL_PATH="../Qwen"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port 6789 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --enable-prefix-caching