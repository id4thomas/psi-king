#!/bin/bash
# export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH" # for cannot fine lcuda error
# export VLLM_LOGGING_LEVEL=DEBUG

MODEL_DIR=""
MODEL="Qwen2.5-VL-7B-Instruct"

vllm serve "${MODEL_DIR}/${MODEL}" \
    --port 8010 \
    --served-model-name ${MODEL} \
    --gpu-memory-utilization=0.8
    
# --mm-processor-kwargs {"num_crops": 4}