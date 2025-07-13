#!/bin/bash
VLLM_VERSION="v0.9.1"

MODEL="jina-embeddings-v4-vllm-retrieval"

docker container rm -f vllm_serving
docker run --runtime nvidia --gpus 'device=0' \
        --name vllm_serving \
        -v ./models/${MODEL}:/vllm-workspace/model \
        -v ./cache:/root/.cache/huggingface \
        -p 8010:8000 \
        --ipc=host \
        vllm/vllm-openai:${VLLM_VERSION} \
        --model "/vllm-workspace/model" \
        --task "embed" \
        --served-model-name ${MODEL} \
        --gpu-memory-utilization=0.95 \
        --override-pooler-config '{"pooling_type":"ALL", "normalize": false}'