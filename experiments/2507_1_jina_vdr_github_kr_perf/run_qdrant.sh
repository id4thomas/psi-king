#!/bin/bash

QDRANT_VERSION="1.14.1"

docker run \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$(pwd)/storage:/qdrant/storage:z" \
    qdrant/qdrant:v${QDRANT_VERSION}