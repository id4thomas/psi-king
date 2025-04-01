#!/bin/bash
source .env

QDRANT_VERSION="v1.13.2"

docker run -d \
    -p ${QDRANT_PORT}:6333 \
    -v $(pwd)/database/storage:/qdrant/storage \
    -v $(pwd)/database/snapshots:/qdrant/snapshots \
    qdrant/qdrant:${QDRANT_VERSION}