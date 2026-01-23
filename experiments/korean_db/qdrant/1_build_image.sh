#/bin/bash
# building qdrant image with cjk (chinese, japanese, korean) language support

# https://qdrant.tech/documentation/concepts/indexing/#full-text-index
# --features multiling-chinese,multiling-japanese,multiling-korean

cd qdrant

QDRANT_VERSION="v1.13.2"
docker build . \
    --build-arg FEATURES="multiling-chinese,multiling-japanese,multiling-korean" \
    --tag=qdrant/qdrant:${QDRANT_VERSION}-cjk
