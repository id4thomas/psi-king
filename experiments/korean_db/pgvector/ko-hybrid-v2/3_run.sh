#!/bin/bash

PGVECTOR_VERSION="0.8.1-pg18-trixie"
POSTGRES_PORT=9010
echo "pgvector-ko:${PGVECTOR_VERSION}"

STORAGE_NAME="pg_vector_data"

docker run -it --rm \
    --env-file ./.env \
    -p ${POSTGRES_PORT:-6024}:5432 \
    -v ${STORAGE_NAME}:/var/lib/postgresql \
    pgvector-ko:${PGVECTOR_VERSION}