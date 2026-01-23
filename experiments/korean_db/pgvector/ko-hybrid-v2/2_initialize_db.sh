#!/bin/bash

PGVECTOR_VERSION="0.8.1-pg18-trixie"
POSTGRES_PORT=9010

echo "pgvector-ko:${PGVECTOR_VERSION}"

STORAGE_NAME="pg_vector_data"

docker volume rm ${STORAGE_NAME}

docker run -it --rm \
  --name postgres-init \
  --env-file ./.env \
  -v ${STORAGE_NAME}:/var/lib/postgresql \
  -v ./db-initialization:/docker-entrypoint-initdb.d \
  -p ${POSTGRES_PORT:-6024}:5432 \
  pgvector-ko:${PGVECTOR_VERSION}