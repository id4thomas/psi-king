#!/bin/bash
source .env
echo "User: ${POSTGRES_USER}"
echo "DB Name: ${POSTGRES_DB}"
echo "ENV: ${APP_ENV}"

POSTGRES_VERSION="16"

docker run \
  --platform linux/amd64 \
  --name postgres-init \
  -e POSTGRES_USER=${POSTGRES_USER:-id4thomas} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password} \
    -e POSTGRES_DB=${POSTGRES_DB:-pgvector_psiking} \
  -v ./local_storage:/var/lib/postgresql/data \
  -v ./db-initialization:/docker-entrypoint-initdb.d \
  -p ${POSTGRES_PORT:-6024}:5432 \
  pgvector-${POSTGRES_VERSION}-mecab:latest

docker container rm -f postgres-init