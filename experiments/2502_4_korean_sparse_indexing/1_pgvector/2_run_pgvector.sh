#!/bin/bash
source .env
echo "User: ${POSTGRES_USER}"
echo "DB Name: ${POSTGRES_DB}"
echo "ENV: ${APP_ENV}"

POSTGRES_VERSION="16"

docker container rm -f pgvector-psiking-db

docker run \
    --name pgvector-psiking-db \
    -e POSTGRES_USER=${POSTGRES_USER:-id4thomas} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password} \
    -e POSTGRES_DB=${POSTGRES_DB:-pgvector_psiking} \
    -p ${POSTGRES_PORT:-6024}:5432 \
    -v ./local_storage:/var/lib/postgresql/data \
    pgvector-${POSTGRES_VERSION}-mecab:latest