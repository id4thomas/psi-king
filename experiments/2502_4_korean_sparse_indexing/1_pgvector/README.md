# PGVector + mecab for fts
* [docs](../../../docs/vectordb/pgvector/korean_text_search_with_pg.md)

## Usage
### 1. Build
* macos (arm64) 설치
```
---> [Warning] The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8) and no specific platform was requested
```

### 2. Initialize DB (text search config 등록)
* `docker-entrypoint-initdb.d`에 mecab 함수 등록하는 sql문 넣어서 mount->초기화

```
docker run \
  --platform linux/amd64 \
  --name postgres-init \
  -e POSTGRES_USER=${POSTGRES_USER:-user} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password} \
    -e POSTGRES_DB=${POSTGRES_DB:-pgvector_psiking} \
  -v ./local_storage:/var/lib/postgresql/data \
  -v ./db-initialization:/docker-entrypoint-initdb.d \
  -p ${POSTGRES_PORT:-6024}:5432 \
  pgvector-${POSTGRES_VERSION}-mecab:latest
```
