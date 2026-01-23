#!/bin/bash
PGVECTOR_VERSION="0.8.1-pg18-trixie"
# --platform linux/amd64 

# tested on aarch64 env (dgx spark)
docker build -t pgvector-ko:${PGVECTOR_VERSION} -f Dockerfile.arm .