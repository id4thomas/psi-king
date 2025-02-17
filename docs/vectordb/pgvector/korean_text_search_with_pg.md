# PGVector + textsearch_ko
* pgvector로 한국어 hybrid 검색을 위해 `textsearch_ko`를 사용하여 한국어 full-textsearch를 사용한다
    * `select to_tsvector('korean', content) as parsed`
* 실험 폴더: [link](../../../experiments/2502_4_korean_sparse_indexing/1_pgvector/)

## Overview
* `textsearch_ko`: mecab 형태소 분석기를 이용하여 ts_vector를 구현

### Example
sqalchemy 기준 사용 예시
* example from [llama-index](https://github.com/run-llama/llama_index/blob/ea04280768cc6026ecc4ff715ccfce0446907912/llama-index-integrations/vector_stores/llama-index-vector-stores-postgres/llama_index/vector_stores/postgres/base.py#L76)
    * when passing `text_search_config="korean"`, llama-index creates gin index for tsvector
    * gin index: 각 키워드별로 해당 키워드를 포함하는 entry를 인덱싱
```
from sqlalchemy.dialects.postgresql import BIGINT, JSON, JSONB, TSVECTOR, VARCHAR
from sqlalchemy.schema import Index

class HybridAbstractData(base):  # type: ignore
    ...
    text = Column(VARCHAR, nullable=False)
    text_search_tsv = Column(  # type: ignore
        TSVector(),
        # compute tsvector from `text` column
        Computed(
            "to_tsvector('%s', text)" % text_search_config, persisted=True
        ),
    )

Index(
    indexname,
    model.text_search_tsv,  # type: ignore
    postgresql_using="gin",
)
```

## Usage
### Dockerfile
* [Dockerfile](../../../experiments/2502_4_korean_sparse_indexing/1_pgvector/Dockerfile)
* `pgvector/pgvector` 이미지 기반으로 작업
* `mecab-ko` 설치 (mecab-ko, mecab-ko-dic)
* `textsearch_ko` 빌드

### pg에 한국어 검색 등록
* `docker-entrypoint-initdb.d`에 초기화 sql을 넣어서 설정 등록을 진행한다
* [초기화 sql](../../../experiments/2502_4_korean_sparse_indexing/1_pgvector/db-initialization/ts_mecab_ko.sql)
```
...
CREATE TEXT SEARCH PARSER korean (
    START    = ts_mecabko_start,
    GETTOKEN = ts_mecabko_gettoken,
    END      = ts_mecabko_end,
    HEADLINE = pg_catalog.prsd_headline,
    LEXTYPES = pg_catalog.prsd_lextype
);
...
CREATE TEXT SEARCH CONFIGURATION korean (PARSER = korean);
```

## Resources
* [PostgreSQL 16 한글 검색 설정](https://taejoone.jeju.onl/posts/2024-01-27-postgres-16-korean/#2-gin-인덱스)
* [github - i0seph/textsearch_ko](https://github.com/i0seph/textsearch_ko)
* [onbaba - PostgreSQL GIN Index](https://onbaba.tistory.com/2)