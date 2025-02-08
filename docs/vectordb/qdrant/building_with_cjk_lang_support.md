# Qdrant + CJK
cjk (chinese, japanese, korean) tokenization support with qdrant

## Creating Collection
**full-text index**
* https://qdrant.tech/documentation/concepts/indexing/#full-text-index
    * special type of tokenizer based on charabia package
    *  Korean not enabled by default, but can be enabled by building qdrant from source with flags
        * `--features multiling-chinese,multiling-japanese,multiling-korean`
* charabia
    * https://github.com/meilisearch/charabia?tab=readme-ov-file
    * used in meilisearch
    * uses linedera Ko-dict


Example:
* using IDF (Inverse document frequency) modifier
```
client.create_collection(
    collection_name="{collection_name}",
    vectors_config={
        "{dense_vector_field}": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "{sparse_vector_field}": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    }
)
```

### Creating index
* set specific fields to index with `field_name`
* set tokenizer to `TokenizerType.MULTILINGUAL`

```
client.create_payload_index(
    collection_name="{collection_name}",
    field_name="name_of_the_field_with_text_to_index",
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.MULTILINGUAL,
    ),
)
```

## building qdrant
* CJK support is not enabled by default, but can be enabled by building qdrant from source with --features 

Building qdrant docker image with features
```
QDRANT_VERSION="v1.13.2"
docker build . \
    --build-arg FEATURES="multiling-chinese,multiling-japanese,multiling-korean" \
    --tag=qdrant/qdrant:${QDRANT_VERSION}-cjk
```

## Resources
### Qdrant Full-text index documentation
* https://qdrant.tech/documentation/concepts/indexing/#full-text-index

### qdrant's korean implementations
* https://velog.io/@silveris23/meilisearch-로-간단하게-RAG
* https://choiseokwon.tistory.com/420
* lindera ko-dict
    * https://github.com/lindera/lindera/tree/main/lindera-ko-dic
    * repository contains mecab-ko-dic.
    * https://docs.rs/lindera-ko-dic/latest/lindera_ko_dic/