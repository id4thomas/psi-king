# Qdrant + CJK Support
* looking at cjk (chinese/japanese/korean) lang support with qdrant

full-text index
* https://qdrant.tech/documentation/concepts/indexing/#full-text-index
    * special type of tokenizer based on charabia package
    *  Korean not enabled by default, but can be enabled by building qdrant from source with --features multiling-chinese,multiling-japanese,multiling-korean flags.
    * charabia
        * https://github.com/meilisearch/charabia?tab=readme-ov-file
        * used in meilisearch
        * uses linedera Ko-dict

Creating Index for text search indexing
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

## Resources
* https://velog.io/@silveris23/meilisearch-로-간단하게-RAG
* https://choiseokwon.tistory.com/420
* lindera ko-dict
    * https://github.com/lindera/lindera/tree/main/lindera-ko-dic
    * repository contains mecab-ko-dic.
    * https://docs.rs/lindera-ko-dic/latest/lindera_ko_dic/