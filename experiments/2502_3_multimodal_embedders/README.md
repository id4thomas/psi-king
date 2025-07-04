# MultiModal Embedders
* Goal: find embedding models for embedding text+image in passage/query

## Model Candidates
* `moca-embed/MoCa-Qwen25VL` [hf_link - 3B](https://huggingface.co/moca-embed/MoCa-Qwen25VL-3B), [hf_link - 7B](https://huggingface.co/moca-embed/MoCa-Qwen25VL-7B)
    * provides 2 versions (3B/7B)
    * bidirectional-attention enabled Qwen2.5-VL
* `BAAI/bge-visualized` [hf_link](https://huggingface.co/BAAI/bge-visualized)
    * provides 2 text-embedding versions (bge-base-en-v1.5/bge-m3)
    * uses `FlagEmbedding.visual_bge` package
* `openbmb/VisRAG-Ret` [hf_link](https://huggingface.co/openbmb/VisRAG-Ret)
    * 3.4B (MiniCPM-V 2.0 base)

non-commercial options
* `jinaai/jina-embeddings-v4` [hf_link](https://huggingface.co/jinaai/jina-embeddings-v4)
    * provides 3 adapters (`retrieval`, `text-matching`, `code`)
    * provides separate adapter-merged versions for vllm deployment
* `jinaai/jina-clip-v2` [hf_link](https://huggingface.co/jinaai/jina-clip-v2)
    * 0.9B params, non-commercial (provides API)
* `nvidia/MM-Embed` [hf_link](https://huggingface.co/nvidia/MM-Embed)
    * 8B params (llava-v1.6-mistral-7b-hf + NV-Embed-v1), non-commercial