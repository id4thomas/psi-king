# MultiModal Embedders
* Goal: find embedding models for embedding text+image in passage/query

## Model Candidates
* `BAAI/bge-visualized` [hf_link](https://huggingface.co/BAAI/bge-visualized)
    * provides 2 text-embedding versions (bge-base-en-v1.5/bge-m3)
    * uses `FlagEmbedding.visual_bge` package
* `openbmb/VisRAG-Ret` [hf_link](https://huggingface.co/openbmb/VisRAG-Ret)
    * 3.4B (MiniCPM-V 2.0 base)

non-commercial options
* `jinaai/jina-clip-v2` [hf_link](https://huggingface.co/jinaai/jina-clip-v2)
    * 0.9B params, non-commercial (provides API)
* `nvidia/MM-Embed` [hf_link](https://huggingface.co/nvidia/MM-Embed)
    * 8B params (llava-v1.6-mistral-7b-hf + NV-Embed-v1), non-commercial