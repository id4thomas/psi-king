# psi-king
framework for building multi-modal first document retriever
<img src="docs/figs/psi_king.png" width=70% height=70%>
<!-- ![psi_king](docs/figs/psi_king.png){: width="80%" height="80%"} -->
> PSI King - King of the Senses from Psychonauts 2

## Overview
* [psiking-core docs](docs/psiking_core.md)

### Document / Node (TextNode, ImageNode, TableNode)
![document](docs/figs/main_document.png)
* a `Document` contains a list of nodes (`document.nodes`)
* each node can be one of the following types
    * `TextNode`
    * `ImageNode`
    * `TableNode`
* schemas are defined [here](src/psiking-core/psiking/core/base/schema.py)
    * detailed descriptions are available [here](docs/psiking_core.md)


### Ingestion Pipeline
Document Ingestion Flow example:
* (Doc) Collection -> Extraction -> Transformation -> Index(?)
    * Extraction: read file into `Document` instance
    * Transformation: merging/chunking/filtering
    * Index: Embedding & inserting into searchable DB

![nodes](docs/figs/main_ingestion_pipeline_temp.png)

## Example
### allganize-RAG-Evaluation-Dataset-KO PDF dataset
Reading PDF files and indexing into qdrant DB for retrieval ([[examples/allganize-rag-evaluation]](./examples/allganize-rag-evaluation/))
* **data**: real-life pdf files from [`allganize-RAG-Evaluation-Dataset-KO`](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
* **models**: [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4-vllm-retrieval/discussions/1) + [Qdrant/BM42](https://huggingface.co/Qdrant/all_miniLM_L6_v2_with_attentions)
* **db**: qdrant (dense + sparse)


**Pipeline Overview**
![3_3_overview](./examples/allganize-rag-evaluation/docs/figs/ingestion_multimodal_hybrid_overview.png)


### BeIR (scifact) Retrieval Benchmark
Indexing BeIR text retrieval benchmark ([[./examples/beir]](./examples/beir))
* **data**: `scifact` data from BeIR: [hflink](https://huggingface.co/BeIR)
* **models**: [bge-m3](https://huggingface.co/BAAI/bge-m3), [Qdrant/BM42](https://huggingface.co/Qdrant/all_miniLM_L6_v2_with_attentions)
* **db**: qdrant (dense + sparse)

## Experiments
### Korean sparse search with vectordb
* pgvector [[docs]](./docs/vectordb/pgvector/korean_text_search_with_pg.md) [[experiments]](./experiments/2502_4_korean_sparse_indexing/1_pgvector/)
    * use mecab-ko + `textsearch_ko` to enable korean `tsvector` calculation
* qdrant [[docs]](./docs/vectordb/qdrant/building_with_cjk_lang_support.md) [[experiments]](./experiments/2502_4_korean_sparse_indexing/2_qdrant/)
    * build qdrant with cjk language support for korean tokenization

## Acknowledgements
* A lot of the structure of this project was inspired by llama-index
    * https://github.com/run-llama/llama_index
* document parsing heavily utilizes docling
    * https://github.com/DS4SD/docling
* `PSI King` is a character from Psychonauts 2
    * https://www.doublefine.com/games/psychonauts-2


History of this framework's development is recorded below
* https://github.com/id4thomas/nlp_building_blocks/tree/main/projects/2024_11_arxiver_rag/experiments