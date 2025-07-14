# allganize-rag-evaluation
* examples of dealing with multimodal real-world PDF files
* Dataset: `allganize/RAG-Evaluation-Dataset-KO` dataset [hflink](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
    * PDF files used in the dataset were collected separately

**Process**
1. Prepare Dataset
2. Read Documents
3. Index Chunks
4. Evaluate Retrieval Performance


## 2. Read Documents
Read PDF Files into PSIKing `Document`s and store them into DocumentStore
* Reader: Try DoclingPDFReader (docling pdf backend) -> Use PDF2ImageReader as fallback

```
1. Load Document Readers
    1-1. Load DoclingPDFReader
        1-1-1. Initialize Docling Converter
        1-1-2. Initialize PSIKing Reader
    1-2. Load PDF2ImageReader
2. Load PDF File Data
3. Ingest Data
    3-1. (Reader) PDF File -> PSIKing Document
    3-2. (Splitter) Chunk Documents
```


## 3. Indexing
| docs | modality | updated | notebook |
| --- | --- | --- | --- |
| [text-hybrid](./docs/indexing_text-hybrid_v2507.md) | text | 2025.07 | [[link]](./3_index_text-hybrid_v2507.ipynb) |
| [multimodal-hybrid](./docs/indexing_multimodal-hybrid_v2507.md) | text+image | 2025.07 | [[link]](./3_index_multimodal-hybrid_v2507.ipynb) |

<!-- | [text](./docs/indexing_text.md) | text | 2025.07 |  | -->
<!-- | [colpali](./docs/indexing_colpali.md) | image | 2025.07 | pdf2image + `colSmol-500M` for colpali style late-interaction | -->
<!-- | [multimodal](./docs/indexing_multimodal.md) | text+image | 2025.07 | docling reader + `jina-embeddings-v4` for dense-only multimodal embedding | -->

Embedding Models:
| vector | model |
| --- | --- |
| dense | `jina-embeddings-v4-vllm-retrieval` |
| sparse | `Qdrant/bm42-all-minilm-l6-v2-attentions` (bm42) |

## 4. Retrieval Evaluation (Updated 2025.07)
Chunk retrieval performance on `finance` (10 PDF Files, 60 Queries)
* hybrid search uses `RRF` fusion

Embedders:
| method | modality | embedders |
| --- | --- | --- |
| text | text | `jina-embeddings-v4-vllm-retrieval` (dense) + `Qdrant/bm42-all-minilm-l6-v2-attentions` (text-sparse) |
| multimodal | text + image | `jina-embeddings-v4-vllm-retrieval` (dense) + `Qdrant/bm42-all-minilm-l6-v2-attentions` (text-sparse) |

<!-- | colpali (`cp`) | image | `` | -->

## 4-1. File-level Relevancy Performance 
| method | mAP@5 | mRR@5 |
| --- | --- | --- |
| text-sparse | 0.3696 | 0.4789 |
| text-dense | 0.5004 | 0.5861 |
| text-hybrid | 0.4641 | 0.5556 |
| multimodal-dense | 0.7662 | 0.8594 |
| multimodal-hybrid | 0.6094 | 0.7653 |


## 4-2. File+Page-level Relevancy Performance
| method | mAP@5 | mRR@5 |
| --- | --- | --- |
| text-sparse | 0.0301 | 0.0628 |
| text-dense | 0.1544 | 0.2867 |
| text-hybrid | 0.1056 | 0.1853 |
| multimodal-dense | 0.2208 | 0.4428 |
| multimodal-hybrid | 0.1604 | 0.3172 |
