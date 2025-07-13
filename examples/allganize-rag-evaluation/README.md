# allganize-rag-evaluation
* examples of dealing with multimodal real-world PDF files
* `allganize/RAG-Evaluation-Dataset-KO` dataset [hflink](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
    * PDF files used in the dataset were collected separately

| title | description | updated |
| --- | --- | --- |
| [ingestion_colpali](./ingestion_colpali_v2507.ipynb) | pdf2image + `colSmol-500M` for colpali style late-interaction | 2025.07 |
| [ingestion_multimodal](./ingestion_multimodal_v2507.ipynb) | docling reader + `bge-visualized` for dense-only multimodal embedding | 2025.07 |
| [ingestion_multimodal_hybrid](./ingestion_multimodal_hybrid_v2507.ipynb) | docling reader (pdf2image fallback) + `bge-visualized` + `qdrant/bm42` for hybrid multimodal embedding | 2025.07 |

## Performance (v2507)
Chunk retrieval performance on `finance` (10 PDF Files, 60 Queries)

Methods:
| method | modality | embedders |
| --- | --- | --- |
| colpali | image | `` |
| multimodal | text + image | `jina-embeddings-v4-vllm-retrieval` |
| multimodal_hybrid | text + image | `jina-embeddings-v4-vllm-retrieval` (dense) + `` (text-sparse) |



**File-level** Relevancy Performance 
| method | mAP@5 | mRR@5 |
| --- | --- | --- |
| colpali | | |
| multimodal | 0.7758 | 0.8678 |
| multimodal_hybrid | | |


**File+Page-level** Relevancy Performance
| method | mAP@5 | mRR@5 |
| --- | --- | --- |
| colpali | | |
| multimodal | 0.2231| 0.4381 |
| multimodal_hybrid | | |

## Process
### ingestion_multimodal (v2507)
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
4. Embed
5. Insert into DocumentStore, VectorStore
    5-1. Insert to DocStore
    5-2. Insert to VectorStore
6. Test Query
```

### evaluate_retrieval (v2507)
```
1. Load DocStore, VectorStore
    1-1. Load DocStore
    1-2. Load VectorStore
2. Initialize Embedder
3. Load Evaluation Data
    3-1. Load Query & Ground Truth
    3-2. Calculate Query Embeddings
4. Run Retrieval
```