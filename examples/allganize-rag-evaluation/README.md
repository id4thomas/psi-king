# allganize-rag-evaluation
* examples of dealing with multimodal real-world PDF files
* `allganize/RAG-Evaluation-Dataset-KO` dataset [hflink](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
    * PDF files used in the dataset were collected separately

| title | description | updated |
| --- | --- | --- |
| [ingestion_colpali](./ingestion_colpali_v2507.ipynb) | pdf2image + `colSmol-500M` for colpali style late-interaction | 2025.07 |
| [ingestion_multimodal](./ingestion_multimodal_v2507.ipynb) | docling reader + `bge-visualized` for dense-only multimodal embedding | 2025.07 |
| [ingestion_multimodal_hybrid](./ingestion_multimodal_hybrid_v2507.ipynb) | docling reader (pdf2image fallback) + `bge-visualized` + `qdrant/bm42` for hybrid multimodal embedding | 2025.07 |

## Details
### ingestion_multimodal
Process:
```
1. Load DoclingPDFReader
    1-1. Initialize Docling Converter
    1-2. Initialize PSIKing Reader
2. Load PDF File Data
3. Ingest Data
    3-1. (Reader) PDF File -> PSIKing Document
    3-2. (Splitter) Chunk Documents
4. Embed
5. Insert into DocumentStore, VectorStore
    5-1. Insert to DocStore
    5-2. Insert to VectorStore
6. Query
```