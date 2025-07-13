# allganize-rag-evaluation
* examples of dealing with multimodal real-world PDF files
* Dataset: `allganize/RAG-Evaluation-Dataset-KO` dataset [hflink](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
    * PDF files used in the dataset were collected separately

## Examples
Examples grouped by Ingestion Methodology
| method | modality | updated | description |
| --- | --- | --- | --- |
| [colpali](./colpali) | image | 2025.07 | pdf2image + `colSmol-500M` for colpali style late-interaction |
| [multimodal](./multimodal) | text+image | 2025.07 | docling reader + `jina-embeddings-v4` for dense-only multimodal embedding |
| [multimodal_hybrid](./multimodal_hybrid) | text+image | 2025.07 | docling reader (pdf2image fallback) + `jina-embeddings-v4` + `qdrant/bm42` for hybrid multimodal embedding |


## Performance (Updated 2025.07)
Chunk retrieval performance on `finance` (10 PDF Files, 60 Queries)

Embedders:
| method | modality | embedders |
| --- | --- | --- |
| colpali | image | `` |
| multimodal | text + image | `jina-embeddings-v4-vllm-retrieval` |
| multimodal_hybrid | text + image | `jina-embeddings-v4-vllm-retrieval` (dense) + `` (text-sparse) |

### File-level Relevancy Performance 
| method | mAP@5 | mRR@5 |
| --- | --- | --- |
| colpali | | |
| multimodal | 0.7758 | 0.8678 |
| multimodal_hybrid | | |


### File+Page-level Relevancy Performance
| method | mAP@5 | mRR@5 |
| --- | --- | --- |
| colpali | | |
| multimodal | 0.2231| 0.4381 |
| multimodal_hybrid | | |
