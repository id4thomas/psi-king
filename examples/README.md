# Examples
* examples using psiking framework

## 1. Core Schema
* examples of using the core schemas of 
* Document, TextNode, ImageNode, TableNode

**Notebooks**:
| title | description | link |
| --- | --- | --- |
| 1_core_schemas | `Document, TextNode, ImageNode, TableNode` with Docling PDF Reader | [link](./1_core_schemas.ipynb) |


## 2. BEIR dataset example
* examples of 'text-only' dataset
* BEIR dataset: [hflink](https://huggingface.co/BeIR)
    * use `scifact` dataset (5K passages)

**Notebooks**:
| title | description | link |
| --- | --- | --- |
| 2_1_beir_ingestion_dense | indexing for dense-only search using bge-m3 | [link](./2_1_beir_ingestion_dense.ipynb) |
| 2_2_beir_ingestion_hybrid | indexing for hybrid search using bge-m3 + BM42 | [link](./2_2_beir_ingestion_hybrid.ipynb) |
| 2_3_beir_ingestion_late_interaction | indexing for late-interaction search using colbert | [link](./2_3_beir_ingestion_late_interaction.ipynb) |

### details
**2_3_beir_ingestion_late_interaction**

colbert model: `sigridjineth/ModernBERT-Korean-ColBERT-preview-v1`
* https://huggingface.co/sigridjineth/ModernBERT-Korean-ColBERT-preview-v1
    * ModernBERT based model
* uses `pylate` package
    * https://github.com/lightonai/pylate


## 3. allganize/rag-evaluation dataset example
* examples of dealing with multimodal real-world PDF files
* `allganize/RAG-Evaluation-Dataset-KO` dataset [hflink](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
    * PDF files used in the dataset were collected separately

**Notebooks**:
| title | description | link |
| --- | --- | --- |
| 3_ | ingestion example | [link]() |
