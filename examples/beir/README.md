# BEIR Ingestion Example
* examples of 'text-only' dataset
* BEIR dataset: [hflink](https://huggingface.co/BeIR)
    * use `scifact` dataset (5K passages)

**Notebooks**:
| title | description | updated |
| --- | --- | --- |
| [scifact_ingestion_dense](./scifact_ingestion_dense.ipynb) | indexing for dense-only search using bge-m3 | 2025.07 |
| [scifact_ingestion_hybrid](./scifact_ingestion_hybrid.ipynb) | indexing for hybrid search using bge-m3 + BM42 | 2025.07 |
| [scifact_ingestion_late_interaction](./scifact_ingestion_late_interaction.ipynb) | indexing for late-interaction search using colbert | 2025.07 |

## details

### scifact_ingestion_late_interaction
colbert model: `sigridjineth/ModernBERT-Korean-ColBERT-preview-v1`
* https://huggingface.co/sigridjineth/ModernBERT-Korean-ColBERT-preview-v1
    * ModernBERT based model
* uses `pylate` package
    * https://github.com/lightonai/pylate
