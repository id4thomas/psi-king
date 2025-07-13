# allganize-rag-evaluation Multimodal Ingestion
## Methodology
**Ingestion** [[notebook]](./ingestion_v2507.ipynb)
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

**Evaluation** [[notebook]](./evaluate_retrieval_v2507.ipynb)
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

## Performance