# PSI-King Core
## Schemas
![nodes](figs/main_node_temp.png)

## Structure
Folder Structure:
```
├── base/
│   ├── component.py
│   ├── constants.py
│   └── schema.py
├── embedder/
│   ├── colpali/
│   ├── fastembed/
│   └── openai/
├── formatter/
│   ├── document
│   └── node
├── processor/
│   ├── document
│   └── node
├── reader/
│   ├── pdf/
├── splitter
│   ├── image/
│   └── text/
└── storage
    ├── collectionstore/
    ├── docstore/
    └── vectorstore/
```

## Reader
* read file and make a Document instance
    * `file path / file` -> `Document

## Processor
* process documents (ex. merge nodes, preprocess)
    * `Document` -> `List[Document]`

## Splitter
* split document into chunks
    * `Document` -> `List[Document]`

## Formatter
* format chunks into contextual chunks (metadata enriched)
    * `Document (chunk)` -> `str / Image.Image`

## Embedder
* embed formatted contents
    * `str / Image.Image` -> embedding
    * intentionally don't receive Document/Node as input, embedder should work as it's own without dependency to psiking schemas

## Storage
### docstore (document storage)
* store document /chunks (`Document`s)

### vectorstore (Vector storage)
* store chunk & embedding (`Document` & embeddings)