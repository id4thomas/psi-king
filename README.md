# psi-king
framework for building multi-modal first document retriever

![psi_king](docs/figs/psi_king.png)
> PSI King - King of the Senses from Psychonauts 2

## Structure
### Concepts
**Document / Node (TextNode, ImageNode, TableNode)**
![document](docs/figs/main_document_temp.png)
* a `Document` contains a list of nodes (`document.nodes`)
* each of these can be one of the following
    * `TextNode`
    * `ImageNode`
    * `TableNode`
* schemas are defined [here](src/psiking/core)

![nodes](docs/figs/main_node_temp.png)

### Pipeline Flow
![nodes](docs/figs/main_ingestion_pipeline_temp.png)

### Folder Structure
```
```

## Acknowledgements
* A lot of the structure of this project was inspired by llama-index
    * https://github.com/run-llama/llama_index
* document parsing heavily utilizes docling
    * https://github.com/DS4SD/docling
* `PSI King` is a character from Psychonauts 2
    * https://www.doublefine.com/games/psychonauts-2