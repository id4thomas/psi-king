# Pylate Embedder
* https://github.com/lightonai/pylate

## Example
loading model
```
from pylate import models
model = models.ColBERT(
    model_name_or_path="sigridjineth/ModernBERT-Korean-ColBERT-preview-v1",
    document_length=300,
)
documents_embeddings = model.encode(
    sentences=texts,
    batch_size=32,
    is_query=False,
    show_progress_bar=True,
)
assert len(embeddings) == 2
```