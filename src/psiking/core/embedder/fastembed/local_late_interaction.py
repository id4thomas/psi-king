from typing import List, Literal, Optional, Tuple, Union, TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

from core.base.schema import (
    MediaResource,
    Document,
    TextType,
    TextLabel,
    TableType,
    Modality,
    TextNode,
    ImageNode,
    TableNode,
)
from core.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from core.base.schema import BaseNode
    
    from fastembed import SparseTextEmbedding
    from fastembed.sparse.sparse_embedding_base import (
        LateInteractionTextEmbedding,
        SparseTextEmbeddingBase
    )
    
class LocalFastEmbedColBERTEmbedder(BaseEmbedder):
    """Embedder using fastembed.SparseTextEmbedding locally"""
    
    def __init__(
        self,
        model: "SparseTextEmbeddingBase"
    ):
        """
        model should be loaded & injected from outside
        SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        """
        self.model = model
        
    def embed(
        self,
        texts: List[str],
        batch_size: int = 16,
        **kwargs
    ) -> List["SparseEmbedding"]:
        """
        Embed given text into SparseEmbedding instances
        embeddings = list(model.embed(documents))
        -> 
        [
            SparseEmbedding(indices=[ 17, 123, 919, ... ], values=[0.71, 0.22, 0.39, ...]),
            SparseEmbedding(indices=[ 38,  12,  91, ... ], values=[0.11, 0.22, 0.39, ...])
        ]
        
        args, kwargs available here
        https://github.com/qdrant/fastembed/blob/a931f143ef3543234bc9d8d0c305496c67199972/fastembed/sparse/sparse_text_embedding.py#L87
        https://github.com/qdrant/fastembed/blob/a931f143ef3543234bc9d8d0c305496c67199972/fastembed/text/onnx_text_model.py#L93
        
        batch_size: Batch size for encoding -- higher values will use more memory, but be faster
        parallel:
            If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
            If 0, use all available cores.
            If None, don't use data-parallel processing, use default onnxruntime threading instead.
        """
        embeddings = list(
            self.model.embed(
                texts,
                batch_size=batch_size,
                **kwargs
            )
        )
        return embeddings
        
    def run(
        self,
        texts: List[str],
        batch_size: int = 16,
        **kwargs
    ) -> Tuple[
        List[List[float]], List[List[int]], 
    ]:
        embeddings: List["SparseEmbedding"] = self.embed(texts, batch_size=batch_size, **kwargs)
        embedding_values = [x.values.tolist() for x in embeddings]
        embedding_indices = [x.indices.tolist() for x in embeddings]
        return embedding_values, embedding_indices