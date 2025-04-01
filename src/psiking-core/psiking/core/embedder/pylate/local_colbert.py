from typing import List, Literal, Optional, Tuple, Union, TYPE_CHECKING

from psiking.core.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    import numpy as np
    from pylate.models import ColBERT
    
 
class LocalPylateColBERTEmbedder(BaseEmbedder):
    """Embedder using pylate.models.ColBERT locally"""
    
    def __init__(
        self,
        model: "ColBERT"
    ):
        """
        model should be loaded & injected from outside
        ColBERT(
            model_name_or_path=model_dir,
            document_length=None, # only set if you need to override
            device="mps",
            prompts={"query": "query: ", "passage": "passage: "} # input prefix text
        )
        """
        self.model = model
        
    def embed(
        self,
        texts: List[str],
        batch_size: int = 16,
        is_query: bool = False,
        **kwargs
    ) -> List["np.ndarray"]:
        """
        Embed given text into SparseEmbedding instances
        embeddings = model.encode(
            sentences=texts,
            batch_size=32,
            is_query=False,
            show_progress_bar=True,
        )
        -> 
        [shape (19, 128) arr, shape (29, 128) arr, ...]
        
        args, kwargs available here
        https://github.com/lightonai/pylate/blob/fe115ff8bd93351670d516859952804ced1198f7/pylate/models/colbert.py#L384
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            is_query=is_query,
            **kwargs
        )
        return embeddings
        
    def run(
        self,
        texts: List[str],
        batch_size: int = 16,
        is_query: bool = False,
        **kwargs
    ) -> List[List[List[float]]]:
        embeddings: List["np.ndarray"] = self.embed(
            texts,
            batch_size=batch_size,
            is_query=is_query,
            **kwargs
        )
        return [x.tolist() for x in embeddings]