from typing import List, Literal, Optional, Union, TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel

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
    import torch
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin
    )
    from fastembed import SparseTextEmbedding
    from core.base.schema import BaseNode
    
class LocalColpaliEngineEmbedder(BaseEmbedder):
    """Embedder using local Colpali Engine"""
    
    def __init__(
        self,
        model: "SparseTextEmbedding",
        tokenizer: "PretrainedTokenizerBase",
        # device: Optional["torch.device"]=None
    ):
        """
        model example:
        SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        """
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        
    def embed(
        self, texts: list[str], batch_size: int = 16
    ) -> "torch.Tensor":
        