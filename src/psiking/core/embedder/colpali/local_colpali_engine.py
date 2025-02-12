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
    
    from core.base.schema import BaseNode
    
class LocalColpaliEngineEmbedder(BaseEmbedder):
    """Embedder using local Colpali Engine"""
    _allowed_nodes: list = [
        ImageNode
    ]
    
    def __init__(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin",
        tokenizer: "PretrainedTokenizerBase",
    ):
        self.model = model
        self.model.eval()
        
        self.processor = processor
        self.tokenizer = tokenizer
        
    def _get_images_from_nodes(self, nodes: List[ImageNode]) -> List[Image.Image]:
        images = []
        for node in nodes:
            if not node.image_loaded:
                node.load_image_data()
            image = node.image
            images.append(image)
        return images
        
    def embed_queries(
        self,
        queries: List[str],
        batch_size: int = 4
    ) -> List[List[List[float]]]:
        """
        Embed a list of query strings using batching.
        
        Each batch is processed by the processor and then passed through the model.
        The returned embedding for each query is computed by averaging the token-level
        representations (i.e. `last_hidden_state` averaged over tokens).
        """
        try:
            import torch
        except ImportError:
            raise ImportError("Please install torch")
        embeddings = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            with torch.no_grad():
                processed_queries = self.processor.process_queries(
                    batch_queries
                ).to(self.model.dtype).to(self.model.device)
                batch_embeddings = self.model(**processed_queries)
                del processed_queries
                batch_embeddings = batch_embeddings.cpu().detach().tolist()
            embeddings.extend(batch_embeddings)
        return embeddings
    
    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
        **kwargs
    ) -> List[List[List[float]]]:
        """
        Embed a list of images using batching.
        
        Each batch of images is processed via the processor and then forwarded
        through the model. The resulting feature maps are averaged (over spatial dimensions
        if necessary) to yield a single embedding vector per image.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("Please install torch")
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            with torch.no_grad():
                processed_images = self.processor.process_images(
                    batch_images
                ).to(self.model.dtype).to(self.model.device)
                # print(processed_images['input_ids'].shape)
                batch_embeddings = self.model(**processed_images)
                del processed_images
                batch_embeddings = batch_embeddings.cpu().detach().tolist()
            embeddings.extend(batch_embeddings)
        # return torch.cat(embeddings, dim=0)
        return embeddings
    
    def calculate_scores(
        self,
        query_embeddings: "torch.Tensor",
        image_embeddings: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Calculate similarity scores between queries and images.
        Returns:
            A torch.Tensor containing the scores.
        """
        scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)
        return scores
    
    def run(
        self,
        queries: Optional[List[str]] = None,
        nodes: Optional[List["ImageNode"]] = None,
        mode: Literal["image", "query"] = "image",
        batch_size: int = 4,
        **kwargs
    ):
        if mode=="image":
            images: List[Image.Image] = self._get_images_from_nodes(
                nodes
            )
            embeddings = self.embed_images(images, batch_size=batch_size, **kwargs)
        elif mode=="query":
            embeddings = self.embed_queries(queries, batch_size=batch_size, **kwargs)
        else:
            raise ValueError("mode must be one of image, query")
        return embeddings