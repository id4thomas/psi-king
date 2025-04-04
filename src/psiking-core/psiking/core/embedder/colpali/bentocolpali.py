# Inference in the style of bentocolpali served with BentoML
# https://github.com/bentoml/BentoColPali
import base64
from enum import Enum
from io import BytesIO
from typing import List, Literal, Optional, Union, TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel

from psiking.core.base.schema import (
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

from psiking.core.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from bentoml import (
        SyncHTTPClient as BentoMLSyncHTTPClient,
        AsyncHTTPClient as BentoMLAsyncHTTPClient,
    )
    
    from psiking.core.base.schema import BaseNode
    
class ImageDetail(Enum):
    """
    Supported image details for `ImagePayload`.
    """
    LOW = "low"
    HIGH = "high"
    AUTO = "auto"
    
def convert_pil_to_b64_image(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 string.
    """
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


class ImagePayload(BaseModel):
    """
    Image payload for `ColPaliService`.

    Args:
        url: The URL of the image or a base 64 encoded image.
        detail: The detail of the image. Defaults to `ImageURLDetail.AUTO`.

    NOTE: `detail` is unused for now and only here to be ISO with the OpenAI API.
    """
    url: str
    text: Optional[str] = None
    detail: ImageDetail = ImageDetail.AUTO
    
class LocalColpaliEngineEmbedder(BaseEmbedder):
    """Embedder using local Colpali Engine"""
    _allowed_nodes: list = [
        TextNode,
        ImageNode,
        TableNode
    ]
    _client: "BentoMLSyncHTTPClient" = None
    _aclient: "BentoMLAsyncHTTPClient" = None
    
    def __init__(
        self,
        *args,
        client: Optional["BentoMLSyncHTTPClient"] = None,
        aclient: Optional["BentoMLAsyncHTTPClient"] = None,
        **kwargs
    ):
        try:
            from bentoml import (
                SyncHTTPClient as BentoMLSyncHTTPClient,
                AsyncHTTPClient as BentoMLAsyncHTTPClient,
            )
        except ImportError:
            raise ImportError("Please install bentoml: 'pip install bentoml'")
        
        if client is None and aclient is None:
            raise ValueError("Must provide either a bentoml SyncHTTPClient or AsyncHTTPClient instance")
        self._client = client
        self._aclient = aclient
        
    def _get_images_from_nodes(self, nodes: List[ImageNode]) -> List[Image.Image]:
        images = []
        for node in nodes:
            if not node.image_loaded:
                node.load_image_data()
            image = node.image
            images.append(image)
        return images
        
        
    def _embed_queries(
        self, texts: List[str], 
    ) -> List[List[List[float]]]:
        with self._client as client:
            embeddings = client.embed_queries(items=texts)
        return embeddings.tolist()
    
    async def _aembed_queries(
        self, texts: List[str], 
    ) -> List[List[List[float]]]:
        async with self._aclient as client:
            embeddings = await client.embed_queries(items=texts)
        return embeddings.tolist()
    
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