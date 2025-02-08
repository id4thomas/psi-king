import json
from typing import List, Literal, Optional, Union, TYPE_CHECKING

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# TODO - Implement retry
# import backoff

from core.base.schema import (
    Document,
    TextNode,
    ImageNode,
    TableNode
)
from core.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from pydantic import BaseModel


class OpenAITextEmbedder(BaseEmbedder):
    _client: "OpenAI" = None
    _aclient: "AsyncOpenAI" = None
    
    def __init__(
        self,
        *args,
        client: Optional["OpenAI"] = None,
        aclient: Optional["AsyncOpenAI"] = None,
        **kwargs
    ):
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError("Please install qdrant-client: 'pip install qdrant-client'")
        
        if client is None and aclient is None:
            raise ValueError("Must provide either a OpenAI or AsyncOpenAI instance")
        self._client = client
        self._aclient = aclient
        
    def _embed(
        self, texts: Union[str, List[str]], model: str = "text-embedding-3-small", dimensions: int = None
    ):
        if dimensions is None:
            response = self._client.embeddings.create(
                model=model,
                input=texts,
                encoding_format="float"
            )
        else:
            response = self._client.embeddings.create(
                model=model,
                input=texts,
                dimensions=dimensions,
                encoding_format="float"
            )
        return [x.embedding for x in response.data]
        
    async def _aembed(
        self, texts: Union[str, List[str]], model: str = "text-embedding-3-small", dimensions: int = None
    ):
        """Return Embedding object"""
        if dimensions is None:
            response = await self._aclient.embeddings.create(
                model=model,
                input=texts,
                encoding_format="float"
            )
        else:
            response = await self._aclient.embeddings.create(
                model=model,
                input=texts,
                dimensions=dimensions,
                encoding_format="float"
            )
        return [x.embedding for x in response.data]
    
    def run(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        batch_size: int = 4,
        dimensions: Optional[int] = None,
        disable_tqdm: bool = True
    ):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), disable=disable_tqdm):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._embed(
                texts=batch,
                model=model,
                dimensions=dimensions
            )
            embeddings.extend(batch_embeddings)
        return embeddings

    async def arun(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        batch_size: int = 4,
        dimensions: Optional[int] = None,
        disable_tqdm: bool = True
    ):
        embeddings = []
        for i in atqdm(range(0, len(texts), batch_size), disable=disable_tqdm):
            batch = texts[i:i+batch_size]
            batch_embeddings = await self._aembed(
                texts=batch,
                model=model,
                dimensions=dimensions
            )
            embeddings.extend(batch_embeddings)
        return embeddings