import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from grpc import RpcError

from core.storage.vectorstore.base import BaseVectorStore

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, AsyncQdrantClient


class BaseQdrantVectorStore(BaseVectorStore):
    """Base qdrant vectorstore. mostly follows llama-index"""
    collection_name: str
    _collection_initialized: bool = False
    _client: "QdrantClient" = None
    _aclient: "AsyncQdrantClient" = None
    
    flat_metadata: bool = True
    
    def __init__(
        self,
        collection_name: str,
        *args,
        client: Optional["QdrantClient"] = None,
        aclient: Optional["AsyncQdrantClient"] = None,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
        **kwargs
    ):
        # Check qdrant-client import
        try:
            from qdrant_client import QdrantClient, AsyncQdrantClient
        except ImportError:
            raise ImportError("Please install qdrant-client: 'pip install qdrant-client'")
        
        if client is None and aclient is None:
            raise ValueError("Must provide either a QdrantClient or AsyncQdrantClient instance")
        self._client = client
        self._aclient = aclient
            
        # Check if Collection Exists
        self.collection_name = collection_name
        if self._client is not None:
            self._collection_initialized = self._collection_exists(collection_name)
        else:
            #  need to do lazy init for async clients
            self._collection_initialized = False
            
        self.batch_size=batch_size
        self.parallel=parallel
        self.max_retries=max_retries
        
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return self._client.collection_exists(collection_name)

    async def _acollection_exists(self, collection_name: str) -> bool:
        """Asynchronous method to check if a collection exists."""
        return await self._aclient.collection_exists(collection_name)
    
    def _create_collection(self, *args, **kwargs):
        """Create collection"""
        from qdrant_client.http.exceptions import UnexpectedResponse
        try:
            self._client.create_collection(
                collection_name=self.collection_name,
                **kwargs
            )
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            raise ValueError(f"Collection {self.collection_name} already exists")
        
    async def _acreate_collection(self, *args, **kwargs):
        """Asynchronously Create collection"""
        from qdrant_client.http.exceptions import UnexpectedResponse
        
        try:
            self._aclient.create_collection(
                collection_name=self.collection_name,
            )
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            raise ValueError(f"Collection {self.collection_name} already exists")
        
    def create(self, **kwargs):
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._create_collection(**kwargs)
        self._collection_initialized = True
        
    async def acreate(self, **kwargs):
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._acreate_collection(**kwargs)
        self._collection_initialized = True
        
    def _build_points(self):
        raise NotImplementedError()
    
    def add(self):
        raise NotImplementedError()
    
    def delete(self):
        raise NotImplementedError()
    
    def query(self):
        raise NotImplementedError()

    def drop(self):
        raise NotImplementedError()