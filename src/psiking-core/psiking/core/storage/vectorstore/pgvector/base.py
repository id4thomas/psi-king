import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

from grpc import RpcError

from psiking.core.storage.vectorstore.base import BaseVectorStore

class BasePGVectorVectorStore(BaseVectorStore):
    """Base pgvector vectorstore. mostly follows llama-index"""
    collection_name: str
    _collection_initialized: bool = False
    _client: "QdrantClient" = None
    _aclient: "AsyncQdrantClient" = None
    
    