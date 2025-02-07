from abc import ABC, abstractmethod
from typing import List, Optional, Union

from core.base.schema import Document

class BaseVectorStore(ABC):
    """Vector storage for a single collection"""
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...
        
    @abstractmethod
    def add(
        self,
        documents: Union[Document, List[Document]],
        embeddings: Union[
            List[float], List[List[float]],
            List[int], List[List[int]]
        ],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        """Add vector embeddings to vector stores

        Args:
            documents: List of documents
            embeddings: List of embeddings (support up to dim 2)
            metadatas: List of metadata of the embeddings
            ids: List of ids of the embeddings
            kwargs: meant for vectorstore-specific parameters

        Returns:
            List of ids of the inserted embeddings
        """
        ...
        
    @abstractmethod
    def delete(self, ids: list[str], **kwargs):
        """Delete vector embeddings from vector stores

        Args:
            ids: List of ids of the embeddings to be deleted
            kwargs: meant for vectorstore-specific parameters
        """
        ...
        
    @abstractmethod
    def query(
        self,
        embedding: Union[List[float], List[int]],
        top_k: int = 1,
        ids: Optional[list[str]] = None,
        **kwargs,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """Return the top k most similar vector embeddings

        Args:
            embedding: List of embeddings
            top_k: Number of most similar embeddings to return
            ids: List of ids of the embeddings to be queried

        Returns:
            the matched embeddings, the similarity scores, and the ids
        """
        ...

    @abstractmethod
    def drop(self):
        """Drop the vector store"""
        ...