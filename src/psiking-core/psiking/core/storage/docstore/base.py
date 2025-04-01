from abc import ABC, abstractmethod
from typing import List, Optional, Union

from psiking.core.base.schema import (
    Document
)
class BaseDocumentStore(ABC):
    """Document storage"""
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...
        
    @abstractmethod
    def add(
        self,
        docs: Union[Document, List[Document]],
        ids: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        """Add document into document store

        Args:
            docs: Document or list of documents
            ids: List of ids of the documents. Optional, if not set will use doc.doc_id
        """
        ...
        
    @abstractmethod
    async def aadd(
        self,
        docs: Union[Document, List[Document]],
        ids: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        """Add document into document store

        Args:
            docs: Document or list of documents
            ids: List of ids of the documents. Optional, if not set will use doc.doc_id
        """
        ...