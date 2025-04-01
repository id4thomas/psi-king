import json
from pathlib import Path
from typing import List, Optional, Union

from psiking.core.base.schema import (
    Document,
    doc_to_json,
    json_to_doc
)
from psiking.core.storage.docstore.base import BaseDocumentStore

class InMemoryDocumentStore(BaseDocumentStore):
    """Simple in-memory document store
    implementation taken from kotaemon
    https://github.com/Cinnamon/kotaemon/blob/f5b2200ffa797cfbcf8e84ba1a3a9d33baa29283/libs/kotaemon/kotaemon/storages/docstores/in_memory.py#L10
    """
    def __init__(self):
        self._store = {}
        
    def add(
        self,
        docs: Union[Document, List[Document]],
        ids: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        """Add document into document store

        Args:
            docs: list of documents to add
            ids: specify the ids of documents to add or
                use existing doc.doc_id
            exist_ok: raise error when duplicate doc-id
                found in the docstore (default to False)
        """
        exist_ok: bool = kwargs.pop("exist_ok", False)

        if ids and not isinstance(ids, list):
            ids = [ids]
        if not isinstance(docs, list):
            docs = [docs]
        doc_ids = ids if ids else [doc.id_ for doc in docs]

        for doc_id, doc in zip(doc_ids, docs):
            if doc_id in self._store and not exist_ok:
                raise ValueError(f"Document with id {doc_id} already exist")
            self._store[doc_id] = doc
            
    async def aadd(
        self,
        docs: Union[Document, List[Document]],
        ids: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        raise NotImplementedError()

    def get(self, ids: Union[List[str], str]) -> List[Document]:
        """Get document by id"""
        if not isinstance(ids, list):
            ids = [ids]

        return [self._store[doc_id] for doc_id in ids]
    
    async def aget(self, ids: Union[List[str], str]) -> List[Document]:
        raise NotImplementedError()
    
    def count(self) -> int:
        """Count number of documents"""
        return len(self._store)
    
    def delete(self, ids: Union[List[str], str]):
        """Delete document by id"""
        if not isinstance(ids, list):
            ids = [ids]

        for doc_id in ids:
            del self._store[doc_id]
            
    def save(self, path: Union[str, Path]):
        """Save document to path"""
        store = {key: doc_to_json(value) for key, value in self._store.items()}
        with open(path, "w") as f:
            json.dump(store, f)
            
    def load(self, path: Union[str, Path]):
        """Load document store from path"""
        with open(path) as f:
            store = json.load(f)
        self._store = {key: json_to_doc(value) for key, value in store.items()}

    def search(self,):
        pass
            
    def drop(self):
        """Drop the document store"""
        self._store = {}
