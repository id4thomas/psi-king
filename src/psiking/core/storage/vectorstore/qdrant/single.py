import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from grpc import RpcError

from core.base.schema import (
    Document,
    BaseNode,
    TextNode,
    ImageNode,
    TableNode,
    doc_to_json,
    json_to_doc
)
from core.storage.vectorstore.base import BaseVectorStore
from core.storage.vectorstore.utils import document_to_metadata_dict
from core.storage.vectorstore.qdrant.base import BaseQdrantVectorStore

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.https.models import PointStruct

class QdrantSingleVectorStore(BaseQdrantVectorStore):
    """
    qdrant based vectorstore for single vector retrieval
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_points(
        self,
        documents: List[BaseNode],
        embeddings: List[
            Union[List[float], List[int]]
        ],
        metadata_keys: Optional[List[str]] = None,
    ) -> List["PointStruct"]:
        """
        Convet documents to qdrant PointStruct instances
        Args
            documents
            embeddings: list of 1D embeddings
            metadata_keys: keys of document.metadata to store in point (default: everything)
        Returns:
            points:
        """
        from qdrant_client.http.models import PointStruct

        points = []
        for document, embedding in zip(documents, embeddings):
            if not (
                isinstance(embedding, list)
                and (
                    isinstance(embedding[0], float)
                    or isinstance(embedding[0], int)
                )
            ):
                raise ValueError("given embedding is not 1d list")
            
            metadata = document_to_metadata_dict(
                document, keys=metadata_keys, flat_metadata=self.flat_metadata
            )
            point = PointStruct(
                id=document.id_,
                payload=metadata,
                vector=embedding
            )
            points.append(point)
        
        return points
    
    def add(
        self,
        documents: Union[Document, List[Document]],
        embeddings: List[
            Union[List[float], List[int]]
        ],
        metadata_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Add documents to index.
        Args:
            documents: List[Document]: list of documents
            embeddings
            metadata_keys
        """
        
        if not isinstance(documents, list):
            documents = [documents]
            embeddings = [embeddings]
        if len(documents) > 0 and not self._collection_initialized:
            raise ValueError(
                f"Collection {self.collection_name} is not initialized, create it first"
            )
        if len(documents)!=len(embeddings):
            raise ValueError(
                "Number of documents and embeddings must be same documents: {}, embeddings:{}".format(
                    len(documents), len(embeddings)
                )
            )
        points = self._build_points(
            documents=documents,
            embeddings=embeddings,
            metadata_keys=metadata_keys
        )

        self._client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=self.batch_size,
            parallel=self.parallel,
            max_retries=self.max_retries,
            wait=True,
        )
    
    async def aadd(self):
        pass
    
    def delete(self):
        pass
    
    def query(self):
        pass

    def drop(self):
        pass