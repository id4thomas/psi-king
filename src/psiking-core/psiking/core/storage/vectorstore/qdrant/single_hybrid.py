import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, TYPE_CHECKING

from grpc import RpcError

from psiking.core.base.schema import (
    Document,
    BaseNode,
    TextNode,
    ImageNode,
    TableNode,
    doc_to_json,
    json_to_doc
)
from psiking.core.storage.vectorstore.base import BaseVectorStore
from psiking.core.storage.vectorstore.utils import document_to_metadata_dict
from psiking.core.storage.vectorstore.qdrant.base import BaseQdrantVectorStore

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.https.models import (
        PointStruct,
        SparseVectorParams,
        VectorParams
    )

class QdrantSingleHybridVectorStore(BaseQdrantVectorStore):
    """
    qdrant based vectorstore for dense+sparse hybrid vector retrieval
    
    following qdrant field names are used by the VectorStore
    * `vector_dense`: dense vector
    * `vector_sparse`: sparse vector
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    # override create_collection
    def create_collection(
        self,
        dense_vector_config: "VectorParams",
        sparse_vector_config: "SparseVectorParams",
        **kwargs
    ):
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._create_collection(
            vectors_config={self._dense_vector_name: dense_vector_config},
            sparse_vectors_config={self._sparse_vector_name: sparse_vector_config},
            **kwargs
        )
        self._collection_initialized = True
        
    async def acreate_collection(
        self,
        dense_vector_config: "VectorParams",
        sparse_vector_config: "SparseVectorParams",
        **kwargs
    ):
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._acreate_collection(
            vectors_config={self._dense_vector_name: dense_vector_config},
            sparse_vectors_config={self._sparse_vector_name: sparse_vector_config},
            **kwargs
        )
        self._collection_initialized = True

    def _validate_embedding(self, embedding):
        if not (
            isinstance(embedding, list)
            and (
                isinstance(embedding[0], float)
                or isinstance(embedding[0], int)
            )
        ):
            raise ValueError("given embedding is not 1d list")

    def _build_points(
        self,
        documents: List[BaseNode],
        dense_embeddings: List[
            Union[List[float], List[int]]
        ],
        sparse_embedding_values: List[
            Union[List[float], List[int]]
        ],
        sparse_embedding_indices: List[List[int]],
        metadata_keys: Optional[List[str]] = None,
    ) -> List["PointStruct"]:
        """
        Convet documents to qdrant PointStruct instances
        Args
            documents
            dense_embeddings: list of 1D dense embeddings
            sparse_embedding_values: list of 1D sparse embedding values (token score)
            sparse_embedding_indices: list of 1D sparse embedding indices (token ids)
            metadata_keys: keys of document.metadata to store in point (default: everything)
        Returns:
            points:
        """
        from qdrant_client.http.models import PointStruct, SparseVector

        points = []
        for i in range(len(documents)):
            document = documents[i]
            dense_embedding = dense_embeddings[i]
            sparse_index = sparse_embedding_indices[i]
            sparse_embedding = sparse_embedding_values[i]
            
            # validate input
            self._validate_embedding(dense_embedding)
            # Don't validate sparse embedding (ex. ImageNode with empty values)
            # self._validate_embedding(sparse_embedding)
            # self._validate_embedding(sparse_index)
                
            # prepare payload
            metadata = document_to_metadata_dict(
                document, keys=metadata_keys, flat_metadata=self.flat_metadata
            )
            point = PointStruct(
                id=document.id_,
                payload=metadata,
                vector={
                    self._dense_vector_name: dense_embedding,
                    self._sparse_vector_name: SparseVector(
                        indices=sparse_index,
                        values=sparse_embedding
                    ),
                }
            )
            points.append(point)
        
        return points
    
    def add(
        self,
        documents: Union[Document, List[Document]],
        dense_embeddings: List[
            Union[List[float], List[int]]
        ],
        sparse_embedding_values: List[
            Union[List[float], List[int]]
        ],
        sparse_embedding_indices: List[List[int]],
        metadata_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Add documents to index.
        Args:
            documents: List[Document]: list of documents
            dense_embeddings
            sparse_indices: (SparseEmbedding.indices)
            sparse_embeddings: (SparseEmbedding.values)
            metadata_keys
        """
        
        if not isinstance(documents, list):
            documents = [documents]
            dense_embeddings = [dense_embeddings]
            sparse_embedding_values = [sparse_embedding_values]
            sparse_embedding_indices = [sparse_embedding_indices]
            
        if len(documents) > 0 and not self._collection_initialized:
            raise ValueError(
                f"Collection {self.collection_name} is not initialized, create it first"
            )
            
        if len(documents)!=len(dense_embeddings):
            raise ValueError(
                "Number of documents, dense embeddings must be same. documents: {}, embeddings:{}".format(
                    len(documents), len(dense_embeddings)
                )
            )
        elif len(documents)!=len(sparse_embedding_values) or len(documents)!=len(sparse_embedding_indices):
            raise ValueError(
                "Number of documents, sparse embeddings must be same. documents: {}, embeddings:{} indices {}".format(
                    len(documents), len(sparse_embedding_values), len(sparse_embedding_indices)
                )
            )
            
        points = self._build_points(
            documents=documents,
            dense_embeddings=dense_embeddings,
            sparse_embedding_values=sparse_embedding_values,
            sparse_embedding_indices=sparse_embedding_indices,
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

    def delete(self):
        pass
    
    def drop(self):
        pass