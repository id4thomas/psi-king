import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, TYPE_CHECKING

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
    from qdrant_client.https.models import (
        PointStruct,
        SparseVectorParams,
        VectorParams
    )

class QdrantSingleHybridVectorStore(BaseQdrantVectorStore):
    """
    qdrant based vectorstore for single vector retrieval
    
    following field names are used by the VectorStore
    * `text`: text for IDF indexing
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
            vectors_config={"vector_dense": dense_vector_config},
            sparse_vectors_config={"vector_sparse": sparse_vector_config},
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
            vectors_config={"vector_dense": dense_vector_config},
            sparse_vectors_config={"vector_sparse": sparse_vector_config},
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
        texts: List[str],
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
            embeddings: list of 1D embeddings
            metadata_keys: keys of document.metadata to store in point (default: everything)
        Returns:
            points:
        """
        from qdrant_client.http.models import PointStruct, SparseVector

        points = []
        for i in range(len(documents)):
            document = documents[i]
            text = texts[i]
            dense_embedding = dense_embeddings[i]
            sparse_index = sparse_embedding_indices[i]
            sparse_embedding = sparse_embedding_values[i]
            
            # validate input
            self._validate_embedding(dense_embedding)
            # Don't validate sparse embedding (ex. ImageNode with empty values)
            # self._validate_embedding(sparse_embedding)
            # self._validate_embedding(sparse_index)
            if not isinstance(text, str):
                raise ValueError("given text is not a string")
                
            # prepare payload
            metadata = document_to_metadata_dict(
                document, keys=metadata_keys, flat_metadata=self.flat_metadata
            )
            point = PointStruct(
                id=document.id_,
                payload=metadata,
                vector={
                    "vector_dense": dense_embedding,
                    "vector_sparse": SparseVector(
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
        texts: Union[str, List[str]],
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
            texts = [texts]
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
        elif len(documents)!=len(texts):
            raise ValueError(
                "Number of documents, texts must be same. documents: {}, texts:{}".format(
                    len(documents), len(texts)
                )
            )
            
        points = self._build_points(
            documents=documents,
            texts=texts,
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
    
    def _dense_query(
        self, 
        dense_embedding: Optional[Union[List[float], List[int]]] = None,
        limit: int = 10
    ):
        # TODO - separate option to be injectable
        points = self._client.query_points(
            collection_name=self.collection_name,
            query=dense_embedding,
            using="vector_dense",
            limit=limit
        )
        return points
    
    def _sparse_query(
        self, 
        sparse_embedding_values: Optional[List[float]] = None,
        sparse_embedding_indices: Optional[List[int]] = None,
        limit: int = 10,
    ):
        from qdrant_client.http.models import SparseVector
        # TODO - separate option to be injectable
        points = self._client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=sparse_embedding_indices,
                values=sparse_embedding_values
            ),
            using="vector_sparse",
            limit=limit
        )
        return points
    
    def _hybrid_query(
        self, 
        dense_embedding: Optional[Union[List[float], List[int]]] = None,
        sparse_embedding_values: Optional[List[float]] = None,
        sparse_embedding_indices: Optional[List[int]] = None,
        limit: int = 10,
        dense_limit: int = 100,
        sparse_limit: int = 100,
    ):
        from qdrant_client.http.models import Prefetch, Fusion, FusionQuery
        # TODO - separate option to be injectable
        points = self._client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query={
                        "values": sparse_embedding_values,
                        "indices": sparse_embedding_indices
                    },
                    using="vector_sparse",
                    limit=sparse_limit
                ),
                Prefetch(
                    query=dense_embedding,
                    using="vector_dense",
                    limit=dense_limit
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF), # RRF combination
            limit=limit
        )
        return points
    
    def query(
        self,
        mode: Literal["sparse", "dense", "hybrid"] = "hybrid",
        dense_embedding: Optional[Union[List[float], List[int]]] = None,
        sparse_embedding_values: Optional[List[float]] = None,
        sparse_embedding_indices: Optional[List[int]] = None,
        limit: int = 10,
        dense_limit: int = 100,
        sparse_limit: int = 100,
    ):
        if mode=="hybrid":
            points = self._hybrid_query(
                dense_embedding=dense_embedding,
                sparse_embedding_values=sparse_embedding_values,
                sparse_embedding_indices=sparse_embedding_indices,
                limit=limit,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit
            )
        elif mode=="dense":
            points = self._dense_query(
                dense_embedding=dense_embedding,
                limit=limit,
                dense_limit=dense_limit,
            )
        elif mode=="sparse":
            points = self._sparse_query(
                sparse_embedding_values=sparse_embedding_values,
                sparse_embedding_indices=sparse_embedding_indices,
                limit=limit,
                dense_limit=dense_limit,
            )
            
        else:
            raise NotImplementedError()
        return points

    def drop(self):
        pass