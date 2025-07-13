import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

from grpc import RpcError

from psiking.core.storage.vectorstore.base import (
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_SPARSE_VECTOR_NAME,
    BaseVectorStore
)
from psiking.core.storage.vectorstore.schema import (
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryOptions,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http.models import (
        Filter,
        FilterCondition,
        QueryResponse,
        ScoredPoint
    )

# Filters
'''Code from llama-index git
'''
def build_subfilter(filters: MetadataFilters) -> "Filter":
    from qdrant_client.http.models import (
        FieldCondition,
        Filter,
        Range,
        MatchAny,
        MatchExcept,
        MatchText,
        MatchValue,
        IsEmptyCondition,
        PayloadField
    )
    
    conditions = []
    for subfilter in filters.filters:
        # only for exact match
        if isinstance(subfilter, MetadataFilters) and len(subfilter.filters) > 0:
            conditions.append(build_subfilter(subfilter))
        elif not subfilter.operator or subfilter.operator == FilterOperator.EQ:
            if isinstance(subfilter.value, float):
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(
                            gte=subfilter.value,
                            lte=subfilter.value,
                        ),
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchValue(value=subfilter.value),
                    )
                )
        elif subfilter.operator == FilterOperator.LT:
            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    range=Range(lt=subfilter.value),
                )
            )
        elif subfilter.operator == FilterOperator.GT:
            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    range=Range(gt=subfilter.value),
                )
            )
        elif subfilter.operator == FilterOperator.GTE:
            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    range=Range(gte=subfilter.value),
                )
            )
        elif subfilter.operator == FilterOperator.LTE:
            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    range=Range(lte=subfilter.value),
                )
            )
        elif (
            subfilter.operator == FilterOperator.TEXT_MATCH
            or subfilter.operator == FilterOperator.TEXT_MATCH_INSENSITIVE
        ):
            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    match=MatchText(text=subfilter.value),
                )
            )
        elif subfilter.operator == FilterOperator.NE:
            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    match=MatchExcept(**{"except": [subfilter.value]}),
                )
            )
        elif subfilter.operator == FilterOperator.IN:
            # match any of the values
            # https://qdrant.tech/documentation/concepts/filtering/#match-any
            if isinstance(subfilter.value, List):
                values = subfilter.value
            else:
                values = str(subfilter.value).split(",")

            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    match=MatchAny(any=values),
                )
            )
        elif subfilter.operator == FilterOperator.NIN:
            # match none of the values
            # https://qdrant.tech/documentation/concepts/filtering/#match-except
            if isinstance(subfilter.value, List):
                values = subfilter.value
            else:
                values = str(subfilter.value).split(",")
            conditions.append(
                FieldCondition(
                    key=subfilter.key,
                    match=MatchExcept(**{"except": values}),
                )
            )
        elif subfilter.operator == FilterOperator.IS_EMPTY:
            # This condition will match all records where the field reports either does not exist, or has null or [] value.
            # https://qdrant.tech/documentation/concepts/filtering/#is-empty
            conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key=subfilter.key))
            )

    qdrant_filter = Filter()
    if filters.condition == FilterCondition.AND:
        qdrant_filter.must = conditions
    elif filters.condition == FilterCondition.OR:
        qdrant_filter.should = conditions
    elif filters.condition == FilterCondition.NOT:
        qdrant_filter.must_not = conditions
    return qdrant_filter
    
def build_search_filter(
    filters: MetadataFilters
) -> "Filter":
    from qdrant_client.http.models import (
        FieldCondition,
        Filter,
        MatchAny,
        MatchExcept,
        MatchText,
        MatchValue,
    )
    must_conditions = []

    # TODO
    # if query.doc_ids:
    #     must_conditions.append(
    #         FieldCondition(
    #             key=DOCUMENT_ID_KEY,
    #             match=MatchAny(any=query.doc_ids),
    #         )
    #     )

    # Point id is a "service" id, it is not stored in payload. There is 'HasId' condition to filter by point id
    # https://qdrant.tech/documentation/concepts/filtering/#has-id
    # if query.node_ids:
    #     must_conditions.append(
    #         HasIdCondition(has_id=query.node_ids),
    #     )

    # Qdrant does not use the query.query_str property for the filtering. Full-text
    # filtering cannot handle longer queries and can effectively filter our all the
    # nodes. See: https://github.com/jerryjliu/llama_index/pull/1181

    if filters and filters.filters:
        must_conditions.append(build_subfilter(filters))

    if len(must_conditions) == 0:
        return None

    return Filter(must=must_conditions)

# SearchRequest


class BaseQdrantVectorStore(BaseVectorStore):
    """Base qdrant vectorstore. mostly follows llama-index"""
    collection_name: str
    _collection_initialized: bool = False
    _client: "QdrantClient" = None
    _aclient: "AsyncQdrantClient" = None
    
    _dense_vector_name="vector_dense"
    _sparse_vector_name="vector_sparse"
    
    flat_metadata: bool = False
    
    def __init__(
        self,
        collection_name: str,
        *args,
        client: Optional["QdrantClient"] = None,
        aclient: Optional["AsyncQdrantClient"] = None,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
        dense_vector_name: Optional[str] = None,
        sparse_vector_name: Optional[str] = None,
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
        
        self._dense_vector_name = dense_vector_name or DEFAULT_DENSE_VECTOR_NAME
        self._sparse_vector_name = sparse_vector_name or DEFAULT_SPARSE_VECTOR_NAME
        
    ## Collection
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
                *args,
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
                *args,
                **kwargs
            )
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            raise ValueError(f"Collection {self.collection_name} already exists")
        
    def create_collection(self, **kwargs):
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._create_collection(**kwargs)
        self._collection_initialized = True
        
    async def acreate_collection(self, **kwargs):
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._acreate_collection(**kwargs)
        self._collection_initialized = True
    
    ## Index
    def _create_payload_index(self, field_name: str, field_schema: str, **kwargs):
        """
        Create payload index, vector indicies must be created with create_collection
        https://qdrant.tech/documentation/concepts/indexing
        """
        self._client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=field_schema,
            **kwargs
        )
        
    def create_index(self, field_name: str, field_schema: Any, **kwargs):
        if not self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} is not initialized")
        
        self._create_payload_index(field_name=field_name, field_schema=field_schema, **kwargs)
        
    async def acreate_index(self, **kwargs):
        pass
    
    ## SearchRequestㅋ
    def _build_search_request(
        self,
        query: VectorStoreQuery,
        options: VectorStoreQueryOptions
    ):
        from qdrant_client.http.models import (
            SparseVector,
            Prefetch
        )
        '''SearchRequest is defined here
        https://github.com/qdrant/qdrant-client/blob/e89c7390c5c4f1330a10114f2c39e5b19222c0e2/qdrant_client/http/models/models.py#L2374
        '''
        if options.filters:
            qdrant_filters=self._build_search_filter(options.filters)
        else:
            qdrant_filters=None
        
        search_request = {
            'limit': options.top_k,
            'query_filter': qdrant_filters,
            'with_payload': True
        }
        
        if options.mode==VectorStoreQueryMode.DENSE:
            if query.dense_embedding is None:
                raise ValueError("query.dense_embedding cannot be None")
            
            search_request['query']=query.dense_embedding
            
            # request = SearchRequest(
            #     vector=query.dense_embedding,
            #     limit=options.top_k,
            #     filter=qdrant_filters,
            #     with_payload=True
            # )
        elif options.mode==VectorStoreQueryMode.SPARSE:
            if query.sparse_embedding_indicies is None or query.sparse_embedding_values is None:
                raise ValueError("query.sparse_embedding_indicies and sparse_embedding_values cannot be None")
            
            search_request['query']=SparseVector(
                indices=query.sparse_embedding_indicies,
                values=query.sparse_embedding_values
            )
            # request = SearchRequest(
            #     vector=NamedSparseVector(
            #         name=self._sparse_vector_name,
            #         vector=SparseVector(
            #             indices=query.sparse_embedding_indicies,
            #             values=query.sparse_embedding_values
            #         )
            #     ),
            #     limit=options.top_k,
            #     filter=qdrant_filters,
            #     with_payload=True
            # )
        elif options.mode==VectorStoreQueryMode.HYBRID:
            if query.dense_embedding is None:
                raise ValueError("query.dense_embedding cannot be None")
            if query.sparse_embedding_indicies is None or query.sparse_embedding_values is None:
                raise ValueError("query.sparse_embedding_indicies and sparse_embedding_values cannot be None")
            
            search_request['prefetch'] = [
                Prefetch(
                    query={
                        "values": query.sparse_embedding_values,
                        "indices": query.sparse_embedding_indicies,
                    },
                    using=self._sparse_vector_name,
                    limit=options.sparse_top_k
                ),
                Prefetch(
                    query=query.dense_embedding,
                    using=self._dense_vector_name,
                    limit=options.dense_top_k
                ),
            ]
            search_request['query'] = {"fusion": options.hybrid_fusion_method}
            
            # request = SearchRequest(
            #     vector=NamedSparseVector(
            #         name=self._sparse_vector_name,
            #         vector=SparseVector(
            #             indices=query.sparse_embedding_indicies,
            #             values=query.sparse_embedding_values
            #         )
            #     ),
            #     limit=options.top_k,
            #     filter=qdrant_filters,
            #     with_payload=True
            # )
            
            
        else:
            raise ValueError("VectorStoreQueryMode {} not supprted".format(options.mode.value))
        
        return search_request

    def _build_points(self):
        raise NotImplementedError()
    
    def add(self):
        raise NotImplementedError()
    
    def delete(self):
        raise NotImplementedError()
    
    def query(
        self,
        query: VectorStoreQuery,
        options: VectorStoreQueryOptions
    ) -> List["ScoredPoint"]:
        '''Use query_points'''
        
        search_request = self._build_search_request(
            query=query,
            options=options
        )
        response: "QueryResponse" = self._client.query_points(
            collection_name=self.collection_name,
            **search_request
        )
        points: List["ScoredPoint"]=response.points
        return points

    def drop(self):
        raise NotImplementedError()