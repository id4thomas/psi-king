from enum import Enum
from typing import List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel

# Metadata Filters
'''
# Metadata Filters
* follows llama-index
* https://github.com/run-llama/llama_index/blob/080faffa641bbb1b0f2304bee58faf58143b9223/llama-index-core/llama_index/core/vector_stores/types.py#L94
'''

class FilterOperator(str, Enum):
    """Vector store filter operator."""

    # TODO add more operators
    EQ = "=="  # default operator (string, int, float)
    GT = ">"  # greater than (int, float)
    LT = "<"  # less than (int, float)
    NE = "!="  # not equal to (string, int, float)
    GTE = ">="  # greater than or equal to (int, float)
    LTE = "<="  # less than or equal to (int, float)
    IN = "in"  # In array (string or number)
    NIN = "nin"  # Not in array (string or number)
    ANY = "any"  # Contains any (array of strings)
    ALL = "all"  # Contains all (array of strings)
    TEXT_MATCH = "text_match"  # full text match (allows you to search for a specific substring, token or phrase within the text field)
    TEXT_MATCH_INSENSITIVE = (
        "text_match_insensitive"  # full text match (case insensitive)
    )
    CONTAINS = "contains"  # metadata array contains value (string or number)
    IS_EMPTY = "is_empty"  # the field is not exist or empty (null or empty array)

class FilterCondition(str, Enum):
    """Vector store filter conditions to combine different filters."""

    # TODO add more conditions
    AND = "and"
    OR = "or"
    NOT = "not"  # negates the filter condition

class MetadataFilter(BaseModel):
    key: str
    value: Optional[
        Union[
            int,
            float,
            str,
            List[int],
            List[float],
            List[str],
        ]
    ]
    operator: FilterOperator = FilterOperator.EQ


class MetadataFilters(BaseModel):
    """Metadata filters for vector stores."""

    # Exact match filters and Advanced filters with operators like >, <, >=, <=, !=, etc.
    filters: List[Union[MetadataFilter, "MetadataFilters"]]
    condition: Optional[FilterCondition] = FilterCondition.AND


# Query
'''
# VectorStore Query
* follows llama-index
* https://github.com/run-llama/llama_index/blob/0327e9e1b041d602f3e8d41fcb60b95cda8f21fd/llama-index-core/llama_index/core/vector_stores/types.py#L240
'''

class VectorStoreQueryMode(str, Enum):
    DENSE="dense"
    SPARSE="sparse"
    HYBRID="hybrid"

class VectorStoreQuery(BaseModel):
    '''Common query data type used for vectorstore retrieval
    
    query_embedding:
    * 1D (List[float], List[int]): single embedding
    * 2D (List[List[float]], List[List[int]]): late-interaction styles
    '''
    text: Optional[str] = None
    dense_embedding: Optional[Union[
        List[float],
        List[List[float]],
        List[int],
        List[List[int]],
        # np.ndarray
    ]]=None
    
    # Sparse
    sparse_embedding_values: Optional[Union[
        List[float],
        List[int]
    ]]=None
    
    sparse_embedding_indicies: Optional[List[int]]=None
    
    
class VectorStoreQueryOptions(BaseModel):
    mode: VectorStoreQueryMode = VectorStoreQueryMode.DENSE
    
    top_k: int = 1
    
    # use values below only in hybrid mode
    sparse_top_k: Optional[int] = None
    dense_top_k: Optional[int] = None
    
    # Hybrid Options
    alpha: Optional[float] = None
    hybrid_fusion_method: Literal['rrf', 'dbsf'] ='rrf'
    
    filters: Optional[MetadataFilters]=None
    