from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Type, Union

from core.base.component  import BaseComponent

from core.base.schema import (
    Document,
    TextNode,
    ImageNode,
    TableNode
)

class BaseEmbedder(BaseComponent):
    """The base class for all embedders"""
    _allowed_nodes: list = [
        TextNode,
        ImageNode,
        TableNode
    ]

    def run(
        self, documents: List[Document]
    ) -> Union[
        List[int], List[List[int]],
        List[float], List[List[float]],
    ]:
        """Receives list of documents & returns embeddings"""
        pass
    