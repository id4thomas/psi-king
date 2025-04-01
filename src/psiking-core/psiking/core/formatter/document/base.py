from pathlib import Path
from typing import Any, List, Type, Union, TYPE_CHECKING

from psiking.core.base.component import BaseComponent
from psiking.core.base.schema import (
    Document,
    TextNode,
    ImageNode,
    TableNode
)

class BaseDocumentFormatter(BaseComponent):
    """The base class for all document formatters"""
    _allowed_nodes: list = [
        TextNode,
        ImageNode,
        TableNode
    ]
    
    def __init__(self):
        pass

    def run(
        self, documents: List[Document]
    ) -> Any:
        """Receives list of documents & returns formatted objects"""
        pass
    