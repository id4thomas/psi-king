from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Type, Union

from core.base.schema import Document
from core.base.component import BaseComponent

class BaseDocumentProcessor(BaseComponent):
    """The base class for all document processors"""
    
    def run(self, document: Document) -> Document:
        """Run the processor on the document"""
        ...