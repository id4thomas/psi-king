from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Type, Union

from core.base.schema import TextNode
from core.base.component import BaseComponent

class BaseTextSplitter(BaseComponent):
    """The base class for all text splitters"""
    
    def run(self, node: TextNode) -> List[TextNode]:
        """Run the splitter on the text"""
        ...