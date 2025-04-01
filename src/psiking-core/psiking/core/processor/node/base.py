from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Type, Union

from psiking.core.base.schema import BaseNode
from psiking.core.base.component import BaseComponent

class BaseNodeProcessor(BaseComponent):
    """The base class for all node processors"""
    
    def run(self, node: BaseNode) -> BaseNode:
        """Run the processor on the node"""
        ...