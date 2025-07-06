from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Type, Union

from psiking.core.base.schema import TextNode, ImageNode
from psiking.core.base.component import BaseComponent

'''
Base classes for all multimodal splitters
[Modal Keywords]
* IT: Image+Text
'''

class BaseITMultimodalSplitter(BaseComponent):
    """The base class for Image+Text Multimodal Splitter"""
    
    def run(self, node: ImageNode) -> List[ImageNode]:
        """Run the splitter on the ImageNode's text"""
        ...