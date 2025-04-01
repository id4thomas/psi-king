"""Splitter using langchain-text-splitter package"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter as LCRecursiveCharacterTextSplitter

from psiking.core.base.schema import TextNode
from psiking.core.splitter.text.base import BaseTextSplitter

class LangchainRecursiveCharacterTextSplitter(BaseTextSplitter):
    """Splitter using langchain-text-splitter package"""
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,   
    ):
        self.splitter = LCRecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    
    def run(self, node: TextNode) -> List[TextNode]:
        text = node.text
        label = node.label
        metadata = node.metadata
        if text is None:
            return []
        return [TextNode(text=t, label=label, metadata=metadata) for t in self.splitter.split_text(text)]
