"""Splitter using langchain-text-splitter package"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter as LCRecursiveCharacterTextSplitter

from psiking.core.base.schema import ImageNode
from psiking.core.splitter.multimodal.base import BaseITMultimodalSplitter

class ITLangchainRecursiveCharacterTextSplitter(BaseITMultimodalSplitter):
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
    
    def run(self, node: ImageNode) -> List[ImageNode]:
        text = node.text
        image_loaded = node.image_loaded
        image_resource = node.image_resource
        caption_resource = node.caption_resource
        
        metadata = node.metadata
        
        if text is None:
            '''Need to keep image'''
            return [node]

        return [
            ImageNode(
                text=t,
                image_loaded=image_loaded,
                image_resource=image_resource,
                caption_resource=caption_resource,
                metadata=metadata
            )
            for t in self.splitter.split_text(text)
        ]
