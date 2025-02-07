from typing import List, Optional, Union, TYPE_CHECKING

from core.base.schema import (
    MediaResource,
    Document,
    TextType,
    TextLabel,
    TableType,
    Modality,
    TextNode,
    ImageNode,
    TableNode,
)
from core.processor.document.base import BaseDocumentProcessor

class TextNodeMerger(BaseDocumentProcessor):
    """Merge text nodes into a single text node"""
    
    def run(self, document: Document) -> Document:
        def _create_text_node(text: str) -> TextNode:
            return TextNode(
                text=text,
                label=TextLabel.PLAIN,
                metadata={}
            )
        
        merged_nodes = []
        _merge_window = []
        stop_threshold = 5000
        for node in document.nodes:
            if not isinstance(node, TextNode):
                # stop merging
                if _merge_window:
                    merged_node = _create_text_node("\n".join(_merge_window))
                    merged_nodes.append(merged_node)
                    _merge_window = []
                    
                merged_nodes.append(node)
            else:
                text = node.text
                _merge_window.append(text)
                
                if len("\n".join(_merge_window)) > stop_threshold:
                    merged_node = _create_text_node("\n".join(_merge_window))
                    merged_nodes.append(merged_node)
                    _merge_window = []
        
        # Create new document
        return Document(
            nodes=merged_nodes,
            metadata=document.metadata
        )