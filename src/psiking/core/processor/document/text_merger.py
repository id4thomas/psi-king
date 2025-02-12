from typing import Callable, List, Optional, Union, TYPE_CHECKING

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
    """Merge text nodes of same text_type (ex. html, markdown, ...) into a single text node"""
    def __init__(self, *args, metadata_merger: Callable[[dict], dict] = None, **kwargs):
        self.metadata_merger = metadata_merger
        super().__init__(*args, **kwargs)
    
    def merge(self, nodes: List[TextNode]) -> TextNode:
        if len(nodes)==0:
            return None
        text = "\n".join([x.text for x in nodes if x.text])
        text_type =  nodes[-1].text_type
        if not self.metadata_merger:
            metadata = nodes[0].metadata
        else:
            metadata = self.metadata_merger(
                [x.metadata for x in nodes]
            )
        return TextNode(
            text=text,
            text_type=text_type,
            label=TextLabel.PLAIN,
            metadata=metadata
        )
        
    
    def run(self, document: Document) -> Document:
        merged_nodes = []
        stop_threshold = 5000
        
        window = []
        window_text_size = 0
        text_type = None
        for node in document.nodes:
            if not isinstance(node, TextNode):
                # stop merging
                if window:
                    merged_node = self.merge(window)
                    merged_nodes.append(merged_node)
                    window = []
                    window_text_size = 0
                    text_type = None
                merged_nodes.append(node)
                continue
            elif not text_type is None and node.text_type==text_type:
                window.append(node)
                window_text_size+=len(node.text)
            else:
                # different text_type -> stop merging
                if window:
                    merged_node = self.merge(window)
                    merged_nodes.append(merged_node)
                window = [node]
                text_type = node.text_type
                window_text_size=len(node.text)

            # check if over stop threshold
            if window_text_size > stop_threshold:
                merged_node = self.merge(window)
                merged_nodes.append(merged_node)
                text_type=None
                window = []
        
        # Create new document
        return Document(
            nodes=merged_nodes,
            metadata=document.metadata
        )