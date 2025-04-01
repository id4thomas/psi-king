from typing import List, Literal, Optional, Union, TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel

from psiking.core.base.schema import (
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

from psiking.core.formatter.document.base import BaseDocumentFormatter
from psiking.core.formatter.node import (
    TextNodeTemplate,
    ImageNodeTemplate,
    TableNodeTemplate,
    SimpleTextNodeTemplate,
    SimpleImageNodeTextOnlyTemplate,
    SimpleTableNodeTextOnlyTemplate,
    format_text_node,
    format_image_node,
    format_table_node
)

class SimpleTextOnlyFormatter(BaseDocumentFormatter):
    """Formats using all the text in document"""
    _allowed_nodes: list = [
        TextNode,
        ImageNode,
        TableNode
    ]
    
    def __init__(
        self,
        text_node_template: TextNodeTemplate = SimpleTextNodeTemplate,
        image_node_template: TextNodeTemplate = SimpleImageNodeTextOnlyTemplate,
        table_node_template: TableNodeTemplate = SimpleTableNodeTextOnlyTemplate
    ):
        self.text_node_template = text_node_template
        self.image_node_template = image_node_template
        self.table_node_template = table_node_template
    
    def _get_textnode_text(self, node: TextNode):
        return format_text_node(
            node = node,
            template = self.text_node_template
        )
    
    def _get_imagenode_text(self, node: ImageNode):
        formatted_text, _ = format_image_node(
            node = node,
            template = self.image_node_template
        )
        return formatted_text
    
    def _get_tablenode_text(self, node: TableNode):
        formatted_text, _ = format_table_node(
            node = node,
            template = self.table_node_template
        )
        return formatted_text
    
    def format(self, document: Document) -> str:
        formatted_node_texts = []
        for node in document.nodes:
            if not type(node) in self._allowed_nodes:
                continue
            
            if isinstance(node, TextNode):
                formatted_text = self._get_textnode_text(node)
            elif isinstance(node, ImageNode):
                formatted_text = self._get_imagenode_text(node)
            elif isinstance(node, TableNode):
                formatted_text = self._get_tablenode_text(node)
            
            formatted_node_texts.append(formatted_text)
        return "\n".join(formatted_node_texts)
    
    def run(self, documents: List[Document]) -> List[str]:
        """Receives list of documents & returns formatted strings"""
        formatted_texts = [
            self.format(x) for x in documents
        ]
        return formatted_texts