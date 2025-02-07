import copy
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from core.base.schema import (
    TextNode,
    ImageNode,
    TableNode
)
from core.formatter.node.schema import (
    TextNodeTemplate,
    ImageNodeTemplate,
    TableNodeTemplate,
    TEXT_RESOURCE_KEY,
    IMAGE_RESOURCE_KEY,
    CAPTION_RESOURCE_KEY
)

def fill_text_template(
    template: str, contents: Dict[str, str], keys: List[str] = []
) -> str:
    '''
    Receive prompt template in the format "..<|key1|>...<|key2|>" and fill in the placeholders with contents
    Args:
        template str: Template string containing placeholders in the form of "<|{KEY VALUE}|>"
        contents dict: Dictionary containing values to fill the placeholders (ex. {"key1": "...", ...})
    Returns:
        filled str: template filled with contents
    '''
    if not isinstance(contents, dict):
        raise ValueError("contents should be a dictionary Dict[str, str]")
    
    filled = copy.deepcopy(template)
    for k in keys:
        if k not in contents:
            raise ValueError(
                "key {} not found in contents. only contains {}".format(
                    k, list(contents.keys())
                )
            )
        filled = filled.replace(f"<|{k}|>", contents[k])
    return filled

def format_text_node(
    node: TextNode,
    template: TextNodeTemplate
) -> str:
    """Make text from node
    1. build contents from node
    2. format template with contents
    """
    # Make contents from node
    contents = {}
    for k in template.content_keys:
        if k==TEXT_RESOURCE_KEY:
            contents[k] = node.text
        else:
            v = node.metadata.get(k, "")
            contents[k] = v    
    
    formatted_text = fill_text_template(
        template = template.text_template,
        contents = contents,
        keys = template.content_keys
    )
    return formatted_text

def format_image_node(
    node: ImageNode,
    template: ImageNodeTemplate,
    ignore_image: bool = False
) -> Tuple[Optional[str], Optional[Image.Image]]:
    """Make text, image from node
    1. build contents from node
    2. format template with contents
    
    Image placeholder token must be provided as metadata
    """
    # Make contents from node
    text_contents = {}
    image_contents = None
    for k in template.content_keys:
        if k==TEXT_RESOURCE_KEY:
            text_contents[k] = node.text
        elif k==CAPTION_RESOURCE_KEY:
            text_contents[k] = node.caption
        else:
            v = node.metadata.get(k, "")
            text_contents[k] = v
    
    if not ignore_image:
        image_contents = node.image
    
    formatted_text = fill_text_template(
        template = template.text_template,
        contents = text_contents,
        keys = template.content_keys
    )
    return formatted_text, image_contents

def format_table_node(
    node: TableNode,
    template: TableNodeTemplate,
    ignore_image: bool = False
) -> Tuple[Optional[str], Optional[Image.Image]]:
    """Make text, image from node
    1. build contents from node
    2. format template with contents
    
    Image placeholder token must be provided as metadata
    """
    # Make contents from node
    text_contents = {}
    image_contents = None
    for k in template.content_keys:
        if k==TEXT_RESOURCE_KEY:
            text_contents[k] = node.text
        elif k==CAPTION_RESOURCE_KEY:
            text_contents[k] = node.caption
        else:
            v = node.metadata.get(k, "")
            text_contents[k] = v
    
    if not ignore_image:
        image_contents = node.image
    
    formatted_text = fill_text_template(
        template = template.text_template,
        contents = text_contents,
        keys = template.content_keys
    )
    return formatted_text, image_contents