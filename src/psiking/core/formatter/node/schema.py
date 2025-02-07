from typing import List

from pydantic import BaseModel, Field

from core.base.schema import (
    TableType
)

"""
Text Template should look like below

ex. 
content keys: ["a", "b"]

'''
... <|a|> ...
... <|b|> ...
... <|a|> ...
'''

"""

TEXT_RESOURCE_KEY = "text_resource"
IMAGE_RESOURCE_KEY = "image_resource"
# IMAGE_PLACEMENT_KEY = "image_placement" # for image placeholder token inside text - TODO
CAPTION_RESOURCE_KEY = "caption_resource"

class BaseNodeTemplate(BaseModel):
    name: str  = Field("template", description = "template name")
    content_keys: List[str] = Field(
        list(),
        description = "which keys of content dictionary should be formatted into template"
    )
    
class TextNodeTemplate(BaseNodeTemplate):
    content_keys: List[str] = [TEXT_RESOURCE_KEY]
    
    text_template: str = Field(
        "", description = "template for text content"
    )

class ImageNodeTemplate(BaseNodeTemplate):
    content_keys: List[str] = [TEXT_RESOURCE_KEY, CAPTION_RESOURCE_KEY]
    
    text_template: str = Field(
        "",
        description = "template for text content"
    )
    image_placement_token: str = Field(
        "",
        description = "token to use for indicating image placement, has dependency with embedding model"
    )

class TableNodeTemplate(BaseNodeTemplate):
    content_keys: List[str] = [TEXT_RESOURCE_KEY, CAPTION_RESOURCE_KEY]
    table_type: TableType = Field(
        TableType.MARKDOWN, description = "target table type"
    )
    
    text_template: str = Field(
        "",
        description = "template for text content"
    )
    image_placement_token: str = Field(
        "",
        description = "token to use for indicating image placement, has dependency with embedding model"
    )