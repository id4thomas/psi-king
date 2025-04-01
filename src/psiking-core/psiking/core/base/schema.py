"""Base schema for data structures. Mostly follows llama-index schema"""

import base64
from enum import Enum, auto
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from PIL import Image
import json
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)
import uuid

import filetype
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    FilePath,
    field_validator,
    PlainSerializer,
    ValidationInfo,
    root_validator
)
import requests

EnumNameSerializer = PlainSerializer(
    lambda e: e.value, return_type="str", when_used="always"
)

class ObjectType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    DOCUMENT = "document"
    
    def __str__(self):
        """Get string value."""
        return str(self.value)

class TextType(str, Enum):
    """Type of the text"""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    TEX = "tex"
    
    def __str__(self):
        """Get string value."""
        return str(self.value)

class TextLabel(str, Enum):
    """Label of the text"""
    PLAIN = "plain"
    TITLE = "title"
    PAGE_HEADER = "page_header"
    SECTION_HEADER = "section_header"
    LIST = "list"
    CODE = "code"
    LINK = "link"
    EQUATION = "equation"
    
    def __str__(self):
        """Get string value."""
        return str(self.value)

class TableType(str, Enum):
    """Type of the table"""
    UNKNOWN = "unknown"
    MARKDOWN = "markdown"
    XML = "xml"
    HTML = "html"
    
    def __str__(self):
        """Get string value."""
        return str(self.value)
    
class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"    
    
    def __str__(self):
        """Get string value."""
        return str(self.value)

class BaseNode(BaseModel):
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields"
    )
    
    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name, used as a unique ID in serialization.

        This provides a key that makes serialization robust against actual class
        name changes.
        """
        return "base_node"
    
    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)
    
    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model_dump(**kwargs)
    
    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)
    
class MediaResource(BaseModel):
    """A container class for media content.

    This class represents a generic media resource that can be stored and accessed
    in multiple ways - as raw bytes, on the filesystem, or via URL.

    Attributes:
        text: Plain text representation of this resource
        data: Raw binary data of the media content
        path: Local filesystem path where the media content can be accessed
        url: URL where the media content can be accessed remotely
    """
    data: bytes | None = Field(
        default=None,
        exclude=True,
        description="base64 binary representation of this resource.",
    )
    text: str | None = Field(
        default=None, description="Text representation of this resource."
    )
    path: Path | None = Field(
        default=None, description="Filesystem path of this resource."
    )
    url: AnyUrl | None = Field(default=None, description="URL to reach this resource.")
    mimetype: str | None = Field(
        default=None, description="MIME type of this resource."
    )

    model_config = {
        # This ensures validation runs even for None values
        "validate_default": True
    }
    
    # def __str__(self) -> str:
    #     return f"MediaResource(data={self.data[:10]}, text={self.text[:10]}, path={self.path}, url={self.url})"
    
    @field_validator("data", mode="after")
    @classmethod
    def validate_data(cls, v: bytes | None, info: ValidationInfo) -> bytes | None:
        """If binary data was passed, store the resource as base64 and guess the mimetype when possible.

        In case the model was built passing binary data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if v is None:
            return v

        try:
            # Check if data is already base64 encoded.
            # b64decode() can succeed on random binary data, we make
            # a full roundtrip to make sure it's not a false positive
            decoded = base64.b64decode(v)
            encoded = base64.b64encode(decoded)
            if encoded != v:
                # Roundtrip failed, this is a false positive, return encoded
                return base64.b64encode(v)
        except Exception:
            # b64decode failed, return encoded
            return base64.b64encode(v)

        # Good as is, return unchanged
        return v
    
    @property
    def hash(self) -> str:
        """Generate a hash to uniquely identify the media resource.

        The hash is generated based on the available content (data, path, text or url).
        Returns an empty string if no content is available.
        """
        bits: list[str] = []
        if self.text is not None:
            bits.append(self.text)
        if self.data is not None:
            # Hash the binary data if available
            bits.append(str(sha256(self.data).hexdigest()))
        if self.path is not None:
            # Hash the file path if provided
            bits.append(str(sha256(str(self.path).encode("utf-8")).hexdigest()))
        if self.url is not None:
            # Use the URL string as basis for hash
            bits.append(str(sha256(str(self.url).encode("utf-8")).hexdigest()))

        doc_identity = "".join(bits)
        if not doc_identity:
            return ""
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the MediaResource, including binary data as base64."""
        data = self.dict()
        if self.data is not None:
            data["data"] = base64.b64encode(self.data).decode("utf-8")
        if "path" in data and data["path"] is not None:
            data["path"] = str(Path(data["path"]))
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaResource":
        """Deserialize MediaResource, including decoding base64 data."""
        if "data" in data and data["data"] is not None:
            data["data"] = base64.b64decode(data["data"])
        return cls.parse_obj(data)

class TextNode(BaseNode):
    text_type: TextType = Field(default=TextType.PLAIN, description="Type of the text")
    label: TextLabel = Field(default=TextLabel.PLAIN, description="Label of the text")
    resource: MediaResource = Field(default_factory=MediaResource, description="Media resource for this node.")
    
    @root_validator(pre=True)
    def set_resource(cls, values):
        tr = None
        if 'resource' in values:
            tr = values.pop('resource')
            if isinstance(tr, MediaResource):
                values['resource'] = tr
            elif isinstance(tr, str):
                values['resource'] = MediaResource(text=tr)
            elif isinstance(tr, dict) and 'text' in tr:
                values['resource'] = MediaResource(text=tr['text'])
            else:
                raise ValueError("Invalid 'resource' format, should be a string or a MediaResource")

        if 'text' in values:
            if not tr is None:
                raise ValueError("'resource' and 'text' cannot be provided together.")
            text = values.pop('text')
            if isinstance(text, str):
                values['resource'] = MediaResource(text=text)
            else:
                raise ValueError("Invalid 'text' format, should be a string.")
        return values
        
    @property
    def text(self) -> str:
        if self.resource.text is None:
            raise ValueError("Text is not set for this node.")
        return self.resource.text
    
    @classmethod
    def class_name(cls) -> str:
        return "TextNode"
    
    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TEXT
    
    def get_text_type(self) -> str:
        """Get text type."""
        return self.text_type
    
    def get_text_label(self) -> str:
        """Get text label."""
        return self.label
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["resource"] = self.resource.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextNode":
        data["resource"] = MediaResource.from_dict(data["resource"])
        return cls.parse_obj(data)
    
class ImageNode(BaseNode):
    image_loaded: bool = Field(default=False, description="Whether the image data has been loaded.")
    image_resource: MediaResource = Field(default_factory=MediaResource, description="Media resource for this node.")
    text_resource: MediaResource = Field(
        default_factory=MediaResource, description="Text resource for this node. (ex. ocr result, classification result)"
    )
    caption_resource: MediaResource = Field(default_factory=MediaResource, description="Caption for this image")
    
    @root_validator(pre=True)
    def set_resource(cls, values):
        if "image_resource" in values:
            ir = values.pop("image_resource")
            if isinstance(ir, MediaResource):
                values["image_resource"] = ir
                if ir.data is not None:
                    values["image_loaded"] = True
            elif isinstance(ir, Image.Image):
                values["image_resource"] = MediaResource(
                    data=base64.b64encode(ir.tobytes()),
                    mimetype=ir.format
                )
                values["image_loaded"] = True
            else:
                raise ValueError("Invalid 'image_resource' format, should be a MediaResource or a PIL.Image")
            
        if "text_resource" in values:
            tr = values.pop("text_resource")
            if isinstance(tr, MediaResource):
                values["text_resource"] = tr
            elif isinstance(tr, str):
                values["text_resource"] = MediaResource(text=tr)
            else:
                raise ValueError("Invalid 'text_resource' format, should be a string or a MediaResource")
        if "caption_resource" in values:
            cr = values.pop("caption_resource")
            if isinstance(cr, MediaResource):
                values["caption_resource"] = cr
            elif isinstance(cr, str):
                values["caption_resource"] = MediaResource(text=cr)
            else:
                raise ValueError("Invalid 'caption_resource' format, should be a string or a MediaResource")
        return values
    
    @property
    def image_data(self) -> bytes:
        return self.image_resource.data
    
    @property
    def image(self) -> Image.Image:
        if not self.image_loaded:
            raise ValueError("Image data is not loaded, load with load_image_data()")
        return Image.open(BytesIO(base64.b64decode(self.image_resource.data)))
    
    @property
    def text(self) -> str:
        if self.text_resource.text is None:
            return ""
        return self.text_resource.text
    
    @property
    def caption(self) -> str:
        if self.caption_resource.text is None:
            return ""
        return self.caption_resource.text
        
    @classmethod
    def class_name(cls) -> str:
        return "ImageNode"
    
    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.IMAGE
    
    def load_image_data(self) -> None:
        """Load the image from path or URL and store it as base64 data."""
        if self.image_loaded or self.image_resource.data:
            binary_data = base64.b64decode(self.image_resource.data)
        elif self.image_resource.path:
            with open(self.image_resource.path, 'rb') as f:
                binary_data = f.read()
            self.image_resource.data = base64.b64encode(binary_data)
            self.image_loaded=True
        elif self.image_resource.url:
            response = requests.get(self.image_resource.url)
            response.raise_for_status()
            binary_data = response.content
            self.image_resource.data = base64.b64encode(binary_data)
            self.image_loaded=True
        else:
            raise ValueError("No image path or URL provided")

        image = Image.open(BytesIO(binary_data))
        # Guess mimetype from image format if not already set
        if not self.image_resource.mimetype:
            self.image_resource.mimetype = f"image/{image.format.lower()}" if image.format else "image/png"
            
    def unload_image_data(self) -> None:
        self.image_resource.data = None
        self.image_loaded = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["image_resource"] = self.image_resource.to_dict()
        data["text_resource"] = self.text_resource.to_dict()
        data["caption_resource"] = self.caption_resource.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageNode":
        data["image_resource"] = MediaResource.from_dict(data["image_resource"])
        data["text_resource"] = MediaResource.from_dict(data["text_resource"])
        data["caption_resource"] = MediaResource.from_dict(data["caption_resource"])
        return cls.parse_obj(data)

class TableNode(BaseNode):
    """Node for storing table data. Stores table image & text"""
    image_loaded: bool = Field(default=False, description="Whether the image data has been loaded.")
    table_type: TableType = Field(default=TableType.UNKNOWN, description="Type of the table")
    image_resource: MediaResource = Field(default_factory=MediaResource, description="Image of the table.")
    text_resource: MediaResource = Field(default_factory=MediaResource, description="Text of the  table. (ex. markdown/html version)")
    caption_resource: MediaResource = Field(default_factory=MediaResource, description="Caption for this table")
    
    @root_validator(pre=True)
    def set_resource(cls, values):
        if "image_resource" in values:
            ir = values.pop("image_resource")
            if isinstance(ir, MediaResource):
                values["image_resource"] = ir
                if ir.data is not None:
                    values["image_loaded"] = True
            elif isinstance(ir, Image.Image):
                values["image_resource"] = MediaResource(
                    data=base64.b64encode(ir.tobytes()),
                    mimetype=ir.format
                )
                values["image_loaded"] = True
            else:
                raise ValueError("Invalid 'image_resource' format, should be a MediaResource or a PIL.Image")
            
        if "text_resource" in values:
            tr = values.pop("text_resource")
            if isinstance(tr, MediaResource):
                values["text_resource"] = tr
            elif isinstance(tr, str):
                values["text_resource"] = MediaResource(text=tr)
            else:
                raise ValueError("Invalid 'text_resource' format, should be a string or a MediaResource")
        if "caption_resource" in values:
            cr = values.pop("caption_resource")
            if isinstance(cr, MediaResource):
                values["caption_resource"] = cr
            elif isinstance(cr, str):
                values["caption_resource"] = MediaResource(text=cr)
            else:
                raise ValueError("Invalid 'caption_resource' format, should be a string or a MediaResource")
        return values
    
    @property
    def image_data(self) -> bytes:
        return self.image_resource.data
    
    @property
    def image(self) -> Image.Image:
        if not self.image_loaded:
            raise ValueError("Image data is not loaded, load with load_image_data()")
        return Image.open(BytesIO(base64.b64decode(self.image_resource.data)))
    
    @property
    def text(self) -> str:
        if self.text_resource.text is None:
            return ""
        return self.text_resource.text
    
    @property
    def caption(self) -> str:
        if self.caption_resource.text is None:
            return ""
        return self.caption_resource.text
    
    @classmethod
    def class_name(cls) -> str:
        return "TableNode"
    
    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TABLE
    
    def get_table_type(self) -> str:
        """Get table type."""
        return self.table_type
    
    def load_image_data(self) -> None:
        """Load the image from path or URL and store it as base64 data."""
        if self.image_loaded or self.image_resource.data:
            binary_data = base64.b64decode(self.image_resource.data)
        elif self.image_resource.path:
            with open(self.image_resource.path, 'rb') as f:
                binary_data = f.read()
            self.image_resource.data = base64.b64encode(binary_data)
            self.image_loaded=True
        elif self.image_resource.url:
            response = requests.get(self.image_resource.url)
            response.raise_for_status()
            binary_data = response.content
            self.image_resource.data = base64.b64encode(binary_data)
            self.image_loaded=True
        else:
            raise ValueError("No image path or URL provided")

        image = Image.open(BytesIO(binary_data))
        # Guess mimetype from image format if not already set
        if not self.image_resource.mimetype:
            self.image_resource.mimetype = f"image/{image.format.lower()}" if image.format else "image/png"
            
    def unload_image_data(self) -> None:
        self.image_resource.data = None
        self.image_loaded = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["image_resource"] = self.image_resource.to_dict()
        data["text_resource"] = self.text_resource.to_dict()
        data["caption_resource"] = self.caption_resource.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageNode":
        data["image_resource"] = MediaResource.from_dict(data["image_resource"])
        data["text_resource"] = MediaResource.from_dict(data["text_resource"])
        data["caption_resource"] = MediaResource.from_dict(data["caption_resource"])
        return cls.parse_obj(data)

class Document(BaseNode):
    """Generic container around nodes"""
    nodes: List[BaseNode] = Field(default_factory=list, description="List of nodes in this document.")
    
    @classmethod
    def class_name(cls) -> str:
        return "Document"
    
    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.DOCUMENT

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["nodes"] = [node.to_dict() for node in self.nodes]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        ## Load nodes from data
        node_classes = {"TextNode": TextNode, "ImageNode": ImageNode, "TableNode": TableNode}
        for i, node_data in enumerate(data.get("nodes", [])):
            node_type = node_data.get("class_name")
            if node_type and node_type in node_classes:
                data["nodes"][i] = node_classes[node_type].from_dict(node_data)
        return cls.parse_obj(data)
    

# https://github.com/run-llama/llama_index/blob/0d09a613d552410b0970894fd1bf549c49f41f40/llama-index-core/llama_index/core/storage/docstore/utils.py#L14C1-L43C19
from .constants import TYPE_KEY, DATA_KEY
def doc_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.to_dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_doc(doc_dict: dict) -> BaseNode:
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode

    if doc_type == Document.get_type():
        doc = Document.from_dict(data_dict)
    elif doc_type == TextNode.get_type():
        doc = TextNode.from_dict(data_dict)
    elif doc_type == ImageNode.get_type():
        doc = ImageNode.from_dict(data_dict)
    elif doc_type == TableNode.get_type():
        doc = TableNode.from_dict(data_dict)
    else:
        raise ValueError(f"Unknown doc type: {doc_type}")

    return doc