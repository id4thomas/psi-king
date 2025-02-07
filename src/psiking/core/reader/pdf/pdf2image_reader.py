import base64
from collections import defaultdict
from pathlib import Path, PurePath
from io import BytesIO
from typing import List, Optional, Union, TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel

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
from core.reader.base import BaseReader
from core.reader.image_utils import crop_image

if TYPE_CHECKING:
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError
    )

# TODO - implement option class
class PDF2ImageReaderOptions(BaseModel):
    pass
    
class PDF2ImageReader(BaseReader):
    """
    Use pdf2image to convert pdf pages to images
    * https://github.com/Belval/pdf2image?tab=readme-ov-file
    """
    
    _dependencies = ["pdf2image"]
    
    def __init__(
        self,
        *args,
        poppler_path: Union[str, PurePath] = None,
        **kwargs,
    ):
        self.poppler_path = poppler_path
        super().__init__(*args, **kwargs)
        
    
    def _convert_pages(self, file_path: str | Path) -> List[Image.Image]:
        """convert pdf pages into images
        pdf2image function params (with default values)
        convert_from_path(
            pdf_path,
            dpi=200,
            output_folder=None,
            first_page=None,
            last_page=None,
            fmt='ppm',
            jpegopt=None,
            thread_count=1,
            userpw=None,
            use_cropbox=False,
            strict=False,
            transparent=False,
            single_file=False,
            output_file=str(uuid.uuid4()),
            poppler_path=None,
            grayscale=False,
            size=None,
            paths_only=False,
            use_pdftocairo=False,
            timeout=600,
            hide_attributes=False
        )
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("Please install pdf2image: 'pip install pdf2image'")
        
        pages: List[Image.Image] = convert_from_path(
            pdf_path=file_path,
            dpi=200,
            poppler_path=self.poppler_path
        )
        return pages
        
    @classmethod
    def _image_to_node(cls, image: Image.Image, metadata: Optional[dict] = None) -> ImageNode:
        """Convert a PIL Image to an ImageNode"""
        # Infer MIME type
        mimetype = f"image/{image.format.lower()}" if image.format else "image/png"

        # Convert Image to Base64
        buffered = BytesIO()
        image.save(buffered, format=image.format or "PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        image_resource = MediaResource(
            data=base64_data.encode("utf-8"),
            mimetype=mimetype
        )
        # TODO: add metadata
        metadata = metadata or {}
        return ImageNode(
            image_resource=image_resource,
            metadata=metadata
        )
    
    def read(
        self,
        file_path: str | Path,
        extra_info: Optional[dict] = None,
    ) -> Document:
        metadata = extra_info or {}
        
        # Convert pdf -> image
        page_images = self._convert_pages(file_path)
        
        # Convert to Nodes
        nodes = []
        for page_i, page_image in enumerate(page_images):
            node = self._image_to_node(
                page_image,
                metadata={"page": page_i+1}
            )
            nodes.append(node)
        
        # Create Document
        document = Document(
            nodes=nodes,
            metadata=metadata
        )
        return document
        
        
    def run(
        self,
        file_path: str | Path,
        extra_info: Optional[dict] = None,
        **kwargs
    ) -> Document:
        return self.read(file_path, extra_info, **kwargs)