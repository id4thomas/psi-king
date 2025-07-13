import base64
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

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
from psiking.core.reader.base import BaseReader
from psiking.core.reader.image_utils import crop_image
from psiking.core.reader.schema import ImageDescription

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, ConversionResult
    from docling_core.types.doc import (
        ImageRefMode,
        TextItem,
        PictureItem,
        TableItem,
        GroupItem,
        DoclingDocument,
        RefItem,
        GroupLabel,
        DocItemLabel,
    )
    from docling_core.types.doc.document import ProvenanceItem

# TODO - implement option class
class DoclingPDFReaderOptions(BaseModel):
    pass

# TODO - utilize bbox starting point info
def _convert_bbox_bl_tl(
        bbox: list[float], page_width: int, page_height: int
    ) -> list[float]:
        """Convert bbox from bottom-left to top-left. for usage with crop_image
        Args:
            bbox: t, l, b, r
        """
        x0, y0, x1, y1 = bbox
        return [
            x0 / page_width,
            (page_height - y1) / page_height,
            x1 / page_width,
            (page_height - y0) / page_height,
        ]

def load_json_string(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    if s.strip().startswith("```json") and s.strip().endswith("```"):
        try:
            json_str = '\n'.join(s.strip().split('\n')[1:-1])
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return None

def dump_prov_info(prov: List["ProvenanceItem"]) -> str:
    return json.dumps(
        [x.model_dump() for x in prov]
    )

class DoclingPDFReader(BaseReader):
    """Use Docling to extract document structure and content"""
    
    _dependencies = ["docling"]
    _name="DoclingPDFReader"
    
    def __init__(
        self,
        *args,
        converter: Optional["DocumentConverter"] = None,
        format_options: Optional[Union[dict, "PdfPipelineOptions"]] = None,
        **kwargs,
    ):
        try:
            from docling.datamodel.base_models import InputFormat
        except ImportError:
            raise ImportError("Please install docling: 'pip install docling'")

        if converter is not None:
            self.converter_=converter
        else:
            self.converter_ = self._load_converter(format_options)
        
        # Initialize pipeline
        self.converter_.initialize_pipeline(InputFormat.PDF)
        super().__init__(*args, **kwargs)
    
    def _load_converter(
        self,
        format_options: Optional[Union[dict, "PdfPipelineOptions"]] = None
    ):
        """Initialize DocumentConverter with format_options"""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
        except ImportError:
            raise ImportError("Please install docling: 'pip install docling'")
        
        if format_options is None:
            # default format options
            format_options = PdfPipelineOptions()
            format_options.images_scale = 1.5
            format_options.generate_page_images = True
            format_options.generate_picture_images = True
            
            format_options.do_ocr = False
            format_options.do_table_structure = False
            format_options.do_picture_description = False
        else:
            if isinstance(format_options, dict):
                format_options = PdfPipelineOptions(**format_options)
        
        converter = DocumentConverter(
            allowed_formats = [
                InputFormat.PDF,
            ],
            format_options = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options = format_options,
                    backend = DoclingParseV2DocumentBackend
                )
            }
        )
        return converter
    
    def _convert(self, file_path: str | Path) -> "ConversionResult":
        return self.converter_.convert(file_path, raises_on_error=True)
    
    @classmethod
    def _get_textitem_by_cref(cls, cref: str, document: "DoclingDocument") -> "TextItem":
        item_id = cref.split("/")[-1]
        return document.texts[int(item_id)]
    
    @classmethod
    def _get_pictureitem_by_cref(cls, cref: str, document: "DoclingDocument") -> "PictureItem":
        item_id = cref.split("/")[-1]
        return document.pictures[int(item_id)]

    @classmethod
    def _get_tableitem_by_cref(cls, cref: str, document: "DoclingDocument") -> "TableItem":
        item_id = cref.split("/")[-1]
        return document.tables[int(item_id)]
    
    @classmethod
    def _get_groupitem_by_cref(cls, cref: str, document: "DoclingDocument") -> "GroupItem":
        item_id = cref.split("/")[-1]
        return document.groups[int(item_id)]
    
    # Docling Item -> Node
    @classmethod
    def _textitem_to_node(cls, item: "TextItem") -> Optional[TextNode]:
        from docling_core.types.doc import DocItemLabel
        
        # Filter unwanted texts
        # TODO: make filtering option configurable
        if item.label in [DocItemLabel.FOOTNOTE, DocItemLabel.PAGE_FOOTER]: 
            return None
        
        # Get Text
        text = item.text
        
        # Get Label / Metadata
        if item.label == DocItemLabel.TITLE:
            label = TextLabel.TITLE
        elif item.label == DocItemLabel.PAGE_HEADER:
            label = TextLabel.PAGE_HEADER
        elif item.label == DocItemLabel.SECTION_HEADER:
            label = TextLabel.SECTION_HEADER
        elif item.label == DocItemLabel.LIST_ITEM:
            label = TextLabel.LIST
        elif item.label == DocItemLabel.CODE:
            label = TextLabel.CODE
        elif item.label == DocItemLabel.FORMULA:
            label = TextLabel.EQUATION
        else:
            label = TextLabel.PLAIN

        # TODO: properly apply metadata
        # if len(item.prov)>0:
        #     page_no = item.prov[0].page_no
        # else:
        #     page_no = -1
        metadata = {
            # "page_no": page_no,
            'prov': dump_prov_info(item.prov)
        }
        return TextNode(
            text = text,
            label = label,
            metadata = metadata
        )
    
    @classmethod
    def _imageitem_to_node(cls, item: "PictureItem", document: "DoclingDocument") -> ImageNode:
        # Filter small images
        # TODO: make filtering option configurable
        # maybe utilize coverage ratio? https://github.com/DS4SD/docling/blob/e1436a8b0574e6bb2bb89bd65e98221e418d7142/docling/models/base_ocr_model.py#L32
        image = item.get_image(document)
        if image.width * image.height < 5000:
            return None
        
        uri = str(item.image.uri)
        base64_data = uri.split(",", 1)[1]
        # Decode the Base64 data to bytes
        binary_data = base64.b64decode(base64_data)
        
        # Text
        ## Caption
        caption = item.caption_text(doc=document)
        
        ## Check for text annotations 
        # example:
        # [PictureDescriptionData(kind='description', text='description here', provenance='not-implemented')]
        # text = item.annotations[0].text if item.annotations else ""

        if item.annotations:
            annotation_text = item.annotations[0].text
            annotation_data = load_json_string(annotation_text)
            if annotation_data is None:
                text = annotation_text
            else:
                try:
                    image_desc = ImageDescription.model_validate(annotation_data)
                    text = image_desc.text
                    description = image_desc.description
                    if caption:
                        caption = " ".join([caption, description])
                    else:
                        caption=description
                except Exception as e:
                    print("[WARNING] ImageDescription validation fail {}".format(str(e)))
                    # Use Annotation Text
                    text = annotation_text
        else:
            text = ""
        
        # Check for caption

        # TODO: add metadata
        # if len(item.prov)>0:
        #     page_no = item.prov[0].page_no
        # else:
        #     page_no = -1
        # metadata = {
        #     "page_no": page_no
        # }
        metadata={'prov': dump_prov_info(item.prov)}
        return ImageNode(
            text_resource=MediaResource(text=text),
            caption_resource=MediaResource(text=caption),
            image_resource=MediaResource(data=binary_data, mimetype=item.image.mimetype),
            metadata=metadata
        )
        
    @classmethod
    def _tableitem_to_node(cls, item: "TableItem", document: "DoclingDocument") -> TableNode:
        # text_resource
        html_text = item.export_to_html(doc=document)
        
        # image_resource
        table_img = item.get_image(document)
        buffered = BytesIO()
        table_img.save(buffered, format="PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # TODO: add metadata
        # if len(item.prov)>0:
        #     page_no = item.prov[0].page_no
        # else:
        #     page_no = -1
        # metadata = {
        #     "page_no": page_no
        # }
        metadata={'prov': dump_prov_info(item.prov)}
        return TableNode(
            table_type=TableType.HTML,
            text_resource=MediaResource(text=html_text),
            image_resource=MediaResource(data=base64_data, mimetype="image/png"),
            metadata=metadata
        )
        
    @classmethod
    def _convert_cref_item_to_node(
        cls, cref_item, document: "DoclingDocument"
    ) -> Optional[Union[TextNode, ImageNode, TableNode]]:
        """Only handle TextItem, PictureItem, TableItem"""
        from docling_core.types.doc import TextItem, PictureItem, TableItem, DocItemLabel
        
        if "texts" in cref_item.cref:
            item = cls._get_textitem_by_cref(cref_item.cref, document)
        elif "picture" in cref_item.cref:
            item = cls._get_pictureitem_by_cref(cref_item.cref, document)
        elif "tables" in cref_item.cref:
            item = cls._get_tableitem_by_cref(cref_item.cref, document)
        else:
            raise ValueError(f"Unknown item type: {cref_item.cref}")

        if isinstance(item, TextItem):
            return cls._textitem_to_node(item)
        elif isinstance(item, PictureItem):
            return cls._imageitem_to_node(item, document)
        elif isinstance(item, TableItem):
            return cls._tableitem_to_node(item, document)
        else:
            raise ValueError(f"Unknown item type: {item.cref}")
    
    @classmethod
    def _flatten_groupitem(
        cls, item: "GroupItem", document: "DoclingDocument"
    ) -> List[Union["TextItem", "PictureItem", "TableItem"]]:
        flattened_items = []
        for child in item.children:
            child_cref = child.cref
            if "groups" in child_cref:
                child_group_item = cls._get_groupitem_by_cref(child_cref, document)
                items = cls._flatten_groupitem(child_group_item, document)
                flattened_items.extend(items)
                continue
            
            if "texts" in child_cref:
                item = cls._get_textitem_by_cref(child_cref, document)
            elif "picture" in child_cref:
                item = cls._get_pictureitem_by_cref(child_cref, document)
            elif "tables" in child_cref:
                item = cls._get_tableitem_by_cref(child_cref, document)
            else:
                raise ValueError(f"Unknown item type: {child_cref}")
            flattened_items.append(item)
        return flattened_items
    
    @classmethod
    def _combine_list_text(
        cls, items: List["TextItem"], ordered: bool = False
    ) -> Tuple[str, List["ProvenanceItem"]]:
        """Restore list hierarchy based on bbox.l (left) coordinates."""
    
        def _is_item_inner(last: float, current: float) -> bool:
            """
            Returns True if `current` (the new item's indentation) is 
            strictly greater (i.e., 'further right') than `last`.
            """
            # If dealing with sub-1.0 floats (scaled coordinates),
            # multiply them up so the comparison is still valid.
            if last < 1 and current < 1:
                last   *= 100
                current *= 100
            return current > last
        
        indent_stack = []
        texts        = []
        prov         = []

        for item in items:
            marker   = item.marker
            bbox_obj = item.prov[0].bbox
            l        = bbox_obj.l
            
            prov.extend(item.prov)

            # If the stack is empty, this is the first item at depth 0.
            if not indent_stack:
                indent_stack.append(l)
                depth = 0
                texts.append("\t"*depth + f"{marker} {item.text}")
            else:
                # If this new item is 'further right' than the previous, it goes deeper.
                if _is_item_inner(indent_stack[-1], l):
                    indent_stack.append(l)
                    # depth is length of stack - 1 since we started from zero
                    depth = len(indent_stack) - 1
                    texts.append("\t"*depth + f"{marker} {item.text}")
                else:
                    # Otherwise, we go back out (pop) until it's valid or stack is empty.
                    while indent_stack and not _is_item_inner(indent_stack[-1], l):
                        indent_stack.pop()
                    
                    # Now we are at the correct "outer" level (or at root).
                    indent_stack.append(l)
                    depth = len(indent_stack) - 1
                    texts.append("\t"*depth + f"{marker} {item.text}")

        return "\n".join(texts), prov
    
    @classmethod
    def _groupitem_to_node(cls, item: "GroupItem", document: "DoclingDocument") -> list:
        from docling_core.types.doc import GroupLabel
        
        nodes = []
        
        if item.label == GroupLabel.KEY_VALUE_AREA:
            for child_cref_item in item.children:
                child_cref = child_cref_item.cref
                if "groups" in child_cref:
                    child_group_item = cls._get_groupitem_by_cref(child_cref, document)
                    child_nodes = cls._groupitem_to_node(child_group_item, document)
                    nodes.extend(child_nodes)
                else:
                    node = cls._convert_cref_item_to_node(child_cref_item, document)
                    if node is None:
                        continue
                    nodes.append(node)
        elif item.label == GroupLabel.LIST:
            child_items =  cls._flatten_groupitem(item, document)
            list_text, list_prov = cls._combine_list_text(child_items, ordered=False)
            node = TextNode(
                text=list_text,
                label=TextLabel.LIST,
                metadata={'prov': dump_prov_info(list_prov)}
            )
            nodes.append(node)
        elif item.label == GroupLabel.ORDERED_LIST:
            child_items =  cls._flatten_groupitem(item, document)
            list_text, list_prov = cls._combine_list_text(child_items, ordered=True)
            node = TextNode(
                text=list_text,
                label=TextLabel.LIST,
                metadata={'prov': dump_prov_info(list_prov)}
            )
            nodes.append(node)
        else:
            for child_cref_item in item.children:
                child_cref = child_cref_item.cref
                if "groups" in child_cref:
                    child_group_item = cls._get_groupitem_by_cref(child_cref, document)
                    child_nodes = cls._groupitem_to_node(child_group_item, document)
                    nodes.extend(child_nodes)
                else:
                    node = cls._convert_cref_item_to_node(child_cref_item, document)
                    if node is None:
                        continue
                    nodes.append(node)
        return nodes
    
    def read(
        self,
        file_path: str | Path,
        extra_info: Optional[dict] = None,
    ) -> Document:
        metadata = self.default_metadata
        if extra_info is not None and isinstance(extra_info, dict):
            metadata = metadata | extra_info
        
        # Convert PDF to Docling Document
        result = self._convert(file_path)
        docling_document = result.document
        
        # Map docling Items to Nodes
        nodes = []
        for item in docling_document.body.children:
            if "groups" in item.cref:
                group_item = self._get_groupitem_by_cref(item.cref, docling_document)
                child_nodes = self._groupitem_to_node(group_item, docling_document)
                nodes.extend(child_nodes)
            else:
                node = self._convert_cref_item_to_node(item, docling_document)
                if node is None:
                    continue
                nodes.append(node)
        # TODO: configure metadata
        
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
    