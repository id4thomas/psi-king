import base64
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

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

class DoclingPDFReader(BaseReader):
    """Use Docling to extract document structure and content"""
    
    _dependencies = ["docling"]
    
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
        if len(item.prov)>0:
            page_no = item.prov[0].page_no
        else:
            page_no = -1
        metadata = {
            "page_no": page_no
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
        
        # Check for text annotations 
        # example:
        # [PictureDescriptionData(kind='description', text='description here', provenance='not-implemented')]
        text = item.annotations[0].text if item.annotations else ""
        
        # Check for caption
        caption = item.caption_text(doc=document)

        # TODO: add metadata
        if len(item.prov)>0:
            page_no = item.prov[0].page_no
        else:
            page_no = -1
        metadata = {
            "page_no": page_no
        }
        return ImageNode(
            text_resource=MediaResource(text=text),
            caption_resource=MediaResource(text=caption),
            image_resource=MediaResource(data=binary_data, mimetype=item.image.mimetype),
            metadata=metadata
        )
        
    @classmethod
    def _tableitem_to_node(cls, item: "TableItem", document: "DoclingDocument") -> TableNode:
        # text_resource
        html_text = item.export_to_html()
        
        # image_resource
        table_img = item.get_image(document)
        buffered = BytesIO()
        table_img.save(buffered, format="PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # TODO: add metadata
        if len(item.prov)>0:
            page_no = item.prov[0].page_no
        else:
            page_no = -1
        metadata = {
            "page_no": page_no
        }
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
    def _combine_list_text(cls, items: List["TextItem"], ordered: bool = False) -> str:
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

        for item in items:
            marker   = item.marker
            bbox_obj = item.prov[0].bbox
            l        = bbox_obj.l

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

        return "\n".join(texts)
    
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
            list_text = cls._combine_list_text(child_items, ordered=False)
            node = TextNode(
                text=list_text,
                label=TextLabel.LIST
            )
            nodes.append(node)
        elif item.label == GroupLabel.ORDERED_LIST:
            child_items =  cls._flatten_groupitem(item, document)
            list_text = cls._combine_list_text(child_items, ordered=True)
            node = TextNode(
                text=list_text,
                label=TextLabel.LIST
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
        metadata = extra_info or {}
        
        # Convert PDF to Docling Document
        result = self._convert(file_path)
        docling_document = result.document
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
    
    
    
    
# class DoclingPDFReader(BaseReader):

#     _dependencies = ["docling"]

#     vlm_endpoint: str = Param(
#         help=(
#             "Default VLM endpoint for figure captioning. "
#             "If not provided, will not caption the figures"
#         )
#     )

#     max_figure_to_caption: int = Param(
#         100,
#         help=(
#             "The maximum number of figures to caption. "
#             "The rest will be indexed without captions."
#         ),
#     )

#     figure_friendly_filetypes: list[str] = Param(
#         [".pdf", ".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".heif", ".tif"],
#         help=(
#             "File types that we can reliably open and extract figures. "
#             "For files like .docx or .html, the visual layout may be different "
#             "when viewed from different tools, hence we cannot use Azure DI location "
#             "to extract figures."
#         ),
#     )

#     @Param.auto(cache=True)
#     def converter_(self):
#         try:
#             from docling.document_converter import DocumentConverter
#         except ImportError:
#             raise ImportError("Please install docling: 'pip install docling'")

#         return DocumentConverter()

#     def run(
#         self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
#     ) -> List[Document]:
#         return self.load_data(file_path, extra_info, **kwargs)

#     def load_data(
#         self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
#     ) -> List[Document]:
#         """Extract the input file, allowing multi-modal extraction"""

#         metadata = extra_info or {}

#         result = self.converter_.convert(file_path)
#         result_dict = result.document.export_to_dict()

#         file_path = Path(file_path)
#         file_name = file_path.name

#         # extract the figures
#         figures = []
#         gen_caption_count = 0
#         for figure_obj in result_dict.get("pictures", []):
#             if not self.vlm_endpoint:
#                 continue
#             if file_path.suffix.lower() not in self.figure_friendly_filetypes:
#                 continue

#             # retrieve extractive captions provided by docling
#             caption_refs = [caption["$ref"] for caption in figure_obj["captions"]]
#             extractive_captions = []
#             for caption_ref in caption_refs:
#                 text_id = caption_ref.split("/")[-1]
#                 try:
#                     caption_text = result_dict["texts"][int(text_id)]["text"]
#                     extractive_captions.append(caption_text)
#                 except (ValueError, TypeError, IndexError) as e:
#                     print(e)
#                     continue

#             # read & crop image
#             page_number = figure_obj["prov"][0]["page_no"]

#             try:
#                 page_number_text = str(page_number)
#                 page_width = result_dict["pages"][page_number_text]["size"]["width"]
#                 page_height = result_dict["pages"][page_number_text]["size"]["height"]

#                 bbox_obj = figure_obj["prov"][0]["bbox"]
#                 bbox: list[float] = [
#                     bbox_obj["l"],
#                     bbox_obj["t"],
#                     bbox_obj["r"],
#                     bbox_obj["b"],
#                 ]
#                 if bbox_obj["coord_origin"] == "BOTTOMLEFT":
#                     bbox = self._convert_bbox_bl_tl(bbox, page_width, page_height)

#                 img = crop_image(file_path, bbox, page_number - 1)
#             except KeyError as e:
#                 print(e, list(result_dict["pages"].keys()))
#                 continue

#             # convert img to base64
#             img_bytes = BytesIO()
#             img.save(img_bytes, format="PNG")
#             img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
#             img_base64 = f"data:image/png;base64,{img_base64}"

#             # generate the generative caption
#             if gen_caption_count >= self.max_figure_to_caption:
#                 gen_caption = ""
#             else:
#                 gen_caption_count += 1
#                 gen_caption = generate_single_figure_caption(
#                     img_base64, self.vlm_endpoint
#                 )

#             # join the extractive and generative captions
#             caption = "\n".join(extractive_captions + [gen_caption])

#             # store the image into document
#             figure_metadata = {
#                 "image_origin": img_base64,
#                 "type": "image",
#                 "page_label": page_number,
#                 "file_name": file_name,
#                 "file_path": file_path,
#             }
#             figure_metadata.update(metadata)

#             figures.append(
#                 Document(
#                     text=caption,
#                     metadata=figure_metadata,
#                 )
#             )

#         # extract the tables
#         tables = []
#         for table_obj in result_dict.get("tables", []):
#             # convert the tables into markdown format
#             markdown_table = self._parse_table(table_obj)
#             caption_refs = [caption["$ref"] for caption in table_obj["captions"]]

#             extractive_captions = []
#             for caption_ref in caption_refs:
#                 text_id = caption_ref.split("/")[-1]
#                 try:
#                     caption_text = result_dict["texts"][int(text_id)]["text"]
#                     extractive_captions.append(caption_text)
#                 except (ValueError, TypeError, IndexError) as e:
#                     print(e)
#                     continue
#             # join the extractive and generative captions
#             caption = "\n".join(extractive_captions)
#             markdown_table = f"{caption}\n{markdown_table}"

#             page_number = table_obj["prov"][0].get("page_no", 1)

#             table_metadata = {
#                 "type": "table",
#                 "page_label": page_number,
#                 "table_origin": markdown_table,
#                 "file_name": file_name,
#                 "file_path": file_path,
#             }
#             table_metadata.update(metadata)

#             tables.append(
#                 Document(
#                     text=markdown_table,
#                     metadata=table_metadata,
#                 )
#             )

#         # join plain text elements
#         texts = []
#         page_number_to_text = defaultdict(list)

#         for text_obj in result_dict["texts"]:
#             page_number = text_obj["prov"][0].get("page_no", 1)
#             page_number_to_text[page_number].append(text_obj["text"])

#         for page_number, txts in page_number_to_text.items():
#             texts.append(
#                 Document(
#                     text="\n".join(txts),
#                     metadata={
#                         "page_label": page_number,
#                         "file_name": file_name,
#                         "file_path": file_path,
#                         **metadata,
#                     },
#                 )
#             )

#         return texts + tables + figures

#     def _convert_bbox_bl_tl(
#         self, bbox: list[float], page_width: int, page_height: int
#     ) -> list[float]:
#         """Convert bbox from bottom-left to top-left"""
#         x0, y0, x1, y1 = bbox
#         return [
#             x0 / page_width,
#             (page_height - y1) / page_height,
#             x1 / page_width,
#             (page_height - y0) / page_height,
#         ]

#     def _parse_table(self, table_obj: dict) -> str:
#         """Convert docling table object to markdown table"""
#         table_as_list: List[List[str]] = []
#         grid = table_obj["data"]["grid"]
#         for row in grid:
#             table_as_list.append([])
#             for cell in row:
#                 table_as_list[-1].append(cell["text"])

#         return make_markdown_table(table_as_list)
