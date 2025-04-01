import copy
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Extra
from tqdm import tqdm

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableStructureOptions,
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from psiking.core.reader.pdf.docling_reader import DoclingPDFReader
from psiking.core.reader import PDF2ImageReader

from .docling_vllm_picture_description_pipeline import (
    VLLMPictureDescriptionApiOptions,
    VLLMPictureDescriptionPdfPipeline
)

DESCRIPTION_INSTRUCTION = '''주어진 이미지에대해 2가지 정보를 반환합니다.
* description: 최대 2문장 정도로 이미지에 대한 간결한 설명
* text: 이미지 내에서 인식된 모든 텍스트
다음 JSON 형식으로 반환하세요 {"description": str, "text": str}'''

class ImageDescription(BaseModel):
    description: str
    text: str
    
    class Config:
        extra=Extra.forbid


class ReaderModule:
    def __init__(self, settings):
        self.settings = settings
        self.docling_reader = self._load_docling_reader()
        self.pdf2img_reader = self._load_pdf2img_reader
    
    def _load_converter(self):
        format_options = PdfPipelineOptions()
        format_options.accelerator_options = AcceleratorOptions(device="mps")

        format_options.images_scale = 1.5
        format_options.generate_page_images = True
        format_options.generate_picture_images = True
        format_options.do_ocr = False
        
        # Image Description
        image_description_output_schema = ImageDescription.model_json_schema()
        image_description_output_schema["name"]="description"
        image_description_options = VLLMPictureDescriptionApiOptions(
            url=f"{self.settings.vlm_base_url}/v1/chat/completions",
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.vlm_api_key}"  
            },
            params=dict(
                model=self.settings.vlm_model,
                seed=42,
                max_completion_tokens=512,
                temperature=0.9,
                # For OpenAI
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "description",
                        "strict": True,
                        "schema": ImageDescription.model_json_schema()
                    }
                }
                # For VLLM Structured Gen
                # extra_body={"guided_json": ImageDescription.model_json_schema()}
            ),
            prompt=DESCRIPTION_INSTRUCTION,
            batch_size=6, # Not implemented inside
            scale=0.9,
            timeout=90,
            min_coverage_area_threshold=0.01,
            bitmap_area_threshold=0.1 # 10% of page area
        )
        format_options.do_picture_description = True
        # format_options.do_picture_description = False
        format_options.enable_remote_services = True
        format_options.picture_description_options = image_description_options

        # TableStructure
        format_options.do_table_structure = True
        format_options.table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)

        # Initialize Converter
        converter = DocumentConverter(
            allowed_formats = [
                InputFormat.PDF,
            ],
            format_options = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VLLMPictureDescriptionPdfPipeline,
                    pipeline_options = format_options,
                    backend = DoclingParseV2DocumentBackend
                )
            }
        )
        return converter
    
    def _load_docling_reader(self):
        converter = self._load_converter()
        docling_reader = DoclingPDFReader(converter=converter)
        return docling_reader
    
    def _load_pdf2img_reader(self):
        # testing on macOS, provide poppler path manually
        pdf2img_reader = PDF2ImageReader(poppler_path=self.settings.poppler_path)
        return pdf2img_reader

    def run(self, file_paths: List[str], source_id_prefix="", extra_infos: Dict[str,str]={}):
        documents = []
        docling_failed_fnames = []
        pdf2img_failed_fnames = []
        for doc_i, file_path in tqdm(enumerate(file_paths)):
            fname = file_path.rsplit("/",1)[-1]
            
            doc_extra_info = copy.deepcopy(
                extra_infos.get(fname, dict())
            )
            doc_extra_info["source_id"] = f"{source_id_prefix}/{doc_i}"
            doc_extra_info["source_file"] = fname
            
            try:
                document = self.docling_reader.run(
                    file_path,
                    extra_info=doc_extra_info
                )
                documents.append(document)
                continue
            except Exception as e:
                print("[DOCLING READER] failed {} - {}".format(fname, str(e)))
                docling_failed_fnames.append(fname)
            
            try:
                document = self.pdf2img_reader.run(
                    file_path,
                    extra_info=doc_extra_info
                )
                documents.append(document)
            except Exception as e:
                print("[PDF2IMG READER] failed {} - {}".format(fname, str(e)))
                pdf2img_failed_fnames.append(fname)
                
        return documents