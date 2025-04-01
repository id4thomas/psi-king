"""Custom PdfPipeline to support image filtering, multithreading batch inference"""

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional
from pathlib import Path
from PIL import Image

from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.picture_description_api_model import PictureDescriptionApiModel

from docling_core.types.doc import PictureItem
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureItem,
)
from docling_core.types.doc.document import (  # TODO: move import to docling_core.types.doc
    PictureDescriptionData,
)
from docling.models.base_model import (
    BaseItemAndImageEnrichmentModel,
    ItemAndImageEnrichmentElement,
)

class VLLMPictureDescriptionApiOptions(PictureDescriptionApiOptions):
    # https://github.com/DS4SD/docling/blob/d8a81c31686449a0bd3a56c0bc8475fead658ba9/docling/datamodel/pipeline_options.py#L212
    min_coverage_area_threshold: float = 0.01 # at least 1% of total page size

class VLLMPictureDescriptionApiModel(PictureDescriptionApiModel):
    """Add page coverage ratio check to prevent VLM (ex. qwen2-vl) errors with very small images
    * https://github.com/vllm-project/vllm/issues/13655
    * `ValueError: height:19 or width:500 must be larger than factor:28`
    """
    def _batch_annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        def annotate_single(image: Image.Image) -> str:
            # Call the existing _annotate_images with a single-image list and retrieve the first result.
            return next(self._annotate_images([image]))
        
        # Use a thread pool with 4 workers to process the images concurrently.
        with ThreadPoolExecutor(max_workers=self.options.batch_size) as executor:
            results = list(executor.map(annotate_single, images))
        return results
    
    # https://github.com/DS4SD/docling/blob/d8a81c31686449a0bd3a56c0bc8475fead658ba9/docling/models/picture_description_base_model.py#L41
    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    )-> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Image.Image] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            assert isinstance(el.item, PictureItem)
            # Check Image page coverage
            page_no = el.item.prov[0].page_no
            page_width = doc.pages[page_no].size.width
            page_height = doc.pages[page_no].size.height
            if page_width and page_height:
                coverage = (el.image.size[0]*el.image.size[1])/(page_width*page_height)
                if coverage<self.options.min_coverage_area_threshold:
                    continue
            elements.append(el.item)
            images.append(el.image)
        print("NUM IMAGES TO ANNOTATE", len(images))
        # outputs = self._annotate_images(images)
        outputs = self._batch_annotate_images(images)

        for item, output in zip(elements, outputs):
            item.annotations.append(
                PictureDescriptionData(text=output, provenance=self.provenance)
            )
            yield item

class VLLMPictureDescriptionPdfPipeline(StandardPdfPipeline):
    # https://github.com/DS4SD/docling/blob/d8a81c31686449a0bd3a56c0bc8475fead658ba9/docling/pipeline/standard_pdf_pipeline.py#L53
    def get_picture_description_model(
        self, artifacts_path: Optional[Path] = None
    ):
        return VLLMPictureDescriptionApiModel(
            enabled=self.pipeline_options.do_picture_description,
            enable_remote_services=self.pipeline_options.enable_remote_services,
            artifacts_path=None,
            options=self.pipeline_options.picture_description_options,
            accelerator_options=None
        )