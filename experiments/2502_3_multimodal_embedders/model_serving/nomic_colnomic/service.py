import asyncio
import base64
from enum import Enum
from io import BytesIO
import os
import logging
from typing import Annotated, List, Optional, cast

import bentoml
import numpy as np
import torch
from annotated_types import MinLen
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.utils.torch_utils import get_torch_device
from PIL import Image
from pydantic import BaseModel

class ImageDetail(Enum):
    """
    Supported image details for `ImagePayload`.
    """

    LOW = "low"
    HIGH = "high"
    AUTO = "auto"


class ImagePayload(BaseModel):
    """
    Image payload for `ColPaliService`.

    Args:
        url: The URL of the image or a base 64 encoded image.
        detail: The detail of the image. Defaults to `ImageURLDetail.AUTO`.

    NOTE: `detail` is unused for now and only here to be ISO with the OpenAI API.
    """

    url: str
    text: Optional[str] = None
    detail: ImageDetail = ImageDetail.AUTO

# with bentoml.importing():
#     from .interfaces import ImagePayload
#     from .utils import convert_b64_to_pil_image, is_url

def is_url(val: str) -> bool:
    return val.startswith("http")

def convert_b64_to_pil_image(b64_image: str) -> Image.Image:
    """
    Convert a base64 image string to a PIL Image.
    """
    if b64_image.startswith("data:image"):
        b64_image = b64_image.split(",")[1]

    try:
        image_data = base64.b64decode(b64_image)
        image_bytes = BytesIO(image_data)
        image = Image.open(image_bytes)
    except Exception as e:
        raise ValueError("Failed to convert base64 string to PIL Image") from e

    return image


def convert_pil_to_b64_image(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 string.
    """
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

logger = logging.getLogger("bentoml")

class ColNomicProcessor(ColQwen2_5_Processor):
    def __init__(
        self,
        *args,
        max_num_visual_tokens: int = 768,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

        self.max_num_visual_tokens = max_num_visual_tokens
        self.factor = 28
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = self.max_num_visual_tokens * 28 * 28

        self.image_processor.min_pixels = self.min_pixels
        self.image_processor.max_pixels = self.max_pixels

# def create_model_pipeline(pretrained_path: str, adapter_path: str) -> ColQwen2_5:
#     model = ColQwen2_5.from_pretrained(
#         pretrained_path,
#         torch_dtype=torch.bfloat16,
#         device_map="mps",
#         attn_implementation="eager",
#         local_files_only=True,
#         low_cpu_mem_usage=True,
#     ).eval()
#     model.load_adapter(adapter_path)
#     return model

def create_model_pipeline(path: str) -> ColQwen2_5:
    model = ColQwen2_5.from_pretrained(
        os.path.join(path, "pretrained"),
        torch_dtype=torch.bfloat16,
        device_map="mps",
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    model.load_adapter(os.path.join(path, "adapter"))
    return model

    # return cast(
    #     ColQwen2_5,
    #     ColQwen2_5.from_pretrained(
    #         pretrained_model_name_or_path=path,
    #         torch_dtype=torch.bfloat16,
    #         device_map=get_torch_device("auto"),
    #         local_files_only=True,
    #         low_cpu_mem_usage=True,
    #     ),
    # ).eval()


def create_processor_pipeline(path: str) -> ColNomicProcessor:
    return cast(
        ColNomicProcessor,
        ColNomicProcessor.from_pretrained(
            pretrained_model_name_or_path=os.path.join(path, "adapter"),
            local_files_only=True,
        ),
    )


@bentoml.service(
    name="colnomic",
    workers=1,
    traffic={"concurrency": 64},
)
class ColNomicService:
    """
    ColPali service for embedding images and queries, and scoring them.
    Provides batch processing capabilities.

    NOTE: You need to build the model using `bentoml.models.create(name="colpali_model")` before using this service.
    """

    _model_ref: bentoml.Model = bentoml.models.get("colnomic-embed-multimodal-3b")

    def __init__(self) -> None:
        # pretrained_path = self._model_ref.path  # Path to the pretrained model
        # adapter_path = self._model_ref.metadata.get("adapter_path")
        # # adapter_path = self._model_ref.custom_fields.get("adapter_path")
        # self.model: ColQwen2_5 = create_model_pipeline(
        #     pretrained_path=pretrained_path,
        #     adapter_path=adapter_path
        # )
        print(self._model_ref.path)
        print(os.listdir(self._model_ref.path))
        self.model: ColQwen2_5 = create_model_pipeline(path=self._model_ref.path)
        self.processor: ColNomicProcessor = create_processor_pipeline(path=self._model_ref.path)
        logger.info(f"ColNomic loaded on device: {self.model.device}")

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=64,
        max_latency_ms=30_000,
    )
    async def embed_images(
        self,
        items: List[ImagePayload],
    ) -> np.ndarray:
        """
        Generate image embeddings of shape (batch_size, sequence_length, embedding_dim).
        """

        images: List[Image.Image] = []
        texts: List[str] = []

        for item in items:
            if is_url(item.url):
                raise NotImplementedError("URLs are not supported.")
            images += [convert_b64_to_pil_image(item.url)]
            
            if item.text is None:
                text = self.processor.visual_prompt_prefix
            else:
                text = item.text
            texts += [text]
            
        batch_images = self.processor.process_images(
            images,
            context_prompts=texts
        ).to(self.model.device)

        with torch.inference_mode():
            image_embeddings = self.model(**batch_images)

        return image_embeddings.cpu().to(torch.float32).detach().numpy()

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=64,
        max_latency_ms=30_000,
    )
    async def embed_queries(
        self,
        items: List[str],
    ) -> np.ndarray:
        """
        Generate query embeddings of shape (batch_size, sequence_length, embedding_dim).
        """
        batch_queries = self.processor.process_queries(items).to(self.model.device)

        with torch.inference_mode():
            query_embeddings = self.model(**batch_queries)

        return query_embeddings.cpu().to(torch.float32).detach().numpy()

    @bentoml.api
    async def score_embeddings(
        self,
        image_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the late-interaction/MaxSim scores of shape (num_queries, num_images).

        Args:
            image_embeddings: The image embeddings of shape (num_images, sequence_length, embedding_dim).
            query_embeddings: The query embeddings of shape (num_queries, sequence_length, embedding_dim).
        """

        image_embeddings_torch: List[torch.Tensor] = [torch.Tensor(x) for x in image_embeddings]
        query_embeddings_torch: List[torch.Tensor] = [torch.Tensor(x) for x in query_embeddings]

        return (
            self.processor.score(
                qs=query_embeddings_torch,
                ps=image_embeddings_torch,
            )
            .cpu()
            .to(torch.float32)
            .numpy()
        )

    @bentoml.api
    async def score(
        self,
        images: Annotated[List[ImagePayload], MinLen(1)],
        queries: Annotated[List[str], MinLen(1)],
    ) -> np.ndarray:
        """
        Returns the late-interaction/MaxSim scores of the queries against the images.
        """

        image_embeddings, query_embeddings = await asyncio.gather(
            self.embed_images(images),
            self.embed_queries(queries),
        )

        return await self.score_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
        )