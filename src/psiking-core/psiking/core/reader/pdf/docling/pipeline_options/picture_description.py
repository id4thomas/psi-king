from docling.datamodel.pipeline_options import (
    PictureDescriptionApiOptions,
)

from psiking.core.reader.schema import ImageDescription

# Picture Annotation (picture_description_options) using VLM
# https://docling-project.github.io/docling/examples/pictures_description_api/

# "Describe the image in three sentences. Be consise and accurate.",
DESCRIPTION_INSTRUCTION = '''주어진 이미지에대해 2가지 정보를 반환합니다.
* description: 최대 2문장 정도로 이미지에 대한 간결한 설명
* text: 이미지 내에서 인식된 모든 텍스트
다음 JSON 형식으로 반환하세요 {"description": str, "text": str}'''

def vllm_online_options(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 180.0
):
    options = PictureDescriptionApiOptions(
        url=f"{base_url}/v1/chat/completions",
        headers = {
            'Authorization': f'Bearer {api_key}'
        },
        # TODO - remove fixed params
        params=dict(
            model=model,
            seed=42,
            max_completion_tokens=8192,
            temperature=0.9,
            extra_body={"guided_json": ImageDescription.model_json_schema()}
        ),
        prompt=DESCRIPTION_INSTRUCTION,
        timeout=timeout,
    )
    return options

def openai_options(
    api_key: str,
    model: str,
    timeout: float = 180.0
):
    options = PictureDescriptionApiOptions(
        url="https://api.openai.com/v1/chat/completions",
        headers = {
            'Authorization': f'Bearer {api_key}'
        },
        params=dict(
            model=model,
            seed=42,
            max_completion_tokens=8192,
            # temperature=0.9,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "picture_description",
                    "strict": True,
                    "schema": ImageDescription.model_json_schema()
                }
            }
        ),
        prompt=DESCRIPTION_INSTRUCTION,
        timeout=timeout,
        # percentage of the area for a picture to processed with the models
        picture_area_threshold=0.02
    )
    return options