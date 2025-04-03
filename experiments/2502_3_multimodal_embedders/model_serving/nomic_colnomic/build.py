import os

import bentoml
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../../../.env", env_file_encoding="utf-8", extra="ignore"
    )
    data_dir: str
    
    model_weight_dir: str
    
    openai_embedding_base_url: str
    openai_embedding_api_key: str
    openai_embedding_model: str
    
    qdrant_url: str
    
settings = Settings()

pretrained_path = os.path.join(
    settings.model_weight_dir, "Qwen2.5-VL-3B-Instruct"
)
adapter_path = os.path.join(
    settings.model_weight_dir, "embedding/colnomic-embed-multimodal-3b"
)

# Load Model
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor



# Save the model artifact using save_model()
# bentoml.models.save_model(
#     "colnomic_3b",
#     model,
#     metadata={
#         "adapter_path": adapter_path,
#         "description": "nomic-ai/colnomic-embed-multimodal-3b",
#     },
#     labels={"framework": "colnomic"},
# )

with bentoml.models.create(name="colnomic-embed-multimodal-3b") as model_ref:
    print(model_ref)
    print(model_ref.path)
    model = ColQwen2_5.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()

    # Load the adapter
    model.load_adapter(adapter_path)
    # model.merge_and_unload().save_pretrained(model_ref.path)
    model.save_pretrained(model_ref.path)
    print("Model successfully built.")
    
    processor = ColQwen2_5_Processor.from_pretrained(adapter_path)
    processor.save_pretrained(model_ref.path)
    print("Preprocessor successfully built.")
    

# You can include the adapter path in custom_fields
# bentoml.models.create(
#     "colnomic_model",
#     # custom_fields={"adapter_path": adapter_path},
#     metadata={
#         "description": "colnomic model",
#         "adapter_path": adapter_path
#     },
#     labels={"framework": "colnomic"},
#     path=pretrained_path,
# )