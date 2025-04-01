import uuid
from typing import List, Dict

from pydantic import BaseModel, Field

class PipelineSettings(BaseModel):
    # PictureDescription
    vlm_base_url: str = Field("")
    vlm_api_key: str = Field("")
    vlm_model: str = Field("")
    
    # PDF2Img
    poppler_path: str = Field("/opt/homebrew/Cellar/poppler/25.01.0/bin")
    
    # Chunking
    text_chunk_size: int = Field(1024)
    text_chunk_overlap: int = Field(128)
    
    output_dir: str = Field("")
    
class EmbeddingSettings(BaseModel):
    # PictureDescription
    embedding_model_path: str = Field("")
    
    dense_batch_size: int = Field(4)
    sparse_batch_size: int = Field(256)

class InputFile(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    metadata: Dict[str, str] = Field(dict())
