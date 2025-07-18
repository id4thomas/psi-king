from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    data_dir: str
    
    vlm_base_url: str
    vlm_api_key: str
    vlm_model: str
    
    model_weight_dir: str
    
    docling_artifacts_path: str
    
    # Embedding Models
    text_embedding_base_url: str
    text_embedding_api_key: str
    text_embedding_model: str
    
    multimodal_embedding_base_url: str
    multimodal_embedding_api_key: str
    multimodal_embedding_model: str
    
    qdrant_url: str
    qdrant_port: int
    
settings = Settings()