from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env", env_file_encoding="utf-8", extra="ignore"
    )
    data_dir: str
    
    model_weight_dir: str
    
    openai_embedding_base_url: str
    openai_embedding_api_key: str
    openai_embedding_model: str
    
    qdrant_url: str
    
settings = Settings()