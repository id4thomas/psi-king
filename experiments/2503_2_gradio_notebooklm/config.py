from pydantic_settings import BaseSettings, SettingsConfigDict

class ExperimentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env", env_file_encoding="utf-8", extra="ignore"
    )
    data_dir: str
    
    vlm_base_url: str
    vlm_api_key: str
    vlm_model: str
    
    model_weight_dir: str
    
    docling_model_weight_dir: str
    
    openai_embedding_base_url: str
    openai_embedding_api_key: str
    openai_embedding_model: str
    
    qdrant_url: str

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    qdrant_port: int

experiment_settings = ExperimentSettings()
app_settings = AppSettings()