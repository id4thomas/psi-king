from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env", env_file_encoding="utf-8", extra="ignore"
    )
    vlm_base_url: str
    vlm_api_key: str
    vlm_model: str
    
    mistral_api_key: str
    
settings=Settings()