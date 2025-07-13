from pydantic import BaseModel

class ImageDescription(BaseModel):
    description: str
    text: str
    
    class Config:
        extra='forbid'