from pydantic import BaseModel

class ImageDescription(BaseModel):
    description: str
    text: str