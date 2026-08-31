from typing import List, Dict

from pydantic import BaseModel, Field


class TaggerInterrogateRequest(BaseModel):
    image: str = Field(
        default='',
        title='Image',
        description='Base64-encoded image'
    )
    model: str = Field(
        title='Model',
        description='The interrogate model used.'
    )

    threshold: float = Field(
        default=0.35,
        title='Threshold',
        description='',
        ge=0,
        le=1
    )


class TaggerInterrogateResponse(BaseModel):
    caption: Dict[str, float] = Field(
        title='Caption',
        description='The generated caption for the image.'
    )


class InterrogatorsResponse(BaseModel):
    models: List[str] = Field(
        title='Models',
        description=''
    )
