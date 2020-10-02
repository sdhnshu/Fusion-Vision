# from typing import Optional, List
from pydantic import BaseModel, conlist


class Image(BaseModel):
    id: str
    value: str


class Style(BaseModel):
    dims: conlist(float, min_items=512, max_items=512)


class Latent(BaseModel):
    styles: conlist(Style, min_items=18, max_items=18)
