from pydantic import BaseModel


class MetadataInput(BaseModel):
    query: str


class ReasoningInput(BaseModel):
    title: str
    description: str
    query: str
