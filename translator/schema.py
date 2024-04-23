from pydantic import BaseModel


class TranslatorInput(BaseModel):
    tgt_lang: str
    src_lang: str
    text: str
