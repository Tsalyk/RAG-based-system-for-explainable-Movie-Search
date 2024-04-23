from typing import Any

from fastapi import FastAPI, HTTPException
from googletrans import Translator
from schema import TranslatorInput


app = FastAPI()
translator = Translator()


@app.post("/translate/")
async def translate(data: TranslatorInput) -> dict[str, str]:
    if data.tgt_lang == data.src_lang:
        return {'translation': data.text}
    try:
        translation = translator.translate(data.text, dest=data.tgt_lang).text
        response = {'translation': str(translation)}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
