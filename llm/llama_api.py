from typing import Any

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel


print('Llama loading started.')

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

llama_model = Llama(
    model_path=model_path,
    n_threads=2,
    n_batch=128,
    n_gpu_layers=43,
    n_ctx=1024
)
print('Llama is loaded.')


METADATA_TEMPLATE = f"""
SYSTEM: You are an AI movie search assistant that extracts key features from user queries.
Your task is to extract information from a user query as input and returns a dictionary containing this information in structured way.
The dictionary format should have the following keys:
'title': A string representing the title of the movie.
'genre': A string representing the genre of the movie.
'min_year': A string representing the minimal release year of the movie.
'max_year': A string representing the maximal release year of the movie.
'query': The original user query.
If any of the keys cannot be extracted from the user query, they should be empty strings in the returned dictionary.
You should return only dictionary without any additional information.

Few shots:
USER: Bring me a horror movie from 2010
AI: {{'title': '', 'genre': 'horror', 'min_year': '2010', 'max_year': '', 'query': 'Find me a horror movie from 2010'}}

USER: I don't know what to watch
AI: {{'title': '', 'genre': '', 'min_year': '', 'max_year': '', 'query': 'I don't know what to watch'}}

USER: <query>
AI:
"""

REASONING_TEMPLATE = f"""
SYSTEM: You are an AI movie search assistant that helps to explain why certain movie might be relevant for the user.
Your task is to explain relevance of the film given user query and reccommended movie description.

Follow the guidlines:
1. Reasoning should be clear and comprehensive
2. It should be short and informative
3. Do not reccommended any other movies, provie reasoning for the described movie

Movie title: <title>
Movie description: <description>

USER: <query>
AI:
"""

app = FastAPI()

class MetadataInput(BaseModel):
    query: str
    parameters: dict[str, Any] | None

class ReasoningInput(BaseModel):
    title: str
    description: str
    query: str
    parameters: dict[str, Any] | None


@app.get("/")
def status_gpu_check() -> dict[str, str]:
    gpu_msg = "Available" if tf.test.is_gpu_available() else "Unavailable"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }

@app.post("/extract_metadata/")
async def extract_metadata(data: MetadataInput) -> dict[str, str]:
    try:
        params = data.parameters or {}
        params = {
                    "max_tokens": 256,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "repeat_penalty": 1.2,
                    "top_k": 50,
                    "stop": ['USER:', '\n\n'],
                    "echo": False
                    }
        prompt = METADATA_TEMPLATE.replace('<query>', data.query)
        response = llama_model(prompt=prompt, **params)
        model_out = response['choices'][0]['text']
        return {"generated_text": model_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_reasoning/")
async def generate_response(data: ReasoningInput) -> dict[str, str]:
    try:
        params = data.parameters or {}
        params = {
                    "max_tokens": 256,
                    "temperature": 0.5,
                    "top_p": 0.95,
                    "repeat_penalty": 1.2,
                    "top_k": 50,
                    "stop": ['USER:'],
                    "echo": False
                    }
        prompt = REASONING_TEMPLATE.replace('<title>', data.title).replace('<description>', data.description).replace('<query>', data.query)
        response = llama_model(prompt=prompt, **params)
        model_out = response['choices'][0]['text']
        return {"generated_text": model_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
