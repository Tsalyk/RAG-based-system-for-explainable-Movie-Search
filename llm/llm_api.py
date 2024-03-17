import os
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

import requests

METADATA_TEMPLATE = """
SYSTEM: You are an AI movie search assistant that extracts key
features from user queries.
Your task is to extract information from a user query as input and returns
a dictionary containing this information in structured way.
The dictionary format should have the following keys:
'title': A string representing the title of the movie.
'genre': A string representing the genre of the movie.
'min_year': A string representing the minimal release year of the movie.
'max_year': A string representing the maximal release year of the movie.
'query': The original user query.
If any of the keys cannot be extracted from the user query, they should be
empty strings in the returned dictionary.
You should return only dictionary without any additional information.

Few shots:
USER: Bring me a horror movie from 2010
AI: {{'title': '', 'genre': 'horror', 'min_year': '2010', 'max_year': '',
'query': 'Find me a horror movie from 2010'}}

USER: I don't know what to watch
AI: {{'title': '', 'genre': '', 'min_year': '', 'max_year': '',
'query': 'I don't know what to watch'}}

USER: <query>
AI:
"""

REASONING_TEMPLATE = """
SYSTEM: You are an AI movie search assistant that helps to explain why certain
movie might be relevant for the user.
Your task is to explain relevance of the film given user query and
reccommended movie description.

Follow the guidlines:
1. Reasoning should be clear and comprehensive
2. It should be short and informative
3. Do not reccommended any other movies, provie reasoning for the described
movie

Movie title: <title>
Movie description: <description>

USER: <query>
AI:
"""

app = FastAPI()


class MetadataInput(BaseModel):
    query: str


class ReasoningInput(BaseModel):
    title: str
    description: str
    query: str


@app.get("/")
def status_api_check() -> dict[str, str]:
    return {
        "status": "LLM API IS RUNNING"
    }


@app.post("/extract_metadata/")
async def extract_metadata(data: MetadataInput) -> dict[str, str]:
    params = {
                    "max_tokens": 256,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "repeat_penalty": 1.2,
                    "top_k": 50,
                    "stop": ['USER:'],
                    "echo": False
                    }
    url = f"{os.getenv('LLM_API')}/generate_response"
    prompt = METADATA_TEMPLATE.replace('<query>', data.query)
    body = {
        "prompt": prompt,
        "parameters": params
    }
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            response = response.json()
            response['generated_response'] = response['generated_response'].replace('{{', '{').replace('}}', '}')
            start = response['generated_response'].find('{')
            end = response['generated_response'].find('}')
            response['generated_response'] = response['generated_response'][start:end+1]
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_reasoning/")
async def generate_reasoning(data: ReasoningInput) -> dict[str, str]:
    params = {
                    "max_tokens": 256,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "repeat_penalty": 1.2,
                    "top_k": 50,
                    "stop": ['USER:'],
                    "echo": False
                    }
    url = f"{os.getenv('LLM_API')}/generate_response"
    prompt = REASONING_TEMPLATE.replace('<title>', data.title)\
                               .replace('<description>', data.description)\
                               .replace('<query>', data.query)
    body = {
        "prompt": prompt,
        "parameters": params
    }
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
