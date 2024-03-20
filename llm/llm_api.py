import os
from typing import Any

import uvicorn
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from api_schema import MetadataInput, ReasoningInput
from prompt_templates import *

load_dotenv()

app = FastAPI()


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
