import os

import requests


def extract_metadata(query: str):
    url = f"{os.getenv('LLM_API')}/extract_metadata"
    body = {
        "query": query,
        "parameters": {}
    }
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


def generate_reasoning(title: str, description: str, query: str):
    url = f"{os.getenv('LLM_API')}/generate_reasoning"
    body = {
        "title": title,
        "query": query,
        "description": description,
        "parameters": {}
    }
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

