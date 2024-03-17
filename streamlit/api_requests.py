import os

import requests
from dotenv import load_dotenv

load_dotenv()


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
            return {"error":
                    f"Request failed with status code {response.status_code}\nAPI URL: {url}"
                    }
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
            return {"error":
                    f"Request failed with status code {response.status_code}"
                    }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


def search_movies(
        query: str,
        metadata: dict,
        k: int,
        min_similarity_score: float):
    url = f"{os.getenv('DB_API')}/search_movies"
    body = {
        "query": query,
        "metadata": metadata,
        "k": k,
        "min_similarity_score": min_similarity_score
    }
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error":
                    f"Request failed with status code {response.status_code}"
                    }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

