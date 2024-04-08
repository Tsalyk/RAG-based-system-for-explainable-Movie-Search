import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pgvector import VectorDB
from populate_db import init_db, populate_all

load_dotenv()


app = FastAPI()
db = VectorDB()
populate_all()


class SearchMoviesInput(BaseModel):
    chunking_strategy: str
    embedding_model: str
    query: str
    metadata: dict
    k: int
    min_similarity_score: float


@app.post("/search_movies/")
async def search_movies(data: SearchMoviesInput) -> dict:
    try:
        search_results = db.search_movies(
                            chunking_strategy=data.chunking_strategy,
                            embedding_model=data.embedding_model,
                            query=data.query,
                            metadata=data.metadata,
                            k=data.k,
                            min_similarity_score=data.min_similarity_score
                            )
        return {"search_results": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
