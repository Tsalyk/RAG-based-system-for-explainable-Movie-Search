import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pinecone_db import init_db
from pydantic import BaseModel

load_dotenv()


app = FastAPI()
db = init_db()


class SearchMoviesInput(BaseModel):
    query: str
    metadata: dict
    k: int
    min_similarity_score: float


@app.post("/search_movies/")
async def search_movies(data: SearchMoviesInput) -> dict:
    try:
        search_results = db.search_movies(
                            query=data.query,
                            metadata=data.metadata,
                            k=data.k,
                            min_similarity_score=data.min_similarity_score
                            )
        return {"search_results": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
