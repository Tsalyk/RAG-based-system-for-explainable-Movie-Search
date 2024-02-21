# Pinecone vector database

import ast
import os
from dotenv import load_dotenv
load_dotenv()
import uuid
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter


class VectorDB:
    def __init__(self, api_key: str, dimension: int=384):
        self.pc = Pinecone(api_key)
        self.vector_dim = dimension

    def create_index(self, index_name: str, metric: str='cosine'):
        self.pc.create_index(
            index_name,
            dimension=self.vector_dim,
            metric=metric,
            spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                ) 
            )

    def get_movies_index(self):
        return self.pc.Index("movies")

    def set_index(self):
        self.index = self.get_movies_index()

    def upsert_movie(self, title: str, genres: list, year: int, description: str):
        year = int(year)
        descriptions = self.chunk_description(description)
        genres = list(map(lambda g: g.lower(), genres))
        metadata = {'title': title, 'year': year, 'genres': genres}
        movie_id = str(uuid.uuid4())

        for description in descriptions:
            record_id = str(uuid.uuid4())
            embeddings = self.get_embedding(description)
            self.index.upsert([(record_id, embeddings, metadata | {'description': description, 'movie_id': movie_id})])

    def search_movies(self, query: str, metadata: dict, k=10, min_similarity_score=0.25) -> list:
        query_embedding = self.get_embedding(query)

        # Applying filters
        filters = {}
        genre = metadata.get('genre', '')
        if genre:
            filters['genre'] = genre.lower()
        year = int(metadata.get('year', '0'))
        if year:
            filters['year'] = int(year)

        search_results = self.index.query(
                    vector=query_embedding,
                    filter=filters,
                    top_k=k,
                    include_values=False
                    )

        movies = [{
            'movie_id': self.get_movie_id(search['id']),
            'title': self.get_movie_title(search['id']),
            'genres': self.get_movie_genres(search['id']),
            'description': self.get_movie_description(search['id']),
            'score': search['score']
            } for search in search_results['matches'] if search['score'] > min_similarity_score]

        return movies[:k]

    @staticmethod
    def get_embedding(text: str) -> list:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(text, convert_to_tensor=False)
        return embeddings.tolist()

    @staticmethod
    def chunk_description(description: str) -> list:
        text_splitter = CharacterTextSplitter(
                            separator="\n",
                            chunk_size=1024,
                            chunk_overlap=20
                        )
        docs = text_splitter.create_documents([description])
        docs = [str(doc) for doc in docs]
        return docs

    def get_movie_title(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('title', '')

    def get_movie_genres(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('genres', '')

    def get_movie_description(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('description', '')
    
    def get_movie_id(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('movie_id', '')

    def get_record_id(self, title: str) -> list:
        metadata = {'title': title}
        sr = self.index.query(
                    vector=list(range(self.vector_dim)),
                    filter=metadata,
                    top_k=1,
                    include_values=False
                    )
        return sr['matches'][0]['id']

    def get_metadata(self, file_id: str) -> dict:
        return self.index.fetch([file_id]).vectors[file_id]['metadata']

    def delete_file(self, filename: str, space_id: str, tenant_id: str):
        file_id = self.get_file_id(filename, space_id, tenant_id)
        self.index.delete(ids=[file_id], namespace='')

    def remove_index(self, index_name: str):
        self.pc.delete_index(index_name)

def upsert_all(df: pd.DataFrame, db: VectorDB):
    for _, row in tqdm(df.iterrows()):
        movie = dict(row)
        db.upsert_movie(**movie)


if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    df['genres'] = df['genres'].apply(ast.literal_eval)

    PINECONE_KEY = os.getenv('PINECONE_KEY')
    db = VectorDB(PINECONE_KEY)
    upsert_all(df, db)
